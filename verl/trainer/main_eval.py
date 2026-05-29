# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional, Union
from urllib.parse import urljoin, urlparse

import numpy as np
import ray
import torch
import yaml
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import ProcessorMixin, PreTrainedTokenizer

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..utils.dataset import RLHFDataset, collate_fn, process_image, process_video
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, unflatten_dict
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.vllm_utils import VLLMHijack
from ..workers.reward import AutoRewardManager
from ..workers.rollout.config import RolloutConfig
from .config import PPOConfig
from .metrics import compute_length_metrics, reduce_metrics


PAPO_COT_INSTRUCTION = (
    "\n\nYou first think through the reasoning process as an internal monologue, enclosed within <think> </think> "
    "tags. Then, provide your final answer enclosed within \\boxed{}."
)


def _safe_metric_name(name: str) -> str:
    return str(name).replace("/", "_")


def _resolve_path(path: str, base_dir: Optional[str], image_root: Optional[str]) -> str:
    if urlparse(path).scheme in ("http", "https"):
        return path
    if os.path.isabs(path):
        return path

    candidates = []
    url_candidate = None
    if base_dir is not None:
        candidates.append(os.path.join(base_dir, path))
    if image_root is not None:
        if urlparse(image_root).scheme in ("http", "https"):
            url_candidate = urljoin(image_root.rstrip("/") + "/", path.lstrip("./"))
        else:
            candidates.extend([os.path.join(image_root, path), os.path.join(image_root, "data", path)])
    candidates.append(path)

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    if url_candidate is not None:
        return url_candidate

    return os.path.abspath(candidates[0])


def _normalize_eval_dataset_specs(config) -> list[dict[str, str]]:
    if config.eval_dataset_config is None:
        if config.eval_json_files is not None:
            return [
                {
                    "name": _safe_metric_name(os.path.splitext(os.path.basename(config.eval_json_files))[0]),
                    "eval_json_files": config.eval_json_files,
                    "eval_image_root": config.eval_image_root,
                }
            ]

        return [{"name": "val", "eval_json_files": None, "eval_image_root": config.image_dir}]

    with open(config.eval_dataset_config, encoding="utf-8") as f:
        raw_specs = yaml.safe_load(f)

    if raw_specs is None:
        raise ValueError(f"Eval dataset config is empty: {config.eval_dataset_config}")

    if isinstance(raw_specs, list):
        items = [(None, item) for item in raw_specs]
    elif isinstance(raw_specs, dict):
        if "eval_json_files" in raw_specs or "file_name" in raw_specs:
            items = [(raw_specs.get("name"), raw_specs)]
        else:
            items = list(raw_specs.items())
    else:
        raise ValueError("Eval dataset config must be a list or dict.")

    base_dir = os.path.dirname(config.eval_dataset_config)
    specs = []
    for index, (key, item) in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Eval dataset entry {key or index} must be a dict.")

        name = item.get("name") or key or f"dataset_{index}"
        eval_json_files = item.get("eval_json_files") or item.get("file_name")
        if eval_json_files is None:
            raise ValueError(f"Eval dataset entry {name} must define `eval_json_files` or `file_name`.")

        eval_image_root = item.get("eval_image_root", config.eval_image_root)
        if eval_image_root is not None:
            eval_image_root = _resolve_path(eval_image_root, base_dir, None)
        specs.append(
            {
                "name": _safe_metric_name(name),
                "eval_json_files": _resolve_path(eval_json_files, base_dir, eval_image_root),
                "eval_image_root": eval_image_root,
            }
        )

    return specs


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)

    return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}

    return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> Optional[dict[str, Any]]:
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a list of eval examples.")

    return data


def _strip_papo_cot_instruction(text: str) -> str:
    return text.replace(PAPO_COT_INSTRUCTION, "")


def _load_papo_records(data_path: str) -> list[dict[str, Any]]:
    if "@" in data_path:
        data_path, _ = data_path.split("@", maxsplit=1)

    if os.path.isdir(data_path):
        data_files = [
            os.path.join(data_path, filename)
            for filename in sorted(os.listdir(data_path))
            if filename.endswith((".json", ".jsonl"))
        ]
    else:
        data_files = [data_path]

    records = []
    for data_file in data_files:
        records.extend(_load_json_or_jsonl(data_file))

    return records


def _normalize_papo_record(
    example: dict[str, Any], prompt_key: str, answer_key: str, image_key: str, video_key: str
) -> dict[str, Any]:
    messages = example.get("messages")
    if not messages or len(messages) != 2:
        raise ValueError("PAPO eval examples are expected to contain exactly two messages.")

    user_message, assistant_message = messages
    if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
        raise ValueError("PAPO eval examples are expected to be one user turn followed by one assistant turn.")

    record = {
        prompt_key: _strip_papo_cot_instruction(user_message.get("content", "")),
        answer_key: assistant_message.get("content", ""),
    }
    if "images" in example:
        record[image_key] = example["images"]
    if "videos" in example:
        record[video_key] = example["videos"]
    if "id" in example:
        record["id"] = example["id"]

    return record


class EvalRolloutWorker:
    """vLLM-only rollout worker for evaluating the initial model."""

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        from vllm import LLM, SamplingParams

        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__

        VLLMHijack.hijack()

        engine_kwargs = {}
        if processor is not None:
            engine_kwargs["mm_processor_cache_gb"] = 0
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **engine_kwargs,
        )

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    if key == "eos_token_id":
                        self.sampling_params.update_from_generation_config(
                            generation_config={"eos_token_id": value},
                            eos_token_id=value if isinstance(value, int) else None,
                        )
                    else:
                        old_value = getattr(self.sampling_params, key)
                        old_sampling_params_args[key] = old_value
                        setattr(self.sampling_params, key, value)

        yield

        for key, value in old_sampling_params_args.items():
            if key == "eos_token_id":
                self.sampling_params.update_from_generation_config(
                    generation_config={"eos_token_id": value},
                    eos_token_id=value if isinstance(value, int) else None,
                )
            else:
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        prompts.meta_info.update(
            {
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("Batch size does not match raw prompt ids.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        with self.update_sampling_params(**prompts.meta_info):
            completions = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=not self.config.disable_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            output_non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            output_non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=output_non_tensor_batch, meta_info=prompts.meta_info).to("cpu")


def create_val_dataloader(config, tokenizer, processor, dataset_spec: Optional[dict[str, str]] = None) -> StatefulDataLoader:
    eval_json_files = config.eval_json_files
    eval_image_root = config.eval_image_root
    if dataset_spec is not None:
        eval_json_files = dataset_spec["eval_json_files"]
        eval_image_root = dataset_spec["eval_image_root"]

    val_prompt_key = getattr(config, "val_prompt_key", None) or config.prompt_key
    val_answer_key = getattr(config, "val_answer_key", None) or config.answer_key

    if eval_json_files is not None:
        records = [
            _normalize_papo_record(example, val_prompt_key, val_answer_key, config.image_key, config.video_key)
            for example in _load_papo_records(eval_json_files)
        ]
        val_dataset = RLHFDataset(
            data_path=eval_json_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=val_prompt_key,
            answer_key=val_answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=eval_image_root,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
            max_val_samples=getattr(config, "max_val_samples", -1),
            use_importance_weighting=config.use_importance_weighting,
            negative_prompt_type=config.negative_prompt_type,
            negative_format_prompt=config.negative_format_prompt,
            records=records,
        )
    else:
        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=val_prompt_key,
            answer_key=val_answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            max_val_samples=getattr(config, "max_val_samples", -1),
            use_importance_weighting=config.use_importance_weighting,
            negative_prompt_type=config.negative_prompt_type,
            negative_format_prompt=config.negative_format_prompt,
        )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(val_dataloader) >= 1
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return val_dataloader


def maybe_log_val_generations(
    logger: Tracker,
    config: PPOConfig,
    inputs: list[str],
    outputs: list[str],
    labels: list[str],
    scores: list[float],
) -> None:
    if config.trainer.val_generations_to_log <= 0:
        return

    samples = list(zip(inputs, outputs, labels, scores))
    samples.sort(key=lambda x: x[0])

    rng = np.random.RandomState(42)
    rng.shuffle(samples)

    samples = samples[: config.trainer.val_generations_to_log]
    logger.log_generation(samples, step=0)


def validate(
    config: PPOConfig,
    tokenizer,
    val_dataloader,
    rollout_worker,
    reward_manager,
    logger,
    dataset_name: str = "val",
) -> dict:
    reward_tensor_lst = []
    sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
    reward_metrics_lst = defaultdict(list)
    length_metrics_lst = defaultdict(list)

    print("Start validation...")
    for batch_dict in val_dataloader:
        test_batch = DataProto.from_single_dict(batch_dict)
        test_gen_batch = test_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
        )
        repeat_times = config.worker.rollout.val_override_config.get("n", config.worker.rollout.n)
        test_gen_batch.meta_info = dict(config.worker.rollout.val_override_config)
        test_gen_batch.meta_info["min_pixels"] = config.data.min_pixels
        test_gen_batch.meta_info["max_pixels"] = config.data.max_pixels
        test_gen_batch.meta_info["video_fps"] = config.data.video_fps

        test_output_gen_batch = ray.get(rollout_worker.generate_sequences.remote(test_gen_batch))

        test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
        test_batch = test_batch.union(test_output_gen_batch)

        reward_tensor, reward_metrics = reward_manager.compute_reward(test_batch)

        input_ids = test_batch.batch["prompts"]
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        output_ids = test_batch.batch["responses"]
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_inputs.extend(input_texts)
        sample_outputs.extend(output_texts)
        sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
        sample_scores.extend(scores)

        reward_tensor_lst.append(reward_tensor)
        for key, value in reward_metrics.items():
            reward_metrics_lst[key].extend(value)

        for key, value in compute_length_metrics(test_batch).items():
            length_metrics_lst[key].append(value)

    maybe_log_val_generations(logger, config, sample_inputs, sample_outputs, sample_labels, sample_scores)
    val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
    metric_prefix = f"val/{dataset_name}"
    val_reward_metrics = {f"{metric_prefix}/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
    val_length_metrics = {f"{metric_prefix}/{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
    print("Finish validation.")
    return {f"{metric_prefix}/reward_score": val_reward_score, **val_reward_metrics, **val_length_metrics}


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for validation only."""

    def run(self, config: PPOConfig):
        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        logger = Tracker(loggers=config.trainer.logger, config=config.to_dict())
        reward_manager = AutoRewardManager(config.worker.reward, tokenizer)

        RolloutWorker = ray.remote(EvalRolloutWorker)
        rollout_worker = RolloutWorker.options(
            num_cpus=1,
            num_gpus=config.worker.rollout.tensor_parallel_size,
        ).remote(
            config.worker.actor.model.model_path,
            config.worker.rollout,
            tokenizer,
            processor,
        )

        all_metrics = {}
        for dataset_spec in _normalize_eval_dataset_specs(config.data):
            dataset_name = dataset_spec["name"]
            print(f"Start evaluating dataset: {dataset_name}")
            val_dataloader = create_val_dataloader(config.data, tokenizer, processor, dataset_spec)
            val_metrics = validate(
                config,
                tokenizer,
                val_dataloader,
                rollout_worker,
                reward_manager,
                logger,
                dataset_name=dataset_name,
            )
            logger.log(data=val_metrics, step=0)
            all_metrics.update(val_metrics)
            print(f"Validation metrics for {dataset_name}:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if len(all_metrics) > 1:
            print(f"All validation metrics:\n{convert_dict_to_str(unflatten_dict(all_metrics))}")


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
