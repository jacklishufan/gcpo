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
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import torch
import yaml
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from ..workers.config import DataConfig


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


def _normalize_eval_dataset_specs(config: DataConfig) -> list[dict[str, Optional[str]]]:
    if config.eval_dataset_config is None:
        return [
            {
                "name": "val",
                "eval_json_files": config.eval_json_files,
                "eval_image_root": config.eval_image_root,
            }
        ]

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


def _create_val_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    dataset_spec: dict[str, Optional[str]],
) -> StatefulDataLoader:
    eval_json_files = dataset_spec["eval_json_files"]
    eval_image_root = dataset_spec["eval_image_root"]

    val_prompt_key = config.val_prompt_key or config.prompt_key
    val_answer_key = config.val_answer_key or config.answer_key

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
            max_val_samples=config.max_val_samples,
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
            max_val_samples=config.max_val_samples,
            use_importance_weighting=config.use_importance_weighting,
            negative_prompt_type=config.negative_prompt_type,
            negative_format_prompt=config.negative_format_prompt,
        )

    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size
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
    print(f"Size of {dataset_spec['name']} val dataloader: {len(val_dataloader)}")
    return val_dataloader


def create_val_dataloaders(
    config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]
) -> StatefulDataLoader | dict[str, StatefulDataLoader]:
    dataset_specs = _normalize_eval_dataset_specs(config)
    if len(dataset_specs) == 1 and dataset_specs[0]["name"] == "val":
        return _create_val_dataloader(config, tokenizer, processor, dataset_specs[0])

    return {spec["name"]: _create_val_dataloader(config, tokenizer, processor, spec) for spec in dataset_specs}


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
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
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        use_importance_weighting=config.use_importance_weighting,
        negative_prompt_type=config.negative_prompt_type,
        negative_format_prompt=config.negative_format_prompt,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataloader = create_val_dataloaders(config, tokenizer, processor)

    assert len(train_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader, val_dataloader
