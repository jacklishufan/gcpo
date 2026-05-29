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
"""
Implement Actor
"""
import math
import re
import torch.nn.functional as F
import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig
from .importance_weights_viz import render_html_heatmap


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        data_config: Optional[Any] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_config = data_config
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float, return_full_logits: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            log_probs: # (bs, response_len)
            full_response_log_probs: # (bs, response_len, vocab_size) if return_full_logits=True, else None
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            full_log_probs_rmpad = F.log_softmax(logits_rmpad, dim=-1)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                full_log_probs_rmpad = gather_outputs_and_unpad(full_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen) for per-token log probs
            full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            # For padding-free, do not reconstruct full vocab tensor. Return None.
            full_response_log_probs = None
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            full_response_log_probs = F.log_softmax(logits, dim=-1) if return_full_logits else None
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs, full_response_log_probs

    def create_white_image(self):
        return Image.new('RGB', (224, 224), (255, 255, 255))

    @torch.no_grad()
    def compute_negative_log_probs(self, model_inputs, temperature, return_full_logits: bool = False):
        # input_ids_aug has already been extended with responses by the trainer before this call,
        # so it is a full (negative_prompt + responses) sequence matching the shape of input_ids.
        negative_model_inputs = {
            "input_ids": model_inputs["input_ids_aug"],
            "attention_mask": model_inputs["attention_mask_aug"],
            "position_ids": model_inputs["position_ids_aug"],
            "responses": model_inputs["responses"],
            "multi_modal_inputs": model_inputs.get("multi_modal_inputs", {}),
        }

        if self.data_config.negative_prompt_type == "empty_image":
            device = negative_model_inputs["input_ids"].device
            batch_size = negative_model_inputs["input_ids"].shape[0]
            for i in range(batch_size):
                if "multi_modal_inputs" in negative_model_inputs and negative_model_inputs["multi_modal_inputs"][i]:
                    white_img = self.create_white_image()
                    processed = self.processor.image_processor(images=[white_img], return_tensors="pt")
                    processed = {k: v.to(device) for k, v in processed.items()}
                    negative_model_inputs["multi_modal_inputs"][i] = dict(processed)

        negative_prompt_logps, negative_full_response_log_probs = self._forward_micro_batch(negative_model_inputs, temperature, return_full_logits=return_full_logits)
        return negative_prompt_logps,negative_full_response_log_probs

    @torch.no_grad()
    def compute_importance_weights(self, answer_token_log_probs, negative_answer_token_log_probs, response_mask=None):
        # answer_token_log_probs: (bs, response_length) - log prob of true response token
        # negative_answer_token_log_probs: (bs, response_length) - log prob of true response token under negative prompt
        # response_mask: (bs, response_length) - mask for valid response tokens (1 for valid, 0 for padding)

        if self.data_config.importance_weighting_type == "kl":
            # Pointwise KL using stable low-variance estimator (Schulman 2020)
            # KL(q || p) = E_q[log q - log p] ≈ e^(log p - log q) - (log p - log q) - 1
            kl_diff = (negative_answer_token_log_probs - answer_token_log_probs).clamp(-20.0, 20.0)
            divergences = (kl_diff.exp() - kl_diff - 1).contiguous()
            divergences = torch.clamp(divergences, min=-10.0, max=10.0)
        else:
            raise NotImplementedError(f"Unknown importance_weighting_type: {self.data_config.importance_weighting_type}")

        # normalize
        if self.data_config.importance_normalization.startswith("hist"):
            x = 40 / 100.0  # top fraction (e.g. 0.20)
            y = 80 / 100.0  # target mass  (e.g. 0.80)
            alpha = math.log(1.0 - y) / math.log(1.0 - x) - 1.0
            weights = torch.zeros_like(divergences)
            bs = divergences.size(0)
            for i in range(bs):
                valid_mask = response_mask[i].bool() if response_mask is not None else torch.ones(divergences.size(1), dtype=torch.bool, device=divergences.device)
                valid_divs = divergences[i][valid_mask]
                N = valid_divs.size(0)
                if N == 0:
                    continue
                # Rank lowest=1, highest=N; convert to percentile in (0, 1]
                order = valid_divs.argsort()
                ranks = torch.empty(N, dtype=divergences.dtype, device=divergences.device)
                ranks[order] = torch.arange(1, N + 1, dtype=divergences.dtype, device=divergences.device)
                percentiles = ranks / N  # in (1/N, 1]
                # Power-law transform; renormalize to sum = N
                w = percentiles.pow(alpha)
                w = w / w.sum() * N
                weights[i][valid_mask] = w
        else:
            raise ValueError("Unknown Normalizaion")

        return weights

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, _ = self._forward_micro_batch(model_inputs, temperature=temperature, return_full_logits=False)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        
        # Add augmented inputs for importance weighting
        if self.data_config.use_importance_weighting:
            select_keys.extend(["input_ids_aug", "attention_mask_aug", "position_ids_aug"])
        
        non_tensor_select_keys = ["multi_modal_inputs", "problem"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        global_step = 0
        viz_samples = []  # Collect visualization samples
        
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_idx, micro_batch in enumerate(micro_batches):
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    responses = model_inputs["responses"]

                    # all return: (bsz, response_length) for log_probs
                    log_probs, _ = self._forward_micro_batch(model_inputs, temperature=temperature, return_full_logits=False)

                    if self.data_config.use_importance_weighting:
                        # Compute negative prompt log probs
                        negative_log_probs, _ = self.compute_negative_log_probs(model_inputs, temperature, return_full_logits=False)
                        # Compute per-token weights using only per-token log probs (no full vocab tensor)
                        weights = self.compute_importance_weights(log_probs, negative_log_probs, response_mask)
                        advantages = advantages * weights
                        
                        # Generate visualization HTML periodically (every 50 steps, rank 0 only)
                        if self.rank == 0 and global_step % 50 == 0 and len(viz_samples) < 1:
                            # Generate HTML for first sample from this micro-batch
                            # try:
                                batch_idx = 0
                                num_valid = int(response_mask[batch_idx].sum().item())
                                if num_valid > 0:
                                    # Extract response tokens
                                    response_ids = responses[batch_idx][:num_valid]
                                    weight_values = weights[batch_idx][:num_valid].cpu().detach().numpy().tolist()
                                    tokens = self.tokenizer.convert_ids_to_tokens(response_ids.cpu().detach().tolist())
                                    answer_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                                    
                                    # Get problem text
                                    problem = self.tokenizer.decode(model_inputs['input_ids'][0], skip_special_tokens=True)
                                    negative_prompt = self.tokenizer.decode(model_inputs['input_ids_aug'][0], skip_special_tokens=True)

                                    # if isinstance(problem, list):
                                    #     problem = problem[batch_idx] if batch_idx < len(problem) else ""
                                    # problem = str(problem) if problem else ""
                                    
                                    # Generate HTML
                                    html_content = render_html_heatmap(
                                        tokens=tokens,
                                        weights=weight_values,
                                        question=problem,
                                        answer=answer_text,
                                        negative_prompt=negative_prompt,
                                        metric_name=self.data_config.importance_weighting_type,
                                    )
                                    viz_samples.append(html_content)
                            # except Exception as e:
                            #     # Silently skip visualization on error
                            #     pass

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    if self.config.entropy_penalty_coef > 0.0:
                        # Use entropy penalty for training
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)
                        loss = loss + entropy_loss * self.config.entropy_penalty_coef
                        metrics["actor/entropy_penalty_coef"] = self.config.entropy_penalty_coef

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                    global_step += 1

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        
        # Add visualization HTML to metrics if importance weighting is enabled
        if self.data_config.use_importance_weighting and viz_samples:
            metrics["importance_weights_viz_html"] = viz_samples[0]  # Return first HTML string

        return metrics
