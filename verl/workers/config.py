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
ActorRolloutRef config
"""

from dataclasses import dataclass, field

from ..utils.py_functional import get_abs_path
from .actor import ActorConfig, FSDPConfig, LoraConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward import RewardConfig
from .rollout import RolloutConfig

from typing import Optional
@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    eval_json_files: Optional[str] = None
    """optional message-format JSON/JSONL eval data used by main_eval.py"""
    eval_image_root: Optional[str] = None
    """root directory for relative image/video paths in eval_json_files"""
    eval_dataset_config: Optional[str] = None
    """optional YAML eval suite config used by main_eval.py"""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    val_prompt_key: Optional[str] = None
    """Override prompt_key for the validation dataset. Falls back to prompt_key when None."""
    val_answer_key: Optional[str] = None
    """Override answer_key for the validation dataset. Falls back to answer_key when None."""
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    max_val_samples: int = -1
    # Importance weighting config
    use_importance_weighting: bool = False
    importance_weighting_type: str = "kl"
    negative_prompt_type: str = "wrong_answer"
    negative_format_prompt: Optional[str] = None
    importance_normalization: str = "softmax"
    importance_normalization_temperature: float = 0.1

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.eval_json_files = get_abs_path(self.eval_json_files, prompt="Eval JSON files")
        self.eval_image_root = get_abs_path(self.eval_image_root, prompt="Eval image root")
        self.eval_dataset_config = get_abs_path(self.eval_dataset_config, prompt="Eval dataset config")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.negative_format_prompt = get_abs_path(self.negative_format_prompt, prompt="Negative format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DataConfig",
    "FSDPConfig",
    "LoraConfig",
    "ModelConfig",
    "OptimConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.ref.micro_batch_size_per_device_for_experience = self.actor.micro_batch_size_per_device_for_experience
        self.ref.padding_free = self.actor.padding_free
        self.ref.dynamic_batching = self.actor.dynamic_batching
        self.ref.ulysses_size = self.actor.ulysses_size
        self.ref.use_torch_compile = self.actor.use_torch_compile
