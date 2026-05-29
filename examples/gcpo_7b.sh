#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

# export CUDA_VISIBLE_DEVICES=0,5,6,7
N_GPU=8

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct


MAX_STEPS=130
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
rollout=8
TOTAL_EPOCHES=2


top_p_perception_tokens=0.4
advantage_scaling_min=0.9
entropy_penalty_coef=0.02


EXP_NAME="rank_kl_importance_wrong_answer_perc${top_p_perception_tokens}_advsc${advantage_scaling_min}_pen${entropy_penalty_coef}_step${MAX_STEPS}_rollout${rollout}"

CONGI_FILE="examples/configs/config_vlm.yaml"
TRAIN_FILE="chamber111/VPPO_ViRL39K_train"
VAL_FILE="chamber111/VPPO_MMK12_validation"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"

python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger=['file','wandb'] \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.project_name="qwen25_7b_vppo" \
    worker.rollout.n=${rollout} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    data.use_importance_weighting=true \
    data.importance_weighting_type="kl" \
    data.negative_prompt_type="wrong_answer" \
    data.importance_normalization="hist"  \
    algorithm.entropy_penalty_coef=${entropy_penalty_coef} \
    trainer.val_before_train=False \
    data.negative_format_prompt=examples/format_prompt/math_format_perception_wrong_answer.jinja \

