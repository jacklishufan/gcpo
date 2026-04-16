#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98


N_GPU=8

MODEL_PATH=/home/schmidt/ssci-shufan/scratch_ssci-adityag/Qwen2.5-VL-7B-Instruct


MAX_STEPS=130
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
rollout=8
TOTAL_EPOCHES=2


top_p_perception_tokens=0.4
advantage_scaling_min=0.9
entropy_penalty_coef=0.12


EXP_NAME="perc${top_p_perception_tokens}_advsc${advantage_scaling_min}_pen${entropy_penalty_coef}_step${MAX_STEPS}_rollout${rollout}"

CONGI_FILE="examples/configs/config_vrpo.yaml"
TRAIN_FILE="chamber111/VPPO_ViRL39K_train"
VAL_FILE="chamber111/VPPO_MMK12_validation"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"
#     trainer.logger=['console','swanlab'] \
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
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    # data.max_val_samples=1000\
    # data.max_response_length=8192 \
    # worker.rollout.max_num_batched_tokens=12289 \
    # trainer.max_steps=${MAX_STEPS}

#        algorithm.advantage_scaling_min=${advantage_scaling_min} \
    #     algorithm.use_advantage_shaping=True \
    # algorithm.use_entropy_penalty=True \