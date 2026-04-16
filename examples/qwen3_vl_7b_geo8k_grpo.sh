#!/bin/bash

set -x

MODEL_PATH=/home/schmidt/ssci-shufan/scratch_ssci-adityag/Qwen3-VL-8B-Instruct  # replace it with your local file path
# CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=leonardPKU/GEOQA_8K_R1V@train \
    data.val_files=leonardPKU/GEOQA_8K_R1V@test \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_8b_geoqa8k \
    trainer.n_gpus_per_node=8
