#!/bin/bash

set -x

MODEL_PATH=/home/schmidt/ssci-shufan/scratch_ssci-adityag/Qwen3-VL-8B-Instruct  # replace it with your local file path
# CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=8
