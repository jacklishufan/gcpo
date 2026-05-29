#!/usr/bin/env bash

# Environment variables
export DEBUG_MODE=true
export LOG_PATH=./outputs/debug.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT=<REPO_ROOT>/t2i

RUN_NAME="gcpo"
QWEN_PATH="<LOCAL_PATH>/Janus-Pro-7B"
HF_DATASET="$ROOT/data/janus_geneval_prompts/evaluation_metadata_flow_grpo.json"
OUTPUT_DIR="<LOCAL_PATH>/janus-grpo/gcpo-geneval-run-1"

# Change directory
cd $ROOT/src/grpo/src || exit 1

# More environment variables
export WANDB_PROJECT="grpo-janus"
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"

# Run command
# lr_scheduler_type
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12346 \
  open_r1/grpo.py \
  --use_vllm False \
  --deepspeed ../configs/zero3.json \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$QWEN_PATH" \
  --semantic_cot False \
  --dataset_name "$HF_DATASET" \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --num_generations 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --bf16 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing false \
  --attn_implementation flash_attention_2 \
  --max_steps 1600 \
  --run_name "$RUN_NAME" \
  --save_steps 100 \
  --new_generations_image 16 \
  --image_token_num_per_image 576 \
  --cfg_weight 5 \
  --reasoning_prompt_path ../../../data/prompt/reasoning_prompt.txt \
  --reward_funcs geneval \
  --beta 0.03 \
  --tf32 true \
  --learning_rate 5e-6 \
  --reward_ckpt_path_file $ROOT/src/grpo/configs/reward_paths_ssci.json \
  --reward_smooth True \
  --kl_reweight True \
  --update_ref False \
  --progress_learning False \
  --add_noise False \
  --entropy_reward True \
  --use_importance_sampling True \
  --importance_normalization "hist" \
  --importance_weighting_type kl \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate":0.05}' \
