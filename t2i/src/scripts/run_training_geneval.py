import os
import subprocess

os.environ["DEBUG_MODE"] = "true"
os.environ["LOG_PATH"] = "./outputs/debug.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,8,9"

run_name = "your-run-name"
qwen_path = "data1/jacklishufan/Janus-Pro-7B"
hf_dataset = "data1/jacklishufan/STAGE/data/janus_geneval_prompts/evaluation_metadata_flow_grpo.json"
output_dir = f"data1/jacklishufan/janus_grpo/baseline_stage"

cmd = [
    "torchrun",
    "--nproc_per_node=6",
    "--nnodes=1",
    "--node_rank=0",
    "--master_addr=127.0.0.1",
    "--master_port=12346",
    "open_r1/grpo.py",
    "--use_vllm", "False",
    "--deepspeed", "../configs/zero3.json",
    "--output_dir", output_dir,
    "--model_name_or_path", qwen_path,
    "--semantic_cot", "False",
    "--dataset_name", hf_dataset,
    "--max_prompt_length", "512",
    "--max_completion_length", "1024",
    "--temperature", "1.0",
    "--num_generations", "1",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--logging_steps", "1",
    "--bf16",
    "--torch_dtype", "bfloat16",
    "--gradient_checkpointing", "false",
    "--attn_implementation", "flash_attention_2",
    "--max_steps", "1600",
    "--run_name", run_name,
    "--save_steps", "100",
    "--new_generations_image", "8",
    "--image_token_num_per_image", "576",
    "--cfg_weight", "5",
    "--reasoning_prompt_path", "../../../data/prompt/reasoning_prompt.txt",
    "--reward_funcs", "geneval",
    "--beta", "0.03",
    "--tf32", "true",
    "--learning_rate", "5e-6",
    "--reward_ckpt_path_file", "/data1/jacklishufan/STAGE/src/grpo/configs/reward_paths.json",

    "--reward_smooth", "True",
    "--kl_reweight", "True",
    "--update_ref", "False",
    "--progress_learning", "False",
    "--add_noise", "False",
    "--entropy_reward", "True",
]

os.chdir("/data1/jacklishufan/STAGE/src/grpo/src")

os.environ['WANDB_PROJECT']="grpo-janus"
os.environ["PYTHONPATH"] = f"{os.getcwd()}/..:" + os.environ.get("PYTHONPATH", "")

subprocess.run(cmd)
