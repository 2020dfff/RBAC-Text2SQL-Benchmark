#!/bin/bash

## Baseline prediction script - using pre-trained models without fine-tuning
## This script provides baseline results for comparison with fine-tuned models

current_date=$(date +"%Y%m%d_%H%M")
pred_log="experiments/output/logs/pred_baseline_${current_date}.log"
log_dir=$(dirname "${pred_log}")
mkdir -p "${log_dir}" experiments/output/pred
start_time=$(date +%s)
echo " Baseline Pred Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${pred_log}

echo "Running baseline prediction without fine-tuning..." >>${pred_log}

# Qwen2.5-Coder-7B-Instruct Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --template chatml \
#     --predicted_input_filename data/selected/bird/bird_dev_with_role.json \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-coder-7b-bird-role.sql

# echo "Qwen2.5-Coder-7B baseline prediction completed" >>${pred_log}

# Llama-2-7B Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --template llama2 \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --predicted_out_filename experiments/output/pred/pred_llama-7b-spider-role.sql

# Llama-3-SQLCoder-8B Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path defog/llama-3-sqlcoder-8b \
#     --template llama2 \
#     --predicted_input_filename data/selected/bird/bird_dev_with_role.json \
#     --predicted_out_filename experiments/output/pred/pred_llama-3-sqlcoder-8b-bird-role.json

# Snowflake Arctic Baseline (with increased max_new_tokens for verbose outputs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
    --model_name_or_path Snowflake/Arctic-Text2SQL-R1-7B \
    --template chatml \
    --max_new_tokens 4096 \
    --predicted_input_filename data/selected/bird/bird_dev_with_role_rbac.json \
    --predicted_out_filename experiments/output/pred/pred_snowflake-arctic-r1-7b-bird-role-rbac.sql

echo "Snowflake Arctic baseline prediction completed" >>${pred_log}

# CUDA_VISIBLE_DEVICES=1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path google/gemma-3-4b-it \
#     --template gemma \
#     --max_new_tokens 4096 \
#     --temperature 0.0 \
#     --top_p 0.9 \
#     --predicted_input_filename data/selected/livesqlbench/livesqlbench_dev_with_role.json \
#     --predicted_out_filename experiments/output/pred/pred_gemma3-4b-livesqlbench_role.sql

# echo "Gemma-3-4B baseline prediction completed" >>${pred_log}

# Qwen2.5-14B-Instruct Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
#     --template chatml \
#     --predicted_input_filename data/selected/livesqlbench/livesqlbench_dev.json \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-14b-livesqlbench.sql

# echo "Qwen2.5-14B baseline prediction completed" >>${pred_log}

# echo "Running LiveSQLBench baseline prediction with Qwen2.5-14B..." >>${pred_log}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
#     --template chatml \
#     --predicted_input_filename data/selected/livesqlbench/livesqlbench_dev.json \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-14b-livesqlbench.sql >> ${pred_log} 2>&1

# echo "Qwen2.5-14B LiveSQLBench prediction completed" >>${pred_log}

# # CodeLlama-13b-Instruct Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path codellama/CodeLlama-13b-Instruct-hf \
#     --template llama2 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --predicted_out_filename experiments/output/pred/pred_codellama13b-baseline.sql >> ${pred_log} 2>&1

# # Llama-3-SQLCoder-8B Baseline (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path defog/llama-3-sqlcoder-8b \
#     --template llama2 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --predicted_out_filename experiments/output/pred/pred_llama-3-sqlcoder-8b-baseline.sql >> ${pred_log} 2>&1

# # Snowflake Arctic Baseline (no fine-tuning) 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path Snowflake/Arctic-Text2SQL-R1-7B \
#     --template chatml \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --predicted_out_filename experiments/output/pred/pred_snowflake-arctic-r1-7b-baseline.sql >> ${pred_log} 2>&1

# Role-based data baseline predictions
# echo "Running role-based baseline predictions..." >>${pred_log}

# Qwen2.5-14B with role-based data (no fine-tuning)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
#     --template chatml \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-14b-role-baseline.sql >> ${pred_log} 2>&1

echo "Role-based baseline predictions completed" >>${pred_log}

echo "############baseline pred end###############" >>${pred_log}
echo "Baseline pred End time: $(date)" >>${pred_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hours} hour $min min " >>${pred_log}
