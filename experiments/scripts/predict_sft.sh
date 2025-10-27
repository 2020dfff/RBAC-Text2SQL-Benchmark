## shijian llama2 test

current_date=$(date +"%Y%m%d_%H%M")
pred_log="experiments/output/logs/pred_test_${current_date}.log"
start_time=$(date +%s)
echo " Pred Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${pred_log}


# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path defog/llama-3-sqlcoder-8b \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/bird/bird_dev_with_role.json \
#     --checkpoint_dir experiments/output/adapter/llama-3-sqlcoder-8b-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_llama-3-sqlcoder-8b-bird-role-finetuned.sql >> ${pred_log}
#     # --batch_size 8 \

CUDA_VISIBLE_DEVICES=2,3  python experiments/predict/predict.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --template chatml \
    --finetuning_type lora \
    --quantization_bit 4 \
    --predicted_input_filename data/selected/bird/bird_dev_with_role.json \
    --checkpoint_dir experiments/output/adapter/Qwen-2-5-14B-instruct-sql-qlora \
    --predicted_out_filename experiments/output/pred/pred_qwen14b-bird-role-finetuned_2.sql >> ${pred_log}

echo "############pred end###############" >>${pred_log}
echo "pred End time: $(date)" >>${pred_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hour}  hour $min min " >>${pred_log}


# CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict/predict.py \
#     --model_name_or_path huggyllama/llama-7b \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --checkpoint_dir experiments/output/adapter/llama-7b-role-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_llama7b_role.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1  python experiments/predict/predict.py \
#     --model_name_or_path Your_download_CodeLlama-13b-Instruct-hf_path \
#     --template llama2 \
#     --finetuning_type lora \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/CodeLlama-13b-sql-lora \
#     --predicted_out_filename experiments/output/pred/pred_codellama13b.sql >> ${pred_log}


# # wangzai baichua2_eval test
# CUDA_VISIBLE_DEVICES=0 python experiments/predict/predict.py \
#     --model_name_or_path /home/model/Baichuan2-13B-Chat \
#     --template baichuan2_eval \
#     --quantization_bit 4 \
#     --finetuning_type lora \
#     --checkpoint_dir experiments/output/adapter/baichuan2-13b-qlora 


## wangzai codellama2_pred test a100
# CUDA_VISIBLE_DEVICES=0,1  python experiments/predict/predict.py \
#     --model_name_or_path /home/model_files/codellama/CodeLlama-7b-Instruct-hf \
#     --template llama2 \
#     --finetuning_type lora \
#     --checkpoint_dir experiments/output/adapter/code_llama_7b-qlora

# CUDA_VISIBLE_DEVICES=0,1 python experiments/predict/predict.py \
#     --model_name_or_path llama-7b \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/llama-7b-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_llama7b.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1 python experiments/predict/predict.py \
#     --model_name_or_path llama-7b \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename dbgpt_hub_sql/data/example_text2sql_single.json \
#     --checkpoint_dir experiments/output/adapter/llama-7b-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_llama7b_test.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path codellama/CodeLlama-13b-Instruct-hf \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/CodeLlama-13b-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_codellama13b.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen-14B-Chat \
#     --template chatml \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/Qwen-14B-Chat-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_qwen14b.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
#     --template chatml \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/Qwen-2-5-14B-instruct-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-14b.sql >> ${pred_log}
#     # --batch_size 8 \

# Snowflake/Arctic-Text2SQL-R1-7B
# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path Snowflake/Arctic-Text2SQL-R1-7B \
#     --template chatml \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev.json \
#     --checkpoint_dir experiments/output/adapter/snowflake-arctic-r1-7b-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_snowflake-arctic-r1-7b.sql >> ${pred_log}
#     # --batch_size 8 \

# llama-3-sqlcoder-8b-role
# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path defog/llama-3-sqlcoder-8b \
#     --template llama2 \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --checkpoint_dir experiments/output/adapter/llama-3-sqlcoder-8b-role-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_llama-3-sqlcoder-8b-role-clean.sql >> ${pred_log}
#     # --predicted_out_filename experiments/output/pred/pred_llama-3-sqlcoder-8b-role.sql >> ${pred_log}

# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path Snowflake/Arctic-Text2SQL-R1-7B \
#     --template chatml \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --checkpoint_dir experiments/output/adapter/snowflake-arctic-r1-7b-role-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_snowflake-arctic-r1-7b-role.sql >> ${pred_log}
#     # --batch_size 8 \

# CUDA_VISIBLE_DEVICES=0,1,2,3  python experiments/predict/predict.py \
#     --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
#     --template chatml \
#     --finetuning_type lora \
#     --quantization_bit 4 \
#     --predicted_input_filename data/selected/spider/spider_dev_with_role.json \
#     --checkpoint_dir experiments/output/adapter/Qwen-2-5-14B-role-instruct-sql-qlora \
#     --predicted_out_filename experiments/output/pred/pred_qwen2.5-14b-role.sql >> ${pred_log}
#     # --batch_size 8 \