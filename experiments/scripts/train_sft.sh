# wandb offline # Close wandb
# a100 ,单卡
current_date=$(date +"%Y%m%d_%H%M")
train_log="experiments/output/logs/train_sft_test_${current_date}.log"
start_time=$(date +%s)
echo " Train Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${train_log}

# default train , zero-shot, 
num_shot=0

# one-shot train
# num_shot=1

dataset="spider_train_with_role"
if [ "$num_shot" -eq 1 ]; then
    dataset="spider_train_with_role_one_shot"
fi


model_name_or_path=${model_name_or_path-"Qwen/Qwen2.5-14B-Instruct"}
output_dir="experiments/output/adapter/Qwen-2-5-14B-instruct-sql-qlora"

deepspeed --num_gpus 4  experiments/train/sft_train.py \
    --deepspeed experiments/configs/ds_config.json \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --dataset $dataset \
    --max_source_length 2048 \
    --max_target_length 512 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --template chatml \
    --lora_rank 64 \
    --lora_alpha 32 \
    --output_dir $output_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine_with_restarts \
    --logging_steps 50 \
    --save_steps 2000 \
    --learning_rate 2e-4 \
    --num_train_epochs 8 \
    --plot_loss \
    --bf16  >> ${train_log}\
    --quantization_bit 4
    
    
echo "############train end###############" >>${train_log}
echo "Train End time: $(date)" >>${train_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hour}  hour $min min " >>${train_log}

# model_name_or_path=${model_name_or_path-"defog/llama-3-sqlcoder-8b"}
# output_dir="experiments/output/adapter/llama-3-sqlcoder-8b-sql-qlora"

# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj\
#     --template llama2 \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4 
#     # --max_samples 1 \

# 多卡，deepseed启动，A100
# deepspeed --num_gpus 2  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/stage2.json \
#     --quantization_bit 4 \
#     --model_name_or_path /home/model_files/Llama-2-13b-chat-hf \
#     --do_train \
#     --dataset example_text2sql_train \
#     --max_source_length 1024 \
#     --max_target_length 512 \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --lora_target q_proj,v_proj \
#     --output_dir experiments/output/adapter/llama2-13b-qlora_1024_epoch1_debug1008_withDeepseed_mulitCard \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 25 \
#     --save_steps 20 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 0.1 \
#     --plot_loss \
#     --bf16 2>&1 | tee ${train_log}


# 多卡，deepseed，全量微调
# deepspeed --include localhost:4,5,6,7  experiments/train/sft_train.py \
#     --dataset example_text2sql_train \
#     --model_name_or_path CodeLlama-7b-Instruct-hf \
#     --do_train \
#     --finetuning_type full \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --template llama2 \
#     --output_dir experiments/output/adapter/code-llama-7b-2048_epoch4_full \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 4 \
#     --plot_loss \
#     --bf16 True\
#     --deepspeed experiments/configs/stage3.json 2>&1 | tee ${train_log}

# 多卡，deepseed，qlora微调 llama-7b  codellama-13b
# model_name_or_path=${model_name_or_path-"codellama/CodeLlama-13b-Instruct-hf"}
# output_dir="experiments/output/adapter/CodeLlama-13b-sql-qlora"
# model_name_or_path=${model_name_or_path-"llama-7b"}
# output_dir="experiments/output/adapter/llama-7b-sql-qlora"
# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --template llama2 \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4
    

# Qwen-7B-Chat/14B Qlora微调
# model_name_or_path=${model_name_or_path-"Qwen/Qwen-14B-Chat"}
# output_dir="experiments/output/adapter/Qwen-14B-Chat-sql-qlora"

# model_name_or_path=${model_name_or_path-"Qwen/Qwen-7B-Chat"}
# output_dir="experiments/output/adapter/Qwen-7B-Chat-sql-qlora"
# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target c_attn \
#     --template chatml \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4


# Qwen-2.5-14B Qlora sft
# model_name_or_path=${model_name_or_path-"Qwen/Qwen2.5-14B-Instruct"}
# output_dir="experiments/output/adapter/Qwen-2-5-14B-instruct-sql-qlora"
# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj\
#     --template chatml \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4

# model_name_or_path=${model_name_or_path-"defog/llama-3-sqlcoder-8b"}
# output_dir="experiments/output/adapter/llama-3-sqlcoder-8b-sql-qlora"

# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj\
#     --template llama2 \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4


# Snowflake/Arctic-Text2SQL-R1-7B Qlora sft

# model_name_or_path=${model_name_or_path-"Snowflake/Arctic-Text2SQL-R1-7B"}
# output_dir="experiments/output/adapter/snowflake-arctic-r1-7b-sql-qlora"

# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj\
#     --template chatml \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4


# Qwen-2.5-14B Qlora sft
# model_name_or_path=${model_name_or_path-"Qwen/Qwen2.5-14B-Instruct"}
# output_dir="experiments/output/adapter/Qwen-2-5-14B-role-instruct-sql-qlora"

# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj\
#     --template chatml \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4


# # Llama-7B Qlora sft
# model_name_or_path=${model_name_or_path-"huggyllama/llama-7b"}
# output_dir="experiments/output/adapter/llama-7b-role-sql-qlora"

# deepspeed --num_gpus 4  experiments/train/sft_train.py \
#     --deepspeed experiments/configs/ds_config.json \
#     --model_name_or_path $model_name_or_path \
#     --do_train \
#     --dataset $dataset \
#     --max_source_length 2048 \
#     --max_target_length 512 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --template llama2 \
#     --lora_rank 64 \
#     --lora_alpha 32 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine_with_restarts \
#     --logging_steps 50 \
#     --save_steps 2000 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 8 \
#     --plot_loss \
#     --bf16  >> ${train_log}\
#     --quantization_bit 4