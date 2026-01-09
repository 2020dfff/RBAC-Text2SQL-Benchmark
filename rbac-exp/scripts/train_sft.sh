#!/bin/bash
################################################################################
#                    Column-Level RBAC SFT Training Script                      
#                                                                               
# This script trains models for Column-Level Role-Based Access Control          
# Text-to-SQL task. The model learns to:                                         
#   - Generate SQL when user has column-level permission                         
#   - Refuse with "Sorry, I cannot answer." when permission denied               
#                                                                               
# Training data: outputs/column_level_rbac_dataset_spider_train_20260109.json   
# - 29,052 samples (59.1% allowed, 40.9% denied)                                
################################################################################

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME_OR_PATH="Snowflake/Arctic-Text2SQL-R1-7B"
TEMPLATE="chatml"                               # Options: chatml, llama2, gemma, mistral

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DATASET="column_level_rbac_spider_train"
MAX_SAMPLES=""                                  # Empty = all samples, or set number for testing

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
NUM_GPUS=4                                      # Number of GPUs
NUM_TRAIN_EPOCHS=8                              # Number of epochs
BATCH_SIZE=1                                    # Per-device batch size
GRADIENT_ACCUMULATION=16                        # Gradient accumulation steps
LEARNING_RATE=2e-4                              # Learning rate

# ═══════════════════════════════════════════════════════════════════════════════
# LORA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
LORA_RANK=64                                    # LoRA rank
LORA_ALPHA=32                                   # LoRA alpha
LORA_TARGET="q_proj,v_proj"                     # Target modules

# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE LENGTH
# ═══════════════════════════════════════════════════════════════════════════════
MAX_SOURCE_LENGTH=2048                          # Max input length
MAX_TARGET_LENGTH=512                           # Max output length

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=0,1,2,3             # GPUs to use

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION (optional)
# ═══════════════════════════════════════════════════════════════════════════════
QUANTIZATION_BIT=4                              # 4, 8, or empty for none

################################################################################
#                         END OF CONFIGURATION                                  
################################################################################

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBAC_EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$RBAC_EXP_DIR")"

cd "$PROJECT_ROOT"

# Output Configuration
OUTPUT_BASE_DIR="rbac-exp/output/adapter"
LOG_DIR="rbac-exp/output/logs"
mkdir -p "$OUTPUT_BASE_DIR" "$LOG_DIR"

# Generate output directory name
MODEL_SHORT=$(basename "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_SHORT}-column-rbac-${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/train_sft_column_rbac_${MODEL_SHORT}_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Column-Level RBAC SFT Training"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Model:      $MODEL_NAME_OR_PATH"
echo "  Template:   $TEMPLATE"
echo "  Dataset:    $DATASET"
echo "  Samples:    $MAX_SAMPLES"
echo "  Output:     $OUTPUT_DIR"
echo "  GPU:        $CUDA_VISIBLE_DEVICES"
echo "  Epochs:     $NUM_TRAIN_EPOCHS"
echo "  Batch Size: $BATCH_SIZE (x $GRADIENT_ACCUMULATION grad accum)"
echo "  LoRA:       rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

# Record start time
START_TIME=$(date +%s)
echo "Training started at: $(date)" | tee -a "$LOG_FILE"

# Build base arguments
BASE_ARGS="--model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train \
    --dataset $DATASET \
    --template $TEMPLATE \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --finetuning_type lora \
    --lora_target $LORA_TARGET \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --lr_scheduler_type cosine_with_restarts \
    --logging_steps 50 \
    --save_steps 2000 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --plot_loss"

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    BASE_ARGS="$BASE_ARGS --max_samples $MAX_SAMPLES"
fi

# Build command based on NUM_GPUS
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with DeepSpeed
    # Note: DeepSpeed doesn't work well with quantization (QLoRA)
    # bf16 is controlled by ds_config.json, don't pass --bf16 to avoid conflict
    if [ -n "$QUANTIZATION_BIT" ]; then
        echo "⚠️  Warning: Quantization (QLoRA) is disabled for multi-GPU DeepSpeed training"
        echo "    DeepSpeed ZeRO doesn't support quantized models well"
        echo ""
    fi
    CMD="deepspeed --num_gpus $NUM_GPUS rbac-exp/train/sft_train.py \
        --deepspeed rbac-exp/configs/ds_config.json \
        $BASE_ARGS"
else
    # Single GPU without DeepSpeed - can use quantization and bf16
    BASE_ARGS="$BASE_ARGS --bf16"
    if [ -n "$QUANTIZATION_BIT" ]; then
        BASE_ARGS="$BASE_ARGS --quantization_bit $QUANTIZATION_BIT"
    fi
    CMD="python rbac-exp/train/sft_train.py $BASE_ARGS"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute training
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Training Complete!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  End time:    $(date)"
echo "  Duration:    ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Output:      $OUTPUT_DIR"
echo "  Log:         $LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════════════════"

echo "" >> "$LOG_FILE"
echo "Training ended at: $(date)" >> "$LOG_FILE"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$LOG_FILE"
