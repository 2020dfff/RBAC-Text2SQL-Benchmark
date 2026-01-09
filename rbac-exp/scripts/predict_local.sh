#!/bin/bash
################################################################################
#                         CONFIGURATION SECTION                                 
#                    (MODIFY THESE VARIABLES TO RUN)                            
################################################################################

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MODEL="Snowflake/Arctic-Text2SQL-R1-7B"         # HuggingFace model name or local path
TEMPLATE="chatml"                               # Options: chatml, llama2, gemma, default

# LoRA Adapter Configuration (leave empty for base model)
LORA_ADAPTER="rbac-exp/output/adapter/arctic-text2sql-r1-7b-column-rbac-test"
FINETUNING_TYPE="lora"                          # Options: lora, full, freeze
QUANTIZATION_BIT=""                             # Options: 4, 8, or empty for none

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION  
# ═══════════════════════════════════════════════════════════════════════════════
DATASET="spider"                                # Options: spider, bird, livesqlbench

# Dataset file paths
SPIDER_DATASET="outputs/column_level_rbac_dataset_spider_20251229.json"
BIRD_DATASET="outputs/column_level_rbac_dataset_bird_20251230.json"
LIVESQLBENCH_DATASET="outputs/livesqlbench_crud/crud_rbac_dataset_v2_20251230.json"

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
SHOT_NUM=0                                      # Number of few-shot examples (0-5)
MAX_NEW_TOKENS=4096                             # Max tokens to generate
TEMPERATURE=0.0                                 # 0.0 = deterministic, higher = more random
TOP_P=1.0                                       # Nucleus sampling threshold

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CUDA_DEVICES="0,1,2"                          # GPUs to use (comma-separated)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT & RESUME
# ═══════════════════════════════════════════════════════════════════════════════
SAVE_EVERY=100                                  # Save checkpoint every N samples
ENABLE_RESUME=true                              # Resume from checkpoint if exists

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODE
# ═══════════════════════════════════════════════════════════════════════════════
MODE="rbac"                                     # Options: rbac, baseline

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE LIMIT (for testing, empty = all samples)
# ═══════════════════════════════════════════════════════════════════════════════
MAX_SAMPLES=                                    # Limit samples for testing

################################################################################
#                         END OF CONFIGURATION                                  
#              (Do not modify below unless you know what you're doing)          
################################################################################

set -e

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBAC_EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$RBAC_EXP_DIR")"

cd "$PROJECT_ROOT"

# Validate required parameters
if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Error: MODEL and DATASET must be configured"
    echo ""
    echo "Edit the CONFIGURATION SECTION at the top of this script, or run:"
    echo "  ./predict_local.sh <model> <dataset> [shot_num] [template] [lora_adapter]"
    exit 1
fi

# Resolve dataset path
case $DATASET in
    "spider")  INPUT_FILE="${PROJECT_ROOT}/${SPIDER_DATASET}" ;;
    "bird")    INPUT_FILE="${PROJECT_ROOT}/${BIRD_DATASET}" ;;
    "livesqlbench") INPUT_FILE="${PROJECT_ROOT}/${LIVESQLBENCH_DATASET}" ;;
    *) echo "Error: Unknown dataset '$DATASET' (must be: spider, bird, livesqlbench)"; exit 1 ;;
esac

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Check dataset paths in the CONFIGURATION SECTION"
    exit 1
fi

# Output paths
OUTPUT_DIR="${RBAC_EXP_DIR}/output/pred"
LOG_DIR="${RBAC_EXP_DIR}/output/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Generate mode suffix for output filename
MODE_SUFFIX="_rbac"
if [[ "$MODE" == "baseline" ]]; then
    MODE_SUFFIX="_baseline"
fi

OUTPUT_FILE="${OUTPUT_DIR}/pred_${MODEL_SHORT}_${DATASET}${MODE_SUFFIX}.sql"
LOG_FILE="${LOG_DIR}/pred_${MODEL_SHORT}_${DATASET}${MODE_SUFFIX}_${TIMESTAMP}.log"

# Build flags
RESUME_FLAG=""; [ "$ENABLE_RESUME" = true ] && RESUME_FLAG="--resume"
SNOWFLAKE_FLAG=""; [[ "$MODEL" == *"Arctic"* || "$MODEL" == *"nowflake"* ]] && SNOWFLAKE_FLAG="--snowflake_mode"

# Print configuration
echo "═══════════════════════════════════════════════════════════════════"
echo " RBAC LOCAL INFERENCE"
echo "═══════════════════════════════════════════════════════════════════"
echo " Model      : $MODEL"
echo " Dataset    : $DATASET"
echo " Mode       : $MODE"
echo " Input      : $INPUT_FILE"
echo " Output     : $OUTPUT_FILE"
echo " GPUs       : $CUDA_DEVICES"
echo " Shot       : $SHOT_NUM"
echo " Template   : $TEMPLATE"
[ -n "$LORA_ADAPTER" ] && echo " Adapter    : $LORA_ADAPTER"
[ -n "$MAX_SAMPLES" ] && echo " Max Samples: $MAX_SAMPLES"
echo "═══════════════════════════════════════════════════════════════════"

# Build and run command
CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ${RBAC_EXP_DIR}/predict/predict_local.py \
    --model_name_or_path $MODEL \
    --template $TEMPLATE \
    --predicted_input_filename $INPUT_FILE \
    --predicted_out_filename $OUTPUT_FILE \
    --dataset $DATASET \
    --shot_num $SHOT_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --save_every $SAVE_EVERY \
    --mode $MODE \
    $RESUME_FLAG $SNOWFLAKE_FLAG"

# Add sample limit if specified
[ -n "$MAX_SAMPLES" ] && CMD="$CMD --max_samples $MAX_SAMPLES"

# Add LoRA adapter configuration if specified
if [ -n "$LORA_ADAPTER" ]; then
    CMD="$CMD --checkpoint_dir $LORA_ADAPTER --finetuning_type $FINETUNING_TYPE"
    [ -n "$QUANTIZATION_BIT" ] && CMD="$CMD --quantization_bit $QUANTIZATION_BIT"
fi

echo "Start: $(date)" | tee -a "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_FILE"
