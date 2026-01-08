#!/bin/bash

# Cloud inference prediction script for RBAC benchmark evaluation
# Aligned with experiments/scripts/predict_cloud.sh

set -e  # Exit on any error

# Load .env file if it exists
load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            key=$(echo "$key" | tr -d ' ')
            value=$(echo "$value" | tr -d ' ')
            [[ -n "$key" && -n "$value" ]] && export "$key=$value"
        done < <(grep -E '^[^#]' "$env_file" | grep '=')
        echo "✓ Loaded environment variables from $env_file"
        return 0
    fi
    return 1
}

load_env_file ".env" || load_env_file "../../.env" || load_env_file "../../../.env" || true

# Default parameters (aligned with experiments)
PROVIDER="deepseek"
MODEL="deepseek-reasoner"  # Will use provider default
DATASET="spider"  # spider, bird, livesqlbench
INPUT_FILE=""  # Will be constructed from dataset
OUTPUT_DIR="rbac-exp/output/pred"
MAX_SAMPLES=""  # Set to empty for all samples
WORKERS="3"  # Use provider default
RATE_LIMIT="2.0"
TEMPERATURE="0.0"
MAX_TOKENS="4096"
SHOT_NUM="0"  # Zero/Few-shot examples (0, 2, 4, 6)
STRUCTURED="true"  # Use structured prompt format
MODE="baseline"  # Evaluation mode: rbac or baseline

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --rate_limit)
            RATE_LIMIT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --shot_num)
            SHOT_NUM="$2"
            shift 2
            ;;
        --structured)
            STRUCTURED="true"
            shift 1
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --provider PROVIDER       API provider (openai, anthropic, deepseek, deepinfra, gemini) [default: deepseek]"
            echo "  --model MODEL            Model name [default: provider default]"
            echo "  --dataset DATASET        Dataset name (spider, bird, livesqlbench)"
            echo "  --input INPUT_FILE       Input JSON file (overrides --dataset)"
            echo "  --output_dir OUTPUT_DIR  Output directory [default: rbac-exp/output/pred]"
            echo "  --max_samples N          Maximum samples to process [default: all]"
            echo "  --workers N              Number of concurrent workers [default: provider default]"
            echo "  --rate_limit SECONDS     Rate limit delay [default: 1.0]"
            echo "  --temperature TEMP       Sampling temperature [default: 0.0]"
            echo "  --max_tokens N           Maximum tokens [default: 4096]"
            echo "  --shot_num N             Few-shot examples (0, 1, 3, 5) [default: 0]"
            echo "  --structured             Use structured prompt format (extracts schema, uses JSON policy)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate provider
if [[ "$PROVIDER" != "openai" && "$PROVIDER" != "anthropic" && "$PROVIDER" != "deepseek" && "$PROVIDER" != "deepinfra" && "$PROVIDER" != "gemini" ]]; then
    echo "Error: Provider must be 'openai', 'anthropic', 'deepseek', 'deepinfra', or 'gemini'"
    exit 1
fi

# Resolve input file if not specified
if [[ -z "$INPUT_FILE" ]]; then
    case $DATASET in
        spider)
            INPUT_FILE="data/selected/spider/column_level_rbac_dataset_spider_20251229.json"
            ;;
        bird)
            INPUT_FILE="data/selected/bird/column_level_rbac_dataset_bird_20251230.json"
            ;;
        livesqlbench)
            INPUT_FILE="data/selected/livesqlbench-full/crud_rbac_dataset_v2_20251230.json"
            ;;
        *)
            echo "Error: Unknown dataset '$DATASET'. Please specify --input"
            exit 1
            ;;
    esac
fi

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Set default model if not specified (for output filename)
if [[ -z "$MODEL" ]]; then
    case $PROVIDER in
        openai) MODEL_DISPLAY="gpt-4o-mini" ;;
        anthropic) MODEL_DISPLAY="claude-3-5-sonnet" ;;
        deepseek) MODEL_DISPLAY="deepseek-chat" ;;
        deepinfra) MODEL_DISPLAY="llama-3.1-70b" ;;
        gemini) MODEL_DISPLAY="gemini-1.5-flash" ;;
    esac
else
    MODEL_DISPLAY="${MODEL//\//-}"
fi

# Generate output filename with shot_num suffix
SHOT_SUFFIX=""
if [[ "$SHOT_NUM" != "0" ]]; then
    SHOT_SUFFIX="_${SHOT_NUM}shot"
fi

STRUCT_SUFFIX=""
if [[ "$STRUCTURED" == "true" ]]; then
    STRUCT_SUFFIX="_structured"
fi

# Generate mode suffix for output filename
MODE_SUFFIX="_rbac"
if [[ "$MODE" == "baseline" ]]; then
    MODE_SUFFIX="_baseline"
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    OUTPUT_FILE="${OUTPUT_DIR}/pred_${MODEL_DISPLAY}_${MAX_SAMPLES}samples_${DATASET}${SHOT_SUFFIX}${STRUCT_SUFFIX}${MODE_SUFFIX}.sql"
else
    OUTPUT_FILE="${OUTPUT_DIR}/pred_${MODEL_DISPLAY}_${DATASET}${SHOT_SUFFIX}${STRUCT_SUFFIX}${MODE_SUFFIX}.sql"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs"

# Generate log filename
LOG_FILE="${OUTPUT_DIR}/logs/rbac_pred_${MODEL_DISPLAY}_${DATASET}${SHOT_SUFFIX}.log"

# Print configuration
echo "========================================="
echo "RBAC Cloud Inference Configuration"
echo "========================================="
echo "Provider: $PROVIDER"
echo "Model: ${MODEL:-'(provider default)'}"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
echo "Max samples: ${MAX_SAMPLES:-'all'}"
echo "Shot num: $SHOT_NUM"
echo "Structured prompt: $STRUCTURED"
echo "Rate limit: ${RATE_LIMIT}s"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Workers: ${WORKERS:-'(provider default)'}"
echo "========================================="

# Check API key
API_KEY_VAR="$(echo ${PROVIDER} | tr '[:lower:]' '[:upper:]')_API_KEY"
if [[ -z "${!API_KEY_VAR}" ]]; then
    echo "Warning: $API_KEY_VAR environment variable not set"
    echo "Make sure your API key is configured in .env file"
fi

# Start timing
START_TIME=$(date +%s)
echo "$(date): Starting RBAC cloud inference prediction" | tee "$LOG_FILE"

# Build command
CMD="python rbac-exp/predict/predict_cloud.py \
    --input \"$INPUT_FILE\" \
    --output \"$OUTPUT_FILE\" \
    --provider \"$PROVIDER\" \
    --dataset \"$DATASET\" \
    --rate_limit \"$RATE_LIMIT\" \
    --max_tokens \"$MAX_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --shot_num \"$SHOT_NUM\""

# Add optional parameters
if [[ -n "$MODEL" ]]; then
    CMD="$CMD --model \"$MODEL\""
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD="$CMD --max_samples \"$MAX_SAMPLES\""
fi

if [[ -n "$WORKERS" ]]; then
    CMD="$CMD --workers \"$WORKERS\""
fi

if [[ "$STRUCTURED" == "true" ]]; then
    CMD="$CMD --structured"
fi

# Add mode parameter
CMD="$CMD --mode \"$MODE\""

# Execute prediction
echo "$(date): Executing: $CMD" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

if eval "$CMD" 2>&1 | tee -a "$LOG_FILE"; then
    # Success
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "$(date): RBAC cloud inference completed successfully" | tee -a "$LOG_FILE"
    echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
    echo "Output saved to: $OUTPUT_FILE" | tee -a "$LOG_FILE"
    
    # Show file size and line count
    if [[ -f "$OUTPUT_FILE" ]]; then
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Generated $LINE_COUNT predictions ($FILE_SIZE)" | tee -a "$LOG_FILE"
    fi
    
    echo "Log saved to: $LOG_FILE"
else
    # Error
    echo "$(date): RBAC cloud inference failed" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi
