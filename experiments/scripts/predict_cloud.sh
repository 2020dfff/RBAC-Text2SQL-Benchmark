#!/bin/bash

# Cloud inference prediction script for baseline comparison

set -e  # Exit on any error

# Load .env file if it exists (safer approach)
load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            # Remove spaces around = and export
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


# Default parameters
PROVIDER="deepinfra" # anthropic, deepseek, openai, deepinfra, gemini
MODEL="google/gemma-3-27b-it"  # claude-sonnet-4-5, deepseek-coder, gpt-4-mini, google/gemma-3-4b-it, gemini-2.0-flash-exp
DATASET="spider"  # Dataset name (spider, bird, livesqlbench etc.)
ROLE="false"  # Role-based evaluation flag
RBAC="false"  # RBAC (Role-Based Access Control) evaluation flag
INPUT_FILE=""  # Will be constructed from dataset and role
OUTPUT_DIR="experiments/output/pred"
OUTPUT_PREFIX="pred" # Prefix for output filename
MAX_SAMPLES="10"  # Set to empty for all samples
WORKERS="40"
RATE_LIMIT="1.0"  # Moderate delay per worker for rate limiting
TEMPERATURE="0.0"
MAX_TOKENS="4096"  # For bird, complex queries may exceed 1024 tokens

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
        --role)
            if [[ "$2" == "false" || "$2" == "no" || "$2" == "0" ]]; then
                ROLE="false"
                shift 2
            else
                ROLE="true"
                shift
            fi
            ;;
        --rbac)
            if [[ "$2" == "false" || "$2" == "no" || "$2" == "0" ]]; then
                RBAC="false"
                shift 2
            else
                RBAC="true"
                shift
            fi
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output_prefix)
            OUTPUT_PREFIX="$2"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --provider PROVIDER       API provider (openai, anthropic, deepseek, deepinfra, gemini) [default: openai]"
            echo "  --model MODEL            Model name [default: provider default]"
            echo "  --dataset DATASET        Dataset name (spider, bird, etc.) for input file path"
            echo "  --role                   Use role-based evaluation data"
            echo "  --rbac                   Use RBAC (Role-Based Access Control) evaluation data"
            echo "  --input INPUT_FILE       Input JSON file (overrides --dataset and --role)"
            echo "  --output_dir OUTPUT_DIR  Output directory [default: experiments/output/pred]"
            echo "  --output_prefix PREFIX   Output filename prefix [default: pred]"
            echo "  --max_samples N          Maximum samples to process [default: all]"
            echo "  --workers N              Number of concurrent workers [default: provider recommended]"
            echo "  --rate_limit SECONDS     Rate limit delay [default: 1.0]"
            echo "  --temperature TEMP       Sampling temperature [default: 0.0]"
            echo "  --max_tokens N           Maximum tokens [default: 4096 for complex queries]"
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

# Set default model if not specified
if [[ -z "$MODEL" ]]; then
    case $PROVIDER in
        openai)
            MODEL="gpt-3.5-turbo"
            ;;
        anthropic)
            MODEL="claude-sonnet-4-5"
            ;;
        deepseek)
            MODEL="deepseek-chat"
            ;;
        deepinfra)
            MODEL="google/gemma-3-4b-it"
            ;;
        gemini)
            MODEL="gemini-2.0-flash-exp"
            ;;
    esac
fi

# Utility: attempt to resolve input file automatically based on dataset and role
resolve_input_file() {
    local dataset_name="$1"
    local role_flag="$2"
    local rbac_flag="$3"

    local -a search_dirs=()
    if [[ -n "$dataset_name" ]]; then
        search_dirs+=("data/selected/${dataset_name}")
    fi
    search_dirs+=("data/selected")

    local dataset_hint=""
    if [[ -n "$dataset_name" ]]; then
        dataset_hint="${dataset_name}_"
    fi

    local -a patterns=()
    
    # Priority 1: RBAC files (if rbac flag is true)
    if [[ "$rbac_flag" == "true" ]]; then
        patterns+=("${dataset_hint}*dev*_with_role_rbac*.json")
        patterns+=("${dataset_hint}*dev*with_role_rbac*.jsonl")
        patterns+=("${dataset_hint}*with_role_rbac*.json")
        patterns+=("*${dataset_name}*dev*_with_role_rbac*.json")
    # Priority 2: Role-based files (if role flag is true and rbac is false)
    elif [[ "$role_flag" == "true" ]]; then
        patterns+=("${dataset_hint}*dev*_with_role*.json")
        patterns+=("${dataset_hint}*dev*with_role*.jsonl")
        patterns+=("${dataset_hint}*with_role*.json")
    # Priority 3: Standard files
    else
        patterns+=("${dataset_hint}*dev_new*.json")
        patterns+=("${dataset_hint}*dev*.json")
        patterns+=("${dataset_hint}*dev*.jsonl")
        patterns+=("${dataset_hint}*.json")
        patterns+=("${dataset_hint}*.jsonl")
    fi

    # Broaden search to any file containing dataset name if initial hint fails
    if [[ -n "$dataset_name" ]]; then
        patterns+=("*${dataset_name}*dev*.json")
        patterns+=("*${dataset_name}*dev*.jsonl")
    fi

    for dir in "${search_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            for pattern in "${patterns[@]}"; do
                while IFS= read -r match; do
                    if [[ -z "$match" ]]; then
                        continue
                    fi
                    # Skip role files if role flag is false
                    if [[ "$role_flag" != "true" && "$match" == *"with_role"* ]]; then
                        continue
                    fi
                    # Skip rbac files if rbac flag is false
                    if [[ "$rbac_flag" != "true" && "$match" == *"rbac"* ]]; then
                        continue
                    fi
                    echo "$match"
                    return 0
                done < <(compgen -G "$dir/$pattern")
            done
        fi
    done

    return 1
}

# Construct input file path if not explicitly provided
if [[ -z "$INPUT_FILE" ]]; then
    if ! INPUT_FILE=$(resolve_input_file "$DATASET" "$ROLE" "$RBAC"); then
        echo "Error: Unable to infer input file. Please specify --input explicitly."
        if [[ "$RBAC" == "true" ]]; then
            echo "Note: RBAC files (*_with_role_rbac.json) may not be available for all datasets."
        fi
        exit 1
    fi
fi

# Generate output filename
# Determine suffix components
ROLE_SUFFIX=""
RBAC_SUFFIX=""
DATASET_SUFFIX=""

# RBAC takes priority over ROLE in suffix naming
if [[ "$RBAC" == "true" ]] || [[ "$INPUT_FILE" == *"rbac"* ]]; then
    RBAC_SUFFIX="_rbac"
elif [[ "$ROLE" == "true" ]] || [[ "$INPUT_FILE" == *"with_role"* ]]; then
    ROLE_SUFFIX="_role"
fi

if [[ -n "$DATASET" ]]; then
    DATASET_SUFFIX="_${DATASET}"
fi

# Sanitize model name for filename (replace / with -)
MODEL_SAFE="${MODEL//\//-}"

if [[ -n "$MAX_SAMPLES" ]]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${MODEL_SAFE}_${MAX_SAMPLES}samples${DATASET_SUFFIX}${ROLE_SUFFIX}${RBAC_SUFFIX}.sql"
else
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${MODEL_SAFE}${DATASET_SUFFIX}${ROLE_SUFFIX}${RBAC_SUFFIX}.sql"
fi

# Generate log filename
LOG_FILE="${OUTPUT_DIR}/logs/cloud_pred_${MODEL_SAFE}${DATASET_SUFFIX}${ROLE_SUFFIX}${RBAC_SUFFIX}.log"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs"

# Print configuration
echo "========================================="
echo "Cloud Inference Configuration"
echo "========================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
echo "Max samples: ${MAX_SAMPLES:-'all'}"
echo "Rate limit: ${RATE_LIMIT}s"
if [[ -n "$TEMPERATURE" ]]; then
    TEMPERATURE_DISPLAY="$TEMPERATURE"
else
    TEMPERATURE_DISPLAY="auto (provider default)"
fi
echo "Temperature: $TEMPERATURE_DISPLAY"
echo "Max tokens: $MAX_TOKENS"
echo "Workers: ${WORKERS:-'auto (provider default)'}"
echo "========================================="

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Check API key
API_KEY_VAR="${PROVIDER^^}_API_KEY"
if [[ -z "${!API_KEY_VAR}" ]]; then
    echo "Warning: $API_KEY_VAR environment variable not set"
    echo "Make sure your API key is configured in .env file"
fi

# Start timing
START_TIME=$(date +%s)
echo "$(date): Starting cloud inference prediction" | tee "$LOG_FILE"

# Build command
CMD="python experiments/predict/predict_cloud.py \
    --input \"$INPUT_FILE\" \
    --output \"$OUTPUT_FILE\" \
    --provider \"$PROVIDER\" \
    --rate_limit \"$RATE_LIMIT\" \
    --max_tokens \"$MAX_TOKENS\""

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

if [[ -n "$TEMPERATURE" ]]; then
    CMD="$CMD --temperature \"$TEMPERATURE\""
fi

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
    echo "$(date): Cloud inference completed successfully" | tee -a "$LOG_FILE"
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
    echo "$(date): Cloud inference failed" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi