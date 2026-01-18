#!/bin/bash
# Evaluation script for RBAC benchmark (column-level and CRUD-level)
# Usage: ./run_evaluation.sh [OPTIONS]
#
# Examples:
#   ./run_evaluation.sh --prediction rbac-exp/output/pred/pred_model_spider.sql --dataset spider
#   ./run_evaluation.sh --prediction rbac-exp/output/pred/pred_model_bird.sql --dataset bird --fair_comparison
#   ./run_evaluation.sh --prediction rbac-exp/output/pred/pred_model_livesqlbench.sql --dataset livesqlbench

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBAC_EXP_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$RBAC_EXP_DIR")"

# Default values
PREDICTION_PATH="rbac-exp/output/pred/final/rbac-fewshot-result/pred_gpt-5-mini_livesqlbench_6shot_structured_rbac.sql"
DATASET="livesqlbench"
ROLE_JSON=""
EXECUTE_SQL="true"
FAIR_COMPARISON="true"
NUM_TRIALS="5"
NUM_WORKERS="8"
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prediction|--prediction_path)
            PREDICTION_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --role_json)
            ROLE_JSON="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no_execute_sql)
            EXECUTE_SQL="false"
            shift 1
            ;;
        --fair_comparison)
            FAIR_COMPARISON="true"
            shift 1
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --prediction PATH        Path to prediction file (.sql or _detailed.json)"
            echo "  --dataset DATASET        Dataset name (spider, bird, livesqlbench) [default: spider]"
            echo "  --role_json PATH         Path to role-based dataset JSON (auto-detected if not provided)"
            echo "  --output_dir PATH        Output directory for evaluation results"
            echo "  --no_execute_sql         Skip SQL execution (only string comparison)"
            echo "  --fair_comparison        Randomly select one role per unique question"
            echo "  --num_trials N           Number of trials for fair_comparison mode [default: 5]"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            # Legacy positional argument support
            if [ -z "$PREDICTION_PATH" ]; then
                PREDICTION_PATH="$1"
            elif [ -z "$DATASET" ]; then
                DATASET="$1"
            fi
            shift 1
            ;;
    esac
done

if [ -z "$PREDICTION_PATH" ]; then
    echo "Error: Prediction path is required"
    echo "Usage: $0 --prediction <path> --dataset <spider|bird|livesqlbench> [options]"
    echo "Use --help for more information"
    exit 1
fi

# Auto-detect role JSON if not provided
if [ -z "$ROLE_JSON" ]; then
    case $DATASET in
        spider)
            ROLE_JSON="${ROOT_DIR}/data/selected/spider/column_level_rbac_dataset_spider_20260113.json"
            ;;
        bird)
            ROLE_JSON="${ROOT_DIR}/data/selected/bird/column_level_rbac_dataset_bird_20251230.json"
            ;;
        livesqlbench)
            ROLE_JSON="${ROOT_DIR}/data/selected/livesqlbench-full/crud_rbac_dataset_v2_20251230.json"
            ;;
        *)
            echo "Error: Unknown dataset '$DATASET'"
            exit 1
            ;;
    esac
fi

# Check if role JSON exists
if [ ! -f "$ROLE_JSON" ]; then
    echo "Error: Role JSON file not found: $ROLE_JSON"
    exit 1
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${RBAC_EXP_DIR}/output/eval_result"
fi
mkdir -p "$OUTPUT_DIR"

# Select evaluation script based on dataset
if [ "$DATASET" = "livesqlbench" ]; then
    EVAL_SCRIPT="${RBAC_EXP_DIR}/evaluation/evaluate_crud_level.py"
else
    EVAL_SCRIPT="${RBAC_EXP_DIR}/evaluation/evaluate_column_level.py"
fi

# Print configuration
echo "========================================="
echo "RBAC Evaluation Configuration"
echo "========================================="
echo "Dataset: $DATASET"
echo "Prediction file: $PREDICTION_PATH"
echo "Role JSON: $ROLE_JSON"
echo "Output directory: $OUTPUT_DIR"
echo "Execute SQL: $EXECUTE_SQL"
echo "Fair comparison: $FAIR_COMPARISON"
if [ "$FAIR_COMPARISON" = "true" ]; then
    echo "Number of trials: $NUM_TRIALS"
fi
echo "Evaluation script: $EVAL_SCRIPT"
echo "========================================="

# Build command based on dataset type
if [ "$DATASET" = "livesqlbench" ]; then
    # CRUD-level evaluation (livesqlbench) - use module syntax
    CMD="python -m rbac-exp.evaluation.evaluate_crud_level \
        --prediction_path \"$PREDICTION_PATH\" \
        --role_json \"$ROLE_JSON\" \
        --db_user feiy \
        --db_password REDACTED \
        --num_workers $NUM_WORKERS"
else
    # Column-level evaluation (spider, bird)
    CMD="python -m rbac-exp.evaluation.evaluate_column_level \
        --prediction_path \"$PREDICTION_PATH\" \
        --dataset $DATASET \
        --role_json \"$ROLE_JSON\" \
        --output_path \"$OUTPUT_DIR\""
    
    # Add execute_sql flag (column-level only)
    if [ "$EXECUTE_SQL" = "false" ]; then
        CMD="$CMD --no_execute_sql"
    fi
fi

# Add fair_comparison flag and num_trials
if [ "$FAIR_COMPARISON" = "true" ]; then
    CMD="$CMD --fair_comparison"
    CMD="$CMD --num_trials $NUM_TRIALS"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD
