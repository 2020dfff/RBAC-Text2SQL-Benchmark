#!/bin/bash

# Schema Exposure Analysis Script for Column-Level RBAC
# Analyzes: ValidG (valid unauthorized column guess), InvalidG (non-existent column), NoUnauth (under-refusal)
#
# Usage: ./run_schema_exposure_eval.sh [OPTIONS]
#
# Examples:
#   ./run_schema_exposure_eval.sh  # Uses default PREDICT_FILE below
#   ./run_schema_exposure_eval.sh --predict rbac-exp/output/pred/pred_model_bird.sql --dataset bird

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBAC_EXP_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$RBAC_EXP_DIR")"

# ============================================================================
# DEFAULT CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# ============================================================================
PREDICT_FILE="rbac-exp/output/pred/pred_gpt-5-mini_bird_structured_rbac.sql"
DATASET="bird"
ROLE_JSON=""  # Auto-detected if empty
NUM_TRIALS=5
FAIR_COMPARISON="true"
EXECUTE_SQL="true"

# ============================================================================
# Parse command line arguments (override defaults)
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --predict|--prediction)
            PREDICT_FILE="$2"
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
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --no_fair)
            FAIR_COMPARISON="false"
            shift 1
            ;;
        --no_execute_sql)
            EXECUTE_SQL="false"
            shift 1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --predict FILE           Path to predictions file (.sql)"
            echo "  --dataset DATASET        Dataset name (spider or bird) [default: bird]"
            echo "  --role_json FILE         Path to role-based dataset JSON (auto-detected)"
            echo "  --num_trials N           Number of trials for fair comparison [default: 5]"
            echo "  --no_fair                Disable fair comparison mode"
            echo "  --no_execute_sql         Skip SQL execution"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Auto-detect role JSON if not specified
# ============================================================================
if [[ -z "$ROLE_JSON" ]]; then
    case $DATASET in
        spider)
            ROLE_JSON="${ROOT_DIR}/data/selected/spider/column_level_rbac_dataset_spider_20260113.json"
            ;;
        bird)
            ROLE_JSON="${ROOT_DIR}/data/selected/bird/column_level_rbac_dataset_bird_filtered_schema_20260111.json"
            ;;
        *)
            echo "Error: Unknown dataset '$DATASET'. Please specify --role_json"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Validate files exist
# ============================================================================
if [[ ! -f "$PREDICT_FILE" ]]; then
    echo "Error: Prediction file not found: $PREDICT_FILE"
    exit 1
fi

if [[ ! -f "$ROLE_JSON" ]]; then
    echo "Error: Role JSON file not found: $ROLE_JSON"
    exit 1
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "========================================="
echo "Schema Exposure Analysis Configuration"
echo "========================================="
echo "Dataset: $DATASET"
echo "Prediction file: $PREDICT_FILE"
echo "Role JSON: $ROLE_JSON"
echo "Fair comparison: $FAIR_COMPARISON"
if [[ "$FAIR_COMPARISON" == "true" ]]; then
    echo "Number of trials: $NUM_TRIALS"
fi
echo "Execute SQL: $EXECUTE_SQL"
echo "========================================="

# ============================================================================
# Build and run command
# ============================================================================
CMD="python -m rbac-exp.evaluation.evaluate_schema_exposure \
    --role_json \"$ROLE_JSON\" \
    --predict \"$PREDICT_FILE\" \
    --dataset \"$DATASET\" \
    --num_trials $NUM_TRIALS"

if [[ "$FAIR_COMPARISON" == "true" ]]; then
    CMD="$CMD --fair_comparison"
fi

if [[ "$EXECUTE_SQL" == "false" ]]; then
    CMD="$CMD --no_execute"
fi

echo ""
echo "Running: $CMD"
echo ""
eval "$CMD"

echo "========================================="
echo "Schema exposure analysis complete!"
echo "========================================="
