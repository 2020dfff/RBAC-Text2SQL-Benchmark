#!/bin/bash
# Baseline Text2SQL Evaluation Script
# Evaluates SQL accuracy without RBAC metrics
#
# This script uses the _detailed.json file from predictions,
# which already contains SystemManager role items with valid gold_sql.
# No separate data file is needed.
#
# For spider/bird: uses SQLite via evaluate_column_baseline.py
# For livesqlbench: uses PostgreSQL via evaluate_crud_level.py

set -e

# Default configuration
DATASET="livesqlbench"  # spider, bird, livesqlbench
PRED_FILE="rbac-exp/output/pred/final/baseline-result/pred_google-gemma-3-27b-it_livesqlbench_structured_baseline.sql"
OUTPUT_DIR="rbac-exp/output/eval_result"
PLUG_VALUE="false"

# PostgreSQL configuration (for livesqlbench)
DB_USER="feiy"
DB_PASSWORD="REDACTED"
DB_HOST="localhost"
DB_PORT="5432"
NUM_WORKERS="8"
DRY_RUN="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --pred)
            PRED_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plug_value)
            PLUG_VALUE="true"
            shift 1
            ;;
        --db_user)
            DB_USER="$2"
            shift 2
            ;;
        --db_password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --db_host)
            DB_HOST="$2"
            shift 2
            ;;
        --db_port)
            DB_PORT="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="true"
            shift 1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET    Dataset name (spider, bird, livesqlbench)"
            echo "  --pred PRED_FILE     Path to prediction file (.sql or _detailed.json)"
            echo "  --output_dir DIR     Output directory for results"
            echo "  --plug_value         Enable value plugging (slow!) [spider/bird only]"
            echo ""
            echo "PostgreSQL options (livesqlbench only):"
            echo "  --db_user USER       PostgreSQL username (default: feiy)"
            echo "  --db_password PASS   PostgreSQL password"
            echo "  --db_host HOST       PostgreSQL host (default: localhost)"
            echo "  --db_port PORT       PostgreSQL port (default: 5432)"
            echo "  --num_workers N      Number of parallel workers (default: 8)"
            echo "  --dry_run            Test without PostgreSQL execution"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prediction file
if [[ -z "$PRED_FILE" ]]; then
    echo "Error: --pred is required"
    exit 1
fi

if [[ ! -f "$PRED_FILE" ]]; then
    echo "Error: Prediction file not found: $PRED_FILE"
    exit 1
fi

# Auto-create subfolder based on prediction filename
PRED_BASENAME=$(basename "$PRED_FILE" .sql)
PRED_BASENAME=${PRED_BASENAME%_detailed}  # Remove _detailed suffix if present
OUTPUT_DIR="${OUTPUT_DIR}/${PRED_BASENAME}"
mkdir -p "$OUTPUT_DIR"

# Build command based on dataset
if [[ "$DATASET" == "livesqlbench" ]]; then
    # LiveSQLBench uses PostgreSQL via evaluate_crud_baseline.py (baseline evaluation)
    # This evaluates pure Text2SQL accuracy without RBAC constraints
    CMD="python -m rbac-exp.evaluation.evaluate_crud_baseline \
        --pred \"$PRED_FILE\" \
        --db_host \"$DB_HOST\" \
        --db_port \"$DB_PORT\" \
        --db_user \"$DB_USER\" \
        --db_password \"$DB_PASSWORD\" \
        --output_dir \"$OUTPUT_DIR\""
else
    # Spider/Bird use SQLite via evaluate_column_baseline.py
    CMD="python rbac-exp/evaluation/evaluate_column_baseline.py \
        --pred \"$PRED_FILE\" \
        --dataset \"$DATASET\" \
        --output_dir \"$OUTPUT_DIR\""
    
    if [[ "$PLUG_VALUE" == "true" ]]; then
        CMD="$CMD --plug_value"
    fi
fi

# Print configuration
echo "========================================="
echo "Baseline Text2SQL Evaluation"
echo "========================================="
echo "Dataset: $DATASET"
echo "Prediction file: $PRED_FILE"
echo "Output dir: $OUTPUT_DIR"
if [[ "$DATASET" == "livesqlbench" ]]; then
    echo "PostgreSQL: $DB_USER@$DB_HOST:$DB_PORT"
    echo "Workers: $NUM_WORKERS"
    echo "Dry run: $DRY_RUN"
fi
echo "========================================="

# Run evaluation
echo "Running: $CMD"
eval $CMD
