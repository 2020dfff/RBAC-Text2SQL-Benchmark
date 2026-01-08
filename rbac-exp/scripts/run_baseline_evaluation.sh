#!/bin/bash
# Baseline Text2SQL Evaluation Script
# Evaluates SQL accuracy without RBAC metrics
#
# This script uses the _detailed.json file from predictions,
# which already contains SystemManager role items with valid gold_sql.
# No separate data file is needed.

set -e

# Default configuration
DATASET="spider"  # spider, bird, livesqlbench
PRED_FILE=""
OUTPUT_DIR="rbac-exp/output/eval_result"
PLUG_VALUE="false"

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET    Dataset name (spider, bird, livesqlbench)"
            echo "  --pred PRED_FILE     Path to prediction file (.sql or _detailed.json)"
            echo "  --output_dir DIR     Output directory for results"
            echo "  --plug_value         Enable value plugging (slow!)"
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

# Build command (no --data needed, uses _detailed.json directly)
CMD="python rbac-exp/evaluation/evaluate_column_baseline.py \
    --pred \"$PRED_FILE\" \
    --dataset \"$DATASET\" \
    --output_dir \"$OUTPUT_DIR\""

if [[ "$PLUG_VALUE" == "true" ]]; then
    CMD="$CMD --plug_value"
fi

# Print configuration
echo "========================================="
echo "Baseline Text2SQL Evaluation"
echo "========================================="
echo "Dataset: $DATASET"
echo "Prediction file: $PRED_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

# Run evaluation
echo "Running: $CMD"
eval $CMD
