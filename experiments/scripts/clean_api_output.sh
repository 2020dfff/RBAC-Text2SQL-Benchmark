#!/bin/bash

# Clean API output script
# This script processes cloud prediction outputs and cleans them for evaluation

# Default file paths
INPUT_FILE="experiments/output/pred/pred_qwen2.5-coder-7b-bird-role.sql"
OUTPUT_FILE="experiments/output/pred/pred_qwen2.5-coder-7b-bird-role_cleaned.sql"
CLEANER_MODE="default"  # default or snowflake

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m|--mode)
            CLEANER_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -i, --input FILE    Input prediction file (default: pred_qwen2.5-14b-spider-role.sql)"
            echo "  -o, --output FILE   Output cleaned file (default: pred_qwen2.5-14b-spider-role_cleaned.sql)"
            echo "  -m, --mode MODE     Cleaner mode: 'default' or 'snowflake' (default: default)"
            echo "                      - default:   Use cloud_output_cleaner (for general cloud model outputs)"
            echo "                      - snowflake: Use snowflake_output_cleaner (extracts last SQL code block)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Validate cleaner mode
if [[ "$CLEANER_MODE" != "default" && "$CLEANER_MODE" != "snowflake" ]]; then
    echo "Error: Invalid cleaner mode '$CLEANER_MODE'. Must be 'default' or 'snowflake'."
    exit 1
fi

# Run the cleaning process
echo "Cleaning API output..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Cleaner mode: $CLEANER_MODE"
echo ""

if [[ "$CLEANER_MODE" == "snowflake" ]]; then
    # Use snowflake_output_cleaner for Snowflake models
    echo "Using Snowflake output cleaner (extracts last SQL code block)..."
    python3 experiments/data_process/snowflake_output_cleaner.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE"
else
    # Use default cloud_output_cleaner for general cloud models
    echo "Using default cloud output cleaner..."
    python3 -c "from experiments.data_process.cloud_output_cleaner import process_cloud_predictions; print(process_cloud_predictions('$INPUT_FILE', '$OUTPUT_FILE'))"
fi

# Check if the cleaning was successful
if [[ $? -eq 0 ]]; then
    echo " Cleaning completed successfully!"
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo " Output file created: $OUTPUT_FILE"
        echo " File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo " Line count: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo " Cleaning failed!"
    exit 1
fi