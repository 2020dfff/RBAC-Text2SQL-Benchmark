#!/usr/bin/env python3
"""
Convert BIRD JSON data to gold standard format required by evaluation
Creates SQL\tDB_ID format files from JSON input
"""
import json
import argparse
import os

def convert_bird_json_to_gold(input_json_path, output_gold_path):
    """
    Convert BIRD JSON data to gold standard format
    
    Args:
        input_json_path: Path to BIRD JSON file
        output_gold_path: Path to output gold file (SQL\tDB_ID format)
    """
    
    print(f"Loading BIRD JSON from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_gold_path), exist_ok=True)
    
    with open(output_gold_path, 'w', encoding='utf-8') as f:
        for item in data:
            db_id = item.get('db_id', '')
            output_sql = item.get('output', '')
            
            # Handle role-based data where output might be "Sorry, I cannot answer."
            # In this case, we still need to provide the ground truth SQL for evaluation
            if 'Sorry' in output_sql:
                # For role-based data, we might need to find the corresponding non-role SQL
                # For now, we'll keep the "Sorry" response as is
                pass
            
            # Write in format: SQL\tDB_ID
            f.write(f"{output_sql}\t{db_id}\n")
    
    print(f"Gold standard file saved to: {output_gold_path}")
    print(f"Format: SQL\\tDB_ID (one line per sample)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BIRD JSON to gold standard format")
    parser.add_argument("--input", required=True, help="Input BIRD JSON file path")
    parser.add_argument("--output", required=True, help="Output gold standard file path")
    
    args = parser.parse_args()
    convert_bird_json_to_gold(args.input, args.output)