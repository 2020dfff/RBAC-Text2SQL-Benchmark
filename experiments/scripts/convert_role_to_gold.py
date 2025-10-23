#!/usr/bin/env python3
"""
Convert role-based JSON dataset to original evaluation.py compatible format
Creates a gold.txt file that can be used with the original evaluation script
"""
import json
import argparse

def convert_role_json_to_gold_txt(role_json_file, output_gold_file):
    """Convert role-based JSON to gold.txt format"""
    
    with open(role_json_file, 'r') as f:
        role_data = json.load(f)
    
    print(f"Loading {len(role_data)} role-based samples...")
    
    with open(output_gold_file, 'w') as f:
        for item in role_data:
            db_id = item["db_id"]
            output_sql = item["output"]
            
            # Original evaluation.py expects: "SQL_QUERY\tDB_ID" format
            f.write(f"{output_sql}\t{db_id}\n")
    
    print(f"Created {output_gold_file} with {len(role_data)} entries")
    print(f"Now you can run: python dbgpt_hub_sql/eval/evaluation.py --input your_predictions.sql --gold {output_gold_file} --plug_value")

def main():
    parser = argparse.ArgumentParser(description='Convert role-based JSON to gold.txt format')
    parser.add_argument('--role_json', required=True, 
                       help='Path to role-based JSON file (e.g., example_text2sql_with_role_dev.json)')
    parser.add_argument('--output_gold', required=True,
                       help='Output gold.txt file path')
    
    args = parser.parse_args()
    convert_role_json_to_gold_txt(args.role_json, args.output_gold)

if __name__ == "__main__":
    main()