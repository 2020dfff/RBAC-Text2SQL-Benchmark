#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snowflake Output Cleaner

Usage:
    python snowflake_output_cleaner.py --input <input_file> --output <output_file>
"""

import re
import argparse
from pathlib import Path
from typing import Optional


def extract_last_sql_block(response: str) -> Optional[str]:
    """
    Args:
        response
        
    Returns:
        extracted SQL statement with leading/trailing whitespace removed,
    """
    if not response or not response.strip():
        return ""
    
    # First, check for "Sorry, I cannot answer" responses in various formats
    # This should be detected early and normalized
    # Use case-insensitive search for sorry/cannot/answer anywhere in the text
    if re.search(r"sorry.{0,50}cannot.{0,50}answer", response, re.IGNORECASE | re.DOTALL):
        return "Sorry, I cannot answer."
    
    # match ```sql ... ``` blocks (multi-line)
    # use re.DOTALL to make . match newlines
    # use non-greedy .*? to ensure matching each separate block
    # Note: code blocks may contain blank lines, so allow whitespace after \n
    sql_blocks = re.findall(
        r'```sql\s*(.*?)```',
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if sql_blocks:
        # select the last matched SQL block
        last_sql = sql_blocks[-1].strip()
        # Check if this SQL block is a "Sorry" response
        if re.search(r"sorry.{0,50}cannot.{0,50}answer", last_sql, re.IGNORECASE | re.DOTALL):
            return "Sorry, I cannot answer."
        # remove any extra whitespace and newlines
        last_sql = ' '.join(last_sql.split())
        return last_sql
    
    # if no ```sql ``` blocks found, try to extract ``` ``` blocks
    code_blocks = re.findall(
        r'```\s*(.*?)```',
        response,
        re.DOTALL
    )
    
    if code_blocks:
        # get the last code block
        last_code = code_blocks[-1].strip()
        # Check if this is a "Sorry" response
        if re.search(r"sorry.{0,50}cannot.{0,50}answer", last_code, re.IGNORECASE | re.DOTALL):
            return "Sorry, I cannot answer."
        # check if it looks like SQL (contains common SQL keywords)
        if re.search(
            r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE|JOIN)\b',
            last_code,
            re.IGNORECASE
        ):
            # clean and return cleaned SQL
            last_code = ' '.join(last_code.split())
            return last_code
    
    # if no code blocks found, or no SQL-like code blocks, try to clean the whole response
    cleaned = response.strip()
    
    # remove "To translate..." and similar prefixes
    cleaned = re.sub(
        r'^To translate.*?natural language.*?:?\s*',
        '',
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # remove markdown headers like # Header
    cleaned = re.sub(r'^#{1,6}\s+.*$', '', cleaned, flags=re.MULTILINE)
    
    # remove empty lines and join into single line
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    cleaned = ' '.join(lines)
    
    return cleaned if cleaned else response.strip()


def clean_snowflake_output(input_file: str, output_file: str) -> None:
    """
    Args:
        input_file
        output_file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {input_file}")
    print(f"Output to: {output_file}")
    
    cleaned_count = 0
    total_count = 0
    
    with input_path.open('r', encoding='utf-8') as infile, \
         output_path.open('w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            original = line.rstrip('\n')
            
            if not original.strip():
                outfile.write('\n')
                continue
            
            # get the last SQL code block
            cleaned_sql = extract_last_sql_block(original)
            
            # 计数 if the cleaned SQL is different from original, count it
            if cleaned_sql != original:
                cleaned_count += 1
        
            single_line_sql = ' '.join(cleaned_sql.split())
            outfile.write(single_line_sql + '\n')
    
    print(f"\n✓ Processing complete!")
    print(f"  Total lines: {total_count}")
    print(f"  Cleaned lines: {cleaned_count}")
    print(f"  Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Snowflake model output by extracting the last SQL code block from each line"
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Input prediction file path (e.g., pred_snowflake-arctic-r1-7b-spider-role.sql)'
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Output cleaned file path (e.g., pred_snowflake-arctic-r1-7b-spider-role_cleaned.sql)'
    )
    
    args = parser.parse_args()
    
    try:
        clean_snowflake_output(args.input, args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
