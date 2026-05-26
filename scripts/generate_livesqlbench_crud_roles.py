"""
Generate a fresh round of CRUD-level role assignments for LiveSQLBench.
Replicates the logic from notebooks/livesqlbench_crud_rbac.ipynb Section 3.

Usage:
    conda activate llm4db
    python scripts/generate_livesqlbench_crud_roles.py
"""
import sys
import json
import os
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

# Setup project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.llm_oracle import Oracle
from configs.crud_level_prompts import (
    CRUD_LEVEL_SYSTEM_PROMPT,
    CRUD_LEVEL_USER_PROMPT_TEMPLATE,
    DEFAULT_MAX_TOKENS
)
import jsonlines

# ============================================================
# 1. Initialize LLM
# ============================================================
if os.getenv('DEEPSEEK_API_KEY'):
    MODEL = "deepseek-chat"
    API_KEY = os.getenv('DEEPSEEK_API_KEY')
elif os.getenv('OPENAI_API_KEY'):
    MODEL = "gpt-4o-mini"
    API_KEY = os.getenv('OPENAI_API_KEY')
else:
    raise RuntimeError("No API key available (DEEPSEEK_API_KEY or OPENAI_API_KEY)")

oracle = Oracle(model=MODEL, apikey=API_KEY)
print(f"Using model: {MODEL}")

# ============================================================
# 2. Load data
# ============================================================
DATA_DIR = project_root / 'data' / 'livesqlbench-full-postgresql'
OUTPUT_DIR = project_root / 'outputs' / 'livesqlbench_crud'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DATA_PATH = DATA_DIR / 'livesqlbench_data.jsonl'
GT_DATA_PATH = DATA_DIR / 'livesqlbench_base_full_v1_gt_kg_testcases_0904.jsonl'

# Load and merge data
main_data = {}
with jsonlines.open(MAIN_DATA_PATH) as reader:
    for item in reader:
        instance_id = item.get('instance_id')
        if instance_id:
            main_data[instance_id] = item

gt_data = {}
with jsonlines.open(GT_DATA_PATH) as reader:
    for item in reader:
        instance_id = item.get('instance_id')
        if instance_id:
            gt_data[instance_id] = item

raw_data = []
for instance_id, main in main_data.items():
    gt = gt_data.get(instance_id, {})
    merged = {**main, **gt}
    db_id = main.get('selected_database') or '_'.join(instance_id.rsplit('_', 1)[:-1])
    merged['db_id'] = db_id
    raw_data.append(merged)

print(f"Loaded {len(raw_data)} entries")

# Get unique databases
unique_dbs = sorted(set(e['db_id'] for e in raw_data))
print(f"Unique databases: {len(unique_dbs)}")

# ============================================================
# 3. Load HuggingFace schema texts
# ============================================================
unique_base_db_ids = set()
for e in raw_data:
    db_id = e['db_id']
    base_db_id = db_id[:-2] if db_id.endswith('_M') else db_id
    unique_base_db_ids.add(base_db_id)

schema_texts = {}
for db_id in sorted(unique_base_db_ids):
    db_dir = DATA_DIR / db_id
    if not db_dir.exists():
        print(f"  Warning: directory not found for {db_id}")
        continue
    schema_file = db_dir / f"{db_id}_schema.txt"
    if schema_file.exists():
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_texts[db_id] = f.read()

print(f"Loaded schema texts for {len(schema_texts)} databases")

# ============================================================
# 4. Generate CRUD-level roles via LLM
# ============================================================
target_dbs = [db for db in unique_dbs if schema_texts.get(db[:-2] if db.endswith('_M') else db)]
print(f"\nTarget databases: {len(target_dbs)}")
print(f"Databases: {target_dbs}")

role_assignments = {}
errors = []

for db_id in target_dbs:
    base_db_id = db_id[:-2] if db_id.endswith('_M') else db_id
    schema_text = schema_texts.get(base_db_id, '')
    if not schema_text:
        print(f"Skipping {db_id}: no schema_text")
        continue

    print(f"\nGenerating roles for {db_id}...")
    user_prompt = CRUD_LEVEL_USER_PROMPT_TEMPLATE.format(schema_content=schema_text)

    result = oracle.query(
        prompt_sys=CRUD_LEVEL_SYSTEM_PROMPT,
        prompt_user=user_prompt,
        max_completion_tokens=DEFAULT_MAX_TOKENS
    )
    response = result.get('answer', '')

    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            roles = json.loads(json_match.group())
            role_assignments[db_id] = roles
            print(f"  Generated {len(roles)} roles: {[r.get('role','?') for r in roles]}")
        else:
            print(f"  ERROR: No JSON array found in response")
            errors.append(db_id)
    except json.JSONDecodeError as e:
        print(f"  ERROR: JSON parse failed: {e}")
        errors.append(db_id)

# ============================================================
# 5. Save results
# ============================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = OUTPUT_DIR / f'crud_role_assignments_llm_{timestamp}.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(role_assignments, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Generation complete!")
print(f"  Successful: {len(role_assignments)}/{len(target_dbs)} databases")
print(f"  Errors: {len(errors)} ({errors})")
print(f"  Saved to: {output_path}")
print(f"{'='*60}")

# Quick stats
for db_id, roles in sorted(role_assignments.items()):
    role_names = [r.get('role', '?') for r in roles]
    print(f"  {db_id}: {len(roles)} roles - {role_names}")
