#!/usr/bin/env python3
"""
Generate v3 training data: replace SystemManager with DataOperator scoped admins.

Same logic as v3 eval dataset:
- Remove all SystemManager items
- For each DB, create 2-3 DataOperator roles with overlapping column-level permissions
- For each original SM question, create items for each DataOperator
- Determine allow/deny based on whether the DataOperator has access to required columns
"""
import json
import random
import copy
import os
import sys
from collections import defaultdict

random.seed(42)

def get_db_schema(items, db_id):
    """Extract full schema (all tables + columns) from existing items for a DB."""
    schema = defaultdict(set)
    for item in items:
        if item['db_id'] != db_id:
            continue
        policy = item.get('policy', {})
        for table, cols in policy.items():
            if isinstance(cols, list):
                if cols == ['*']:
                    # Get columns from metadata or instruction
                    pass  # Will handle below
                else:
                    for c in cols:
                        schema[table].add(c)

    # Also try to get full column list from SM items (SM has '*' for all tables)
    # Parse from instruction field
    for item in items:
        if item['db_id'] != db_id:
            continue
        instruction = item.get('instruction', '')
        # Parse "Table X has columns such as A, B, C."
        import re
        for match in re.finditer(r'Table (\w+) has columns such as ([^.]+)\.', instruction):
            table = match.group(1)
            cols = [c.strip() for c in match.group(2).split(',')]
            for c in cols:
                schema[table].add(c)

    return {t: sorted(list(cols)) for t, cols in schema.items()}


def generate_dataoperator_policies(schema, n_operators=None, coverage_prob=0.7, table_star_prob=0.5):
    """Generate overlapping DataOperator policies for a DB schema."""
    tables = list(schema.keys())

    if n_operators is None:
        n_operators = 3 if len(tables) >= 3 else 2

    policies = {}
    for i in range(n_operators):
        policy = {}
        for table in tables:
            cols = schema[table]
            # 50% chance of table-level access ('*')
            if random.random() < table_star_prob:
                policy[table] = ['*']
            else:
                # Column-level: each column has coverage_prob chance of being included
                selected = [c for c in cols if random.random() < coverage_prob]
                # Ensure at least 1 column
                if not selected and cols:
                    selected = [random.choice(cols)]
                # Cap at 90% of columns
                max_cols = max(1, int(len(cols) * 0.9))
                if len(selected) > max_cols:
                    selected = random.sample(selected, max_cols)
                policy[table] = sorted(selected)

        policies[f'DataOperator_{i+1}'] = policy

    return policies


def check_permission(policy, metadata):
    """Check if a DataOperator policy allows the query based on required columns."""
    query_columns = metadata.get('query_columns', {})
    missing_columns = metadata.get('missing_columns', {})
    gold_sql = metadata.get('gold_sql', '')

    # Get the required columns from the query
    # A query is allowed if all required columns are accessible
    for table, needed_cols in query_columns.items():
        if table not in policy:
            return False, {table: needed_cols}

        policy_cols = policy[table]
        if policy_cols == ['*']:
            continue  # Full access to this table

        # Check each needed column
        for col in needed_cols:
            if col not in policy_cols:
                return False, {table: [col]}

    # Also check if the original gold_sql uses any tables not in policy
    # (simplified: if all query_columns tables are in policy, it's ok)
    return True, {}


def build_instruction(original_instruction, role_name, policy):
    """Rebuild instruction field with new role and policy."""
    import re

    # Replace the role name
    instruction = re.sub(
        r'Role: \w+',
        f'Role: {role_name}',
        original_instruction
    )

    # Replace the accessible columns section
    # Find the "Accessible Columns:" section and replace it
    policy_lines = []
    for table, cols in sorted(policy.items()):
        if cols == ['*']:
            policy_lines.append(f'  - {table}: * (all columns)')
        else:
            policy_lines.append(f'  - {table}: {", ".join(cols)}')

    policy_text = '\n'.join(policy_lines)

    # Replace everything after "Accessible Columns:" until the end or next section
    instruction = re.sub(
        r'Accessible Columns:.*',
        f'Accessible Columns:\n{policy_text}',
        instruction,
        flags=re.DOTALL
    )

    return instruction


def main():
    input_path = 'data/selected/spider/column_level_rbac_dataset_spider_train_20260113.json'
    output_path = 'data/selected/spider/column_level_rbac_dataset_spider_train_v3_no_sm.json'

    print(f'Loading: {input_path}')
    data = json.load(open(input_path))
    print(f'Total items: {len(data)}')

    # Separate SM and non-SM items
    sm_items = [item for item in data if item.get('role', '') == 'SystemManager']
    non_sm_items = [item for item in data if item.get('role', '') != 'SystemManager']
    print(f'SM items: {len(sm_items)}, Non-SM items: {len(non_sm_items)}')

    # Group SM items by DB
    sm_by_db = defaultdict(list)
    for item in sm_items:
        sm_by_db[item['db_id']].append(item)
    print(f'Unique DBs with SM: {len(sm_by_db)}')

    # Get full schema for each DB
    db_schemas = {}
    for db_id in sm_by_db:
        db_schemas[db_id] = get_db_schema(data, db_id)

    # Generate DataOperator policies for each DB
    db_do_policies = {}
    for db_id, schema in db_schemas.items():
        db_do_policies[db_id] = generate_dataoperator_policies(schema)

    # Generate DataOperator items
    do_items = []
    for db_id, sm_db_items in sm_by_db.items():
        policies = db_do_policies[db_id]

        for role_name, policy in policies.items():
            for sm_item in sm_db_items:
                new_item = copy.deepcopy(sm_item)
                new_item['role'] = role_name
                new_item['policy'] = policy

                # Check permission
                metadata = sm_item.get('metadata', {})
                allowed, missing = check_permission(policy, metadata)

                if allowed:
                    new_item['output'] = sm_item['output']  # Keep original SQL
                    new_item['metadata']['permission'] = 'allowed'
                    new_item['metadata']['missing_columns'] = {}
                else:
                    new_item['output'] = 'Sorry, I cannot answer.'
                    new_item['metadata']['permission'] = 'denied'
                    new_item['metadata']['missing_columns'] = missing

                # Rebuild instruction with new role/policy
                new_item['instruction'] = build_instruction(
                    sm_item['instruction'], role_name, policy
                )

                do_items.append(new_item)

    # Combine: non-SM items + DataOperator items
    v3_data = non_sm_items + do_items

    # Stats
    allow = sum(1 for item in v3_data if item.get('output', '') != 'Sorry, I cannot answer.')
    deny = len(v3_data) - allow
    do_allow = sum(1 for item in do_items if item.get('output', '') != 'Sorry, I cannot answer.')
    do_deny = len(do_items) - do_allow

    print(f'\nV3 training data:')
    print(f'  Non-SM items: {len(non_sm_items)}')
    print(f'  DataOperator items: {len(do_items)} (allow={do_allow}, deny={do_deny})')
    print(f'  Total: {len(v3_data)}')
    print(f'  Allow: {allow} ({allow*100/len(v3_data):.1f}%)')
    print(f'  Deny: {deny} ({deny*100/len(v3_data):.1f}%)')

    with open(output_path, 'w') as f:
        json.dump(v3_data, f, ensure_ascii=False)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
