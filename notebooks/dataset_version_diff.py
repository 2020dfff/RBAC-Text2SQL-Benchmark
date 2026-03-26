"""
Dataset Version Diff Analysis — for reviewer response
Compare all versions of Spider, BIRD, and LiveSQLBench datasets.
Report which db_id/role/policy were modified and why.
"""

import json
import os
from collections import defaultdict, Counter

BASE = "/home/feiy/Role-SQL-benchmark"

def load_json(path):
    print(f"Loading {os.path.basename(path)} ...")
    with open(path, 'r') as f:
        return json.load(f)

def make_key(item):
    """Create a unique key: (db_id, question_text, role)."""
    q = item.get('input', item.get('question', ''))
    r = item.get('role', '')
    return (item['db_id'], q, r)

def diff_policy(old_policy, new_policy):
    """Compare two policy dicts, return added/removed columns per table."""
    changes = {}
    all_tables = set(list(old_policy.keys()) + list(new_policy.keys()))
    for t in all_tables:
        old_cols = set(old_policy.get(t, []))
        new_cols = set(new_policy.get(t, []))
        added = new_cols - old_cols
        removed = old_cols - new_cols
        if added or removed:
            changes[t] = {'added': list(added), 'removed': list(removed)}
    return changes

def compare_datasets(old_data, new_data, name):
    """Compare two column-level RBAC datasets at item level."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"Total items: OLD={len(old_data)}, NEW={len(new_data)}")

    old_idx = {}
    for item in old_data:
        key = make_key(item)
        old_idx[key] = item
    new_idx = {}
    for item in new_data:
        key = make_key(item)
        new_idx[key] = item

    old_keys = set(old_idx.keys())
    new_keys = set(new_idx.keys())
    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    print(f"\nKey-level: added={len(added_keys)}, removed={len(removed_keys)}, common={len(common_keys)}")

    # ---- Analyze common items ----
    role_changes = []
    policy_changes = []
    perm_changes = []
    output_changes = []

    for key in common_keys:
        old = old_idx[key]
        new = new_idx[key]

        if old.get('role') != new.get('role'):
            role_changes.append({
                'db_id': key[0], 'question': key[1][:60],
                'old_role': old.get('role'), 'new_role': new.get('role')
            })

        old_p = json.dumps(old.get('policy', {}), sort_keys=True)
        new_p = json.dumps(new.get('policy', {}), sort_keys=True)
        if old_p != new_p:
            pd = diff_policy(old.get('policy', {}), new.get('policy', {}))
            policy_changes.append({
                'db_id': key[0], 'question': key[1][:60],
                'role': new.get('role', old.get('role')),
                'policy_diff': pd
            })

        old_perm = old.get('metadata', {}).get('permission')
        new_perm = new.get('metadata', {}).get('permission')
        if old_perm != new_perm:
            perm_changes.append({
                'db_id': key[0], 'question': key[1][:60],
                'old_perm': old_perm, 'new_perm': new_perm,
                'role': new.get('role', old.get('role'))
            })

        if old.get('output') != new.get('output'):
            output_changes.append({
                'db_id': key[0], 'question': key[1][:60],
                'old_output': str(old.get('output', ''))[:80],
                'new_output': str(new.get('output', ''))[:80],
            })

    print(f"\n--- Changes in {len(common_keys)} common items ---")
    print(f"  Role changed:       {len(role_changes)}")
    print(f"  Policy changed:     {len(policy_changes)}")
    print(f"  Permission changed: {len(perm_changes)}")
    print(f"  Output changed:     {len(output_changes)}")

    # ---- DB-level summary ----
    affected_dbs = set()
    for c in role_changes: affected_dbs.add(c['db_id'])
    for c in policy_changes: affected_dbs.add(c['db_id'])
    for c in perm_changes: affected_dbs.add(c['db_id'])

    all_dbs = set(item['db_id'] for item in new_data)
    print(f"\n  Affected DB IDs: {len(affected_dbs)}/{len(all_dbs)}")
    print(f"  List: {sorted(affected_dbs)}")

    # ---- Permission flip details ----
    if perm_changes:
        a2d = [c for c in perm_changes if c['old_perm'] == 'allowed' and c['new_perm'] == 'denied']
        d2a = [c for c in perm_changes if c['old_perm'] == 'denied' and c['new_perm'] == 'allowed']
        print(f"\n  Permission flips: allowed→denied={len(a2d)}, denied→allowed={len(d2a)}")

        if a2d:
            print(f"\n  Examples allowed→denied:")
            db_counts = Counter(c['db_id'] for c in a2d)
            for db, cnt in db_counts.most_common():
                print(f"    {db}: {cnt} items")
                examples = [c for c in a2d if c['db_id'] == db][:2]
                for e in examples:
                    print(f"      Q: {e['question']}")
                    print(f"      Role: {e['role']}")

        if d2a:
            print(f"\n  Examples denied→allowed:")
            db_counts = Counter(c['db_id'] for c in d2a)
            for db, cnt in db_counts.most_common():
                print(f"    {db}: {cnt} items")
                examples = [c for c in d2a if c['db_id'] == db][:2]
                for e in examples:
                    print(f"      Q: {e['question']}")
                    print(f"      Role: {e['role']}")

    # ---- Policy change details ----
    if policy_changes:
        print(f"\n  Policy changes by DB:")
        db_policy = defaultdict(list)
        for c in policy_changes:
            db_policy[c['db_id']].append(c)
        for db_id, changes in sorted(db_policy.items()):
            # Aggregate: which tables had columns added/removed
            table_changes = defaultdict(lambda: {'added': set(), 'removed': set()})
            roles_affected = set()
            for c in changes:
                roles_affected.add(c['role'])
                for table, diff in c['policy_diff'].items():
                    table_changes[table]['added'].update(diff['added'])
                    table_changes[table]['removed'].update(diff['removed'])
            print(f"    {db_id}: {len(changes)} items, {len(roles_affected)} roles")
            for table, diff in sorted(table_changes.items()):
                parts = []
                if diff['added']: parts.append(f"+{list(diff['added'])}")
                if diff['removed']: parts.append(f"-{list(diff['removed'])}")
                print(f"      {table}: {', '.join(parts)}")

    # ---- Added items ----
    if added_keys:
        print(f"\n  Added items ({len(added_keys)}) by DB:")
        db_counts = Counter(k[0] for k in added_keys)
        for db, cnt in db_counts.most_common():
            print(f"    {db}: +{cnt}")

    if removed_keys:
        print(f"\n  Removed items ({len(removed_keys)}) by DB:")
        db_counts = Counter(k[0] for k in removed_keys)
        for db, cnt in db_counts.most_common():
            print(f"    {db}: -{cnt}")

    return {
        'added': len(added_keys), 'removed': len(removed_keys),
        'common': len(common_keys),
        'role_changed': len(role_changes),
        'policy_changed': len(policy_changes),
        'perm_changed': len(perm_changes),
        'output_changed': len(output_changes),
        'affected_dbs': sorted(affected_dbs),
        'total_dbs': len(all_dbs),
    }


def compare_livesqlbench(old_data, new_data, name):
    """Compare LiveSQLBench CRUD datasets."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"Total items: OLD={len(old_data)}, NEW={len(new_data)}")

    # Key: (db_id, question, operation)
    def lsb_key(item):
        return (item['db_id'], item.get('question', item.get('input', '')), item.get('operation', ''))

    old_idx = {lsb_key(item): item for item in old_data}
    new_idx = {lsb_key(item): item for item in new_data}

    old_keys = set(old_idx.keys())
    new_keys = set(new_idx.keys())
    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    print(f"Key-level: added={len(added)}, removed={len(removed)}, common={len(common)}")

    # Compare common
    role_changes = []
    policy_changes = []
    perm_changes = []
    output_changes = []

    for key in common:
        old = old_idx[key]
        new = new_idx[key]

        if old.get('role') != new.get('role'):
            role_changes.append({'db_id': key[0], 'op': key[2],
                'old_role': old.get('role'), 'new_role': new.get('role')})

        old_p = json.dumps(old.get('policy', {}), sort_keys=True)
        new_p = json.dumps(new.get('policy', {}), sort_keys=True)
        if old_p != new_p:
            policy_changes.append({'db_id': key[0], 'op': key[2],
                'role': new.get('role', old.get('role'))})

        old_allowed = old.get('allowed', old.get('metadata', {}).get('permission'))
        new_allowed = new.get('allowed', new.get('metadata', {}).get('permission'))
        if old_allowed != new_allowed:
            perm_changes.append({'db_id': key[0], 'op': key[2],
                'old': old_allowed, 'new': new_allowed})

        if old.get('output') != new.get('output'):
            output_changes.append({'db_id': key[0], 'op': key[2]})

    print(f"\n--- Changes in {len(common)} common items ---")
    print(f"  Role changed:       {len(role_changes)}")
    print(f"  Policy changed:     {len(policy_changes)}")
    print(f"  Permission changed: {len(perm_changes)}")
    print(f"  Output changed:     {len(output_changes)}")

    affected_dbs = set()
    for c in role_changes: affected_dbs.add(c['db_id'])
    for c in policy_changes: affected_dbs.add(c['db_id'])
    for c in perm_changes: affected_dbs.add(c['db_id'])

    all_dbs = set(item['db_id'] for item in new_data)
    print(f"\n  Affected DB IDs: {len(affected_dbs)}/{len(all_dbs)}")

    if perm_changes:
        print(f"\n  Permission flips by DB:")
        db_counts = Counter(c['db_id'] for c in perm_changes)
        for db, cnt in db_counts.most_common(10):
            print(f"    {db}: {cnt}")

    if added:
        print(f"\n  Added: {len(added)} items across {len(set(k[0] for k in added))} DBs")
    if removed:
        print(f"\n  Removed: {len(removed)} items across {len(set(k[0] for k in removed))} DBs")

    return {
        'added': len(added), 'removed': len(removed), 'common': len(common),
        'role_changed': len(role_changes), 'policy_changed': len(policy_changes),
        'perm_changed': len(perm_changes), 'output_changed': len(output_changes),
        'affected_dbs': sorted(affected_dbs), 'total_dbs': len(all_dbs),
    }


if __name__ == '__main__':
    results = {}

    # ============================================================
    # Spider: V1(1225) → V2(1229) → V3(0113)
    # ============================================================
    spider_v1 = load_json(f"{BASE}/outputs/column_level_rbac_dataset_spider_20251225.json")
    spider_v2 = load_json(f"{BASE}/outputs/column_level_rbac_dataset_spider_20251229.json")
    spider_v3 = load_json(f"{BASE}/data/selected/spider/column_level_rbac_dataset_spider_20260113.json")

    results['spider_v1_v2'] = compare_datasets(spider_v1, spider_v2, "Spider V1(12-25) → V2(12-29)")
    results['spider_v2_v3'] = compare_datasets(spider_v2, spider_v3, "Spider V2(12-29) → V3(01-13, current)")

    # ============================================================
    # BIRD: V1(1224_v2) → V2(1230)
    # ============================================================
    bird_v1 = load_json(f"{BASE}/outputs/column_level_rbac_dataset_bird_20251224_v2.json")
    bird_v2 = load_json(f"{BASE}/data/selected/bird/column_level_rbac_dataset_bird_20251230.json")

    results['bird_v1_v2'] = compare_datasets(bird_v1, bird_v2, "BIRD V1(12-24) → V2(12-30, current)")

    # ============================================================
    # LiveSQLBench: earliest full version → v2 → 20260314
    # ============================================================
    # Find the earliest "full" version (131M+)
    lsb_v1 = load_json(f"{BASE}/outputs/livesqlbench_crud/crud_rbac_dataset_llm_20251226_142436.json")
    lsb_v2 = load_json(f"{BASE}/outputs/livesqlbench_crud/crud_rbac_dataset_v2_20251230.json")
    lsb_v3 = load_json(f"{BASE}/outputs/livesqlbench_crud/crud_rbac_dataset_llm_20260314_232655.json")

    results['lsb_v1_v2'] = compare_livesqlbench(lsb_v1, lsb_v2, "LiveSQLBench V1(12-26) → V2(12-30, current)")
    results['lsb_v2_v3'] = compare_livesqlbench(lsb_v2, lsb_v3, "LiveSQLBench V2(12-30) → V3(03-14, re-generated)")

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Comparison':<45} {'Added':>6} {'Removed':>8} {'RoleΔ':>6} {'PolicyΔ':>8} {'PermΔ':>6} {'DBs':>10}")
    print(f"{'-'*45} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*10}")
    for key, r in results.items():
        label = key.replace('_', ' ').upper()
        print(f"{label:<45} {r['added']:>6} {r['removed']:>8} {r['role_changed']:>6} {r['policy_changed']:>8} {r['perm_changed']:>6} {len(r['affected_dbs']):>3}/{r['total_dbs']:<6}")
