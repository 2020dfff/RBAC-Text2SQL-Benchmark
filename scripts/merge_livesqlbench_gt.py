#!/usr/bin/env python3
"""Merge LiveSQLBench ground-truth SQL back into the RBAC CRUD dataset.

The LiveSQLBench authors gate ground truth and test cases behind an email request
(to prevent leakage via automated crawling), so this benchmark ships the
LiveSQLBench split with `gold_sql` removed and `output` blanked for ALLOW items
(`gt_redacted: true`). Everything that is ours — role, policy, allowed flag,
denial reason, operation, category — is included.

To evaluate on the CRUD-level split you must first obtain the ground truth
yourself, per the upstream instructions:

    Email  bird.bench25@gmail.com
    Subject: [livesqlbench-base-full-v1 GT&Test Cases]

Then run this script to join the returned ground truth into the RBAC file by
`instance_id`:

    python scripts/merge_livesqlbench_gt.py \\
        --rbac data/selected/livesqlbench-full/crud_rbac_dataset_v3_no_sm.json \\
        --gt   /path/to/livesqlbench_gt.jsonl \\
        --out  data/selected/livesqlbench-full/crud_rbac_dataset_v3_no_sm.gt.json

The GT file may be JSON or JSONL; each record must carry an instance id and a
SQL field (common field names are auto-detected, or pass --id-key/--sql-key).
"""
import argparse
import json
import os

ID_KEYS = ("instance_id", "id", "question_id", "instance")
SQL_KEYS = ("gold_sql", "sql", "ground_truth", "gt_sql", "query")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _pick(record, candidates, override=None):
    if override:
        return override
    for k in candidates:
        if k in record:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbac", required=True, help="GT-redacted RBAC CRUD dataset")
    ap.add_argument("--gt", required=True, help="ground truth from the LiveSQLBench authors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-key", default=None)
    ap.add_argument("--sql-key", default=None)
    args = ap.parse_args()

    rbac = _load(args.rbac)
    gt = _load(args.gt)
    if not gt:
        raise SystemExit("ground truth file is empty")

    id_key = _pick(gt[0], ID_KEYS, args.id_key)
    sql_key = _pick(gt[0], SQL_KEYS, args.sql_key)
    if not id_key or not sql_key:
        raise SystemExit(
            f"could not detect id/sql fields in {args.gt} (keys: {list(gt[0])}); "
            "pass --id-key and --sql-key explicitly"
        )
    gt_by_id = {str(r[id_key]): r[sql_key] for r in gt if r.get(id_key) is not None}
    print(f"loaded {len(gt_by_id)} ground-truth rows (id='{id_key}', sql='{sql_key}')")

    merged = missing = 0
    for item in rbac:
        sql = gt_by_id.get(str(item.get("instance_id")))
        if sql is None:
            missing += 1
            continue
        item["gold_sql"] = sql
        if item.get("allowed"):
            item["output"] = sql          # ALLOW: reference answer is the gold SQL
        item["gt_redacted"] = False
        merged += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rbac, f, ensure_ascii=False)
    print(f"merged {merged}/{len(rbac)} instances -> {args.out}")
    if missing:
        print(f"WARNING: {missing} instances had no matching ground truth")


if __name__ == "__main__":
    main()
