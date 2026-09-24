#!/usr/bin/env bash
# RBAC evaluation (column-level: spider/bird; CRUD-level: livesqlbench).
# Usage: bash scripts/run_rbac_evaluation.sh --prediction <pred.sql> --dataset <spider|bird|livesqlbench> [options]
# Options are passed to evaluation/evaluate_protocol_v2.py, e.g. --role_json, --db_dir, --output_dir,
# --fair_comparison, --num_trials, --execute_sql, --db_user/--db_password/--db_host/--db_port (LiveSQLBench).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARGS=()
while (($#)); do
    case "$1" in
        --prediction|--pred) ARGS+=(--prediction_path "$2"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
exec python "${SCRIPT_DIR}/../evaluation/evaluate_protocol_v2.py" "${ARGS[@]}"
