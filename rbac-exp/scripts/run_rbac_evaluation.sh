#!/usr/bin/env bash
# Versioned offline RBAC replay. Requires an explicit dataset and schema source.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARGS=()
while (($#)); do
    case "$1" in
        --prediction|--pred) ARGS+=(--prediction_path "$2"); shift 2 ;;
        --dataset)
            if [[ "$2" == "livesqlbench" ]]; then ARGS+=(--crud)
            else ARGS+=(--dataset "$2"); fi
            shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
exec python "${SCRIPT_DIR}/../evaluation/evaluate_protocol_v2.py" "${ARGS[@]}"
