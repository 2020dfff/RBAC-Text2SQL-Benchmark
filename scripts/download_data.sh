#!/usr/bin/env bash
# Download the RBAC-augmented datasets from HuggingFace into data/selected/.
#
# The RBAC annotations (role policies + role-conditioned instances) are hosted on
# HuggingFace rather than bundled in this repository. The underlying databases
# (Spider / BIRD / LiveSQLBench) are NOT redistributed here — see README §2.1.
#
# Usage:  bash scripts/download_data.sh
set -euo pipefail

HF_REPO="${HF_REPO:-2020dfff/RBAC-Text2SQL-Benchmark}"
DEST="${DEST:-data/selected}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install with:  pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

mkdir -p "$DEST"
echo "Downloading ${HF_REPO} -> ${DEST}"
huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir "$DEST"

echo ""
echo "Done. Datasets:"
for f in \
    "$DEST/spider/column_level_rbac_dataset_spider_v3_no_sm.json" \
    "$DEST/bird/column_level_rbac_dataset_bird_v3_no_sm.json" \
    "$DEST/livesqlbench-full/crud_rbac_dataset_v3_no_sm.json" ; do
    [ -f "$f" ] && echo "  ok   $f" || echo "  MISSING $f"
done
echo ""
echo "Next: download the underlying databases (README section 2.1)."
