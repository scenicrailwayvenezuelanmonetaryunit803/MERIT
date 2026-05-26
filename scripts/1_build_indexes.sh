#!/usr/bin/env bash
# Step 1: Build MoisesDB input indexes for all three factors.
#
# Prerequisites:
#   export MOISESDB_ROOT=/path/to/moisesdb   # parent of moisesdb_v0.1/
#
# Outputs (in data_pipeline/triplets_input_index/):
#   melody_index.json
#   rhythm_index.json
#   timbre_index.json
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${MOISESDB_ROOT:-}" ]]; then
    echo "ERROR: MOISESDB_ROOT is not set. Export it to your MoisesDB root." >&2
    exit 1
fi

python data_pipeline/triplets_input_index/build_index.py all \
    --out-dir data_pipeline/triplets_input_index

echo "Index files written to data_pipeline/triplets_input_index/"
