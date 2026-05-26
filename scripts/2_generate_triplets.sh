#!/usr/bin/env bash
# Step 2: Generate triplets for all three factors.
#
# Prerequisites:
#   export MOISESDB_ROOT=/path/to/moisesdb
#   export JASCO_ROOT=/path/to/jasco-audiocraft   # for melody & rhythm only
#
# Adjust NUM_TRIPLETS to your desired dataset size. Paper used 5000 melody,
# 5000 rhythm, and 1855 timbre triplets.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MOISESDB_ROOT:?ERROR: MOISESDB_ROOT is not set}"

NUM_TRIPLETS=${NUM_TRIPLETS:-5000}
MEL_OUT=${MEL_OUT:-./data/melody_triplets}
RHY_OUT=${RHY_OUT:-./data/rhythm_triplets}
TIM_OUT=${TIM_OUT:-./data/timbre_triplets}

echo "=== Generating melody triplets ($NUM_TRIPLETS) ==="
python data_pipeline/melody_triplets_generator/pipeline.py "$NUM_TRIPLETS" \
    --output-dir "$MEL_OUT"

echo "=== Generating rhythm triplets ($NUM_TRIPLETS) ==="
python data_pipeline/rhythm_triplets_generator/pipeline.py "$NUM_TRIPLETS" \
    --output-dir "$RHY_OUT"

echo "=== Generating timbre triplets (all available) ==="
python data_pipeline/timbre_triplets_generator/pipeline.py 99999 \
    --output-dir "$TIM_OUT"

echo "Done. Triplets written to $MEL_OUT, $RHY_OUT, $TIM_OUT"
