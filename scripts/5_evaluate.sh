#!/usr/bin/env bash
# Step 5: Evaluate trained heads on held-out test sets (3×3 disentanglement table).
#
# Runs each trained head against all three factor test sets to produce the
# 3×3 disentanglement table from Table 1 of the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

EMBEDDINGS_DIR=${EMBEDDINGS_DIR:-./data/embeddings}
MODELS_DIR=${MODELS_DIR:-./models}
RESULTS_DIR=${RESULTS_DIR:-./results}

mkdir -p "$RESULTS_DIR"

FACTORS=(mel rhy tim)

for TRAIN_F in "${FACTORS[@]}"; do
    for TEST_F in "${FACTORS[@]}"; do
        echo "=== Evaluating ${TRAIN_F} head on ${TEST_F} test set ==="
        python evaluation/evaluate.py \
            --head "$MODELS_DIR/head_${TRAIN_F}/best_head.pt" \
            --test "$EMBEDDINGS_DIR/${TEST_F}_mert.pkl" \
            --out  "$RESULTS_DIR/${TRAIN_F}_on_${TEST_F}.json"
    done
done

echo "Evaluation results written to $RESULTS_DIR/"
