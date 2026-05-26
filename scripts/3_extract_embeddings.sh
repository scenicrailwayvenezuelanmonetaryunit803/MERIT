#!/usr/bin/env bash
# Step 3: Extract MERT embeddings from triplet directories.
#
# Uses pre-computed train/test splits from splits/.
# Output: one pkl per factor in EMBEDDINGS_DIR.
#
# Set MEL_DIR, RHY_DIR, TIM_DIR to your generated triplet roots.
set -euo pipefail
cd "$(dirname "$0")/.."

MEL_DIR=${MEL_DIR:-./data/melody_triplets}
RHY_DIR=${RHY_DIR:-./data/rhythm_triplets}
TIM_DIR=${TIM_DIR:-./data/timbre_triplets}
EMBEDDINGS_DIR=${EMBEDDINGS_DIR:-./data/embeddings}

mkdir -p "$EMBEDDINGS_DIR"

echo "=== Extracting melody embeddings ==="
python training/extract_embeddings.py \
    --encoder mert \
    --triplets-dir "$MEL_DIR" \
    --out "$EMBEDDINGS_DIR/mel_mert.pkl" \
    --split-file splits/melody_split.json

echo "=== Extracting rhythm embeddings ==="
python training/extract_embeddings.py \
    --encoder mert \
    --triplets-dir "$RHY_DIR" \
    --out "$EMBEDDINGS_DIR/rhy_mert.pkl" \
    --split-file splits/rhythm_split.json

echo "=== Extracting timbre embeddings ==="
python training/extract_embeddings.py \
    --encoder mert \
    --triplets-dir "$TIM_DIR" \
    --out "$EMBEDDINGS_DIR/tim_mert.pkl" \
    --split-file splits/timbre_split.json

echo "Embeddings written to $EMBEDDINGS_DIR"
