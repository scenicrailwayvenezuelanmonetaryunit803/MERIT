#!/usr/bin/env bash
# Step 4: Train MLP projection heads for all three factors.
#
# Uses embeddings from Step 3. Trains with Circle Loss (γ=10, m=0.2),
# AdamW (lr=1e-3), 200 epochs, cosine LR schedule.
set -euo pipefail
cd "$(dirname "$0")/.."

EMBEDDINGS_DIR=${EMBEDDINGS_DIR:-./data/embeddings}
MODELS_DIR=${MODELS_DIR:-./models}

mkdir -p "$MODELS_DIR/head_mel" "$MODELS_DIR/head_rhy" "$MODELS_DIR/head_tim"

echo "=== Training melody head ==="
python training/train_head.py \
    --embeddings "$EMBEDDINGS_DIR/mel_mert.pkl" \
    --out "$MODELS_DIR/head_mel/best_head.pt" \
    --epochs 200

echo "=== Training rhythm head ==="
python training/train_head.py \
    --embeddings "$EMBEDDINGS_DIR/rhy_mert.pkl" \
    --out "$MODELS_DIR/head_rhy/best_head.pt" \
    --epochs 200

echo "=== Training timbre head ==="
python training/train_head.py \
    --embeddings "$EMBEDDINGS_DIR/tim_mert.pkl" \
    --out "$MODELS_DIR/head_tim/best_head.pt" \
    --epochs 200

echo "Trained heads saved in $MODELS_DIR"
