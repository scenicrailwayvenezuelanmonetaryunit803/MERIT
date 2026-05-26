#!/usr/bin/env bash
# Step 6: Run external probe evaluations (Table 2 / Table 3 of the paper).
#
# Encodes each probe dataset with MERT, then evaluates all three heads on
# each probe to measure zero-shot timbral, rhythmic, and melodic retrieval.
#
# Prerequisites:
#   export PROBES_ROOT=/path/to/probes
#   Directory layout expected:
#     $PROBES_ROOT/musdb18hq/       (MUSDB18-HQ stems, .wav)
#     $PROBES_ROOT/ballroom/        (Ballroom dataset, subdirs = class name)
#     $PROBES_ROOT/covers80/        (Covers80, subdirs = cover ID, 2 files each)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PROBES_ROOT:?ERROR: PROBES_ROOT is not set. Set it to the directory containing musdb18hq/, ballroom/, covers80/.}"

MODELS_DIR=${MODELS_DIR:-./models}
RESULTS_DIR=${RESULTS_DIR:-./results}
PROBE_EMBEDDINGS_DIR=${PROBE_EMBEDDINGS_DIR:-./data/probe_embeddings}

mkdir -p "$PROBE_EMBEDDINGS_DIR" "$RESULTS_DIR"

# ---- Encode probe datasets ----
echo "=== Encoding MUSDB18-HQ stems ==="
python evaluation/encode_folder.py \
    --audio-dir "$PROBES_ROOT/musdb18hq" \
    --out "$PROBE_EMBEDDINGS_DIR/musdb18hq.pkl"

echo "=== Encoding Ballroom dataset ==="
python evaluation/encode_folder.py \
    --audio-dir "$PROBES_ROOT/ballroom" \
    --out "$PROBE_EMBEDDINGS_DIR/ballroom.pkl"

echo "=== Encoding Covers80 dataset ==="
python evaluation/encode_folder.py \
    --audio-dir "$PROBES_ROOT/covers80" \
    --out "$PROBE_EMBEDDINGS_DIR/covers80.pkl"

# ---- Evaluate all heads on each probe ----
echo "=== Probe B: MUSDB18-HQ (timbral) ==="
python evaluation/probe_eval.py \
    --embeddings "$PROBE_EMBEDDINGS_DIR/musdb18hq.pkl" \
    --meta-format musdb18 \
    --heads-dir "$MODELS_DIR" \
    --out "$RESULTS_DIR/probe_B.json" \
    --exclude-classes mixture

echo "=== Probe B: Ballroom (rhythmic) ==="
python evaluation/probe_eval.py \
    --embeddings "$PROBE_EMBEDDINGS_DIR/ballroom.pkl" \
    --meta-format ballroom \
    --heads-dir "$MODELS_DIR" \
    --out "$RESULTS_DIR/probe_B_ballroom.json"

echo "=== Probe C: Covers80 (melodic/cover) ==="
python evaluation/probe_eval.py \
    --embeddings "$PROBE_EMBEDDINGS_DIR/covers80.pkl" \
    --meta-format covers80 \
    --heads-dir "$MODELS_DIR" \
    --out "$RESULTS_DIR/probe_C.json"

echo "Probe results written to $RESULTS_DIR/"
