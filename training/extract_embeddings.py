#!/usr/bin/env python3
"""
Extract audio embeddings from triplet directories and save to a .pkl cache.

This is the first step before training. Run once per factor; train_head.py
and train_factor_similarity.py can then iterate quickly on the cached embeddings.

Encoders:
  mert  — m-a-p/MERT-v1-330M, expects 24kHz mono audio.
            Hidden states from layers 3, 4, 5, 6, and 23 (penultimate) are each
            mean-pooled over time and concatenated → 5 × 1024 = 5120-dim output.
            Used for ALL three factors (melody, rhythm, timbre train splits).
  clap  — laion/clap-htsat-fused, 512-dim, expects 48kHz mono audio.
            Used only for CLAP baseline evaluation on test splits.
  clap3 — laion/larger_clap_music_and_speech, 512-dim, expects 48kHz mono audio.
            Used only for CLAP3 baseline evaluation on test splits.

Output .pkl structure:
  {
    "encoder": "mert" | "clap" | "clap3",
    "model_id": str,
    "embed_dim": int,          # 5120 for mert (5 layers × 1024), 512 for clap/clap3
    "embeddings": {str(wav_path): np.ndarray shape (D,)},
    "triplets": [{"anchor": str, "positives": [str,...], "negative": str}, ...]
    "split_partition": "train" | "test" | None,
  }

Usage:
  # Melody (MERT on full audio, melody triplets)
  python extract_embeddings.py --encoder mert \\
    --triplets-dir $TRIPLETS_ROOT/melody_triplets \\
    --out $EMBEDDINGS_ROOT/melody.pkl

  # Rhythm (MERT on full audio, rhythm-conditioned triplets)
  python extract_embeddings.py --encoder mert \\
    --triplets-dir $TRIPLETS_ROOT/rhythm_triplets \\
    --out $EMBEDDINGS_ROOT/rhythm.pkl

  # Timbre (CLAP on full audio)
  python extract_embeddings.py --encoder clap \\
    --triplets-dir $TRIPLETS_ROOT/timbre_triplets \\
    --out $EMBEDDINGS_ROOT/timbre.pkl

  # Run all three in parallel on different GPUs:
  CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py --encoder mert \\
    --triplets-dir $TRIPLETS_ROOT/melody_triplets \\
    --out $EMBEDDINGS_ROOT/melody.pkl &

  CUDA_VISIBLE_DEVICES=1 python extract_embeddings.py --encoder mert \\
    --triplets-dir $TRIPLETS_ROOT/rhythm_triplets \\
    --out $EMBEDDINGS_ROOT/rhythm.pkl &

  CUDA_VISIBLE_DEVICES=2 python extract_embeddings.py --encoder clap \\
    --triplets-dir $TRIPLETS_ROOT/timbre_triplets \\
    --out $EMBEDDINGS_ROOT/timbre.pkl &
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

MERT_SR = 24_000
CLAP_SR = 48_000
SEGMENT_SECS = 10

# MERT hidden-state layer indices to extract (0 = embeddings, 1-24 = transformer layers).
# Layer 23 is the penultimate transformer layer of MERT-v1-330M (which has 24 layers).
# These 5 layers are mean-pooled over time and concatenated → 5 × 1024 = 5120-dim.
# (MERT-v1-330M has hidden_size=1024; the 95M model has 768 — do not confuse them.)
MERT_EXTRACT_LAYERS = (3, 4, 5, 6, 23)


# ---------------------------------------------------------------------------
# Triplet discovery
# ---------------------------------------------------------------------------

def _load_split_set(split_file: str | None, partition: str | None) -> set[str] | None:
    """Return set of folder names for the requested partition, or None if no split specified."""
    if split_file is None:
        return None
    import json
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if partition not in ("train", "test"):
        raise SystemExit(f"--split-partition must be 'train' or 'test', got: {partition!r}")
    return set(data[partition])


def _collect_triplets(triplets_dir: Path, allowed_folders: set[str] | None = None) -> List[dict]:
    """Find all triplets_NNNN/triplet/ folders and return list of path dicts."""
    records = []
    for run_dir in sorted(triplets_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("triplets_"):
            continue
        if allowed_folders is not None and run_dir.name not in allowed_folders:
            continue
        td = run_dir / "triplet"
        if not td.exists():
            continue
        anchor = td / "anchor.wav"
        negative = td / "negative.wav"
        if not anchor.exists() or not negative.exists():
            continue
        positives = sorted(td.glob("positive_*.wav"))
        if not positives:
            continue
        records.append({
            "anchor": str(anchor),
            "positives": [str(p) for p in positives],
            "negative": str(negative),
        })
    return records


def _unique_paths(records: List[dict]) -> List[str]:
    seen = set()
    paths = []
    for r in records:
        for p in [r["anchor"]] + r["positives"] + [r["negative"]]:
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_mono(path: str, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    wav = wav.float().mean(dim=0)          # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    n = target_sr * SEGMENT_SECS
    if wav.shape[0] >= n:
        wav = wav[:n]
    else:
        wav = F.pad(wav, (0, n - wav.shape[0]))
    return wav.numpy()


# ---------------------------------------------------------------------------
# MERT encoder
# ---------------------------------------------------------------------------

def _extract_mert(
    paths: List[str],
    model_id: str,
    batch_size: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], int]:
    from transformers import AutoModel, AutoProcessor

    print(f"Loading MERT: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    hidden_size = model.config.hidden_size          # 1024 for MERT-v1-330M (not 768 — that is the 95M model)
    embed_dim = hidden_size * len(MERT_EXTRACT_LAYERS)  # 5120 = 5 × 1024

    result: Dict[str, np.ndarray] = {}
    batch_wavs: List[list] = []
    batch_keys: List[str] = []

    def _flush() -> None:
        if not batch_wavs:
            return
        inputs = processor(batch_wavs, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(
                input_values=inputs["input_values"].to(device),
                output_hidden_states=True,
            )
        # Mean-pool each target layer over the time dimension, then concatenate → [B, 5120]
        layer_embs = [
            out.hidden_states[i].mean(dim=1).cpu().float()  # [B, 768]
            for i in MERT_EXTRACT_LAYERS
        ]
        embs = torch.cat(layer_embs, dim=-1).numpy()  # [B, hidden_size * len(MERT_EXTRACT_LAYERS)]
        for k, e in zip(batch_keys, embs):
            result[k] = e
        batch_wavs.clear()
        batch_keys.clear()

    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        wav = _load_mono(path, MERT_SR)
        batch_wavs.append(wav.tolist())
        batch_keys.append(path)
        if len(batch_wavs) >= batch_size:
            _flush()
        if i % 50 == 0 or i == len(paths):
            pct = 100 * i / len(paths)
            print(f"  MERT {i}/{len(paths)} ({pct:.0f}%)  elapsed {time.perf_counter()-t0:.0f}s")
    _flush()
    return result, embed_dim


# ---------------------------------------------------------------------------
# CLAP encoder
# ---------------------------------------------------------------------------

def _extract_clap(
    paths: List[str],
    model_id: str,
    batch_size: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], int]:
    from transformers import ClapModel, ClapFeatureExtractor

    print(f"Loading CLAP: {model_id}")
    feature_extractor = ClapFeatureExtractor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device)
    model.eval()

    # CLAP audio projection output dim
    embed_dim = model.config.projection_dim  # 512 for clap-htsat-fused

    result: Dict[str, np.ndarray] = {}
    batch_wavs: List[np.ndarray] = []
    batch_keys: List[str] = []

    def _flush() -> None:
        if not batch_wavs:
            return
        inputs = feature_extractor(
            raw_speech=batch_wavs,
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(device)
        # is_longer may be absent or None depending on transformers version;
        # default to all-False (no clip exceeds the 10-second max duration)
        is_longer = inputs.get("is_longer")
        if is_longer is None:
            is_longer = torch.zeros(input_features.shape[0], dtype=torch.bool, device=device)
        else:
            is_longer = is_longer.to(device)
        with torch.no_grad():
            embs = model.get_audio_features(
                input_features=input_features,
                is_longer=is_longer,
            )
        embs = embs.cpu().float().numpy()  # [B, 512]
        for k, e in zip(batch_keys, embs):
            result[k] = e
        batch_wavs.clear()
        batch_keys.clear()

    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        wav = _load_mono(path, CLAP_SR)
        batch_wavs.append(wav)
        batch_keys.append(path)
        if len(batch_wavs) >= batch_size:
            _flush()
        if i % 50 == 0 or i == len(paths):
            pct = 100 * i / len(paths)
            print(f"  CLAP {i}/{len(paths)} ({pct:.0f}%)  elapsed {time.perf_counter()-t0:.0f}s")
    _flush()
    return result, embed_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract audio embeddings from triplets and cache to .pkl"
    )
    ap.add_argument("--encoder", choices=["mert", "clap", "clap3"], required=True,
                    help="Which encoder to use: mert (melody/rhythm), clap (timbre), clap3 (baseline)")
    ap.add_argument("--triplets-dir", required=True,
                    help="Root dir with triplets_NNNN/ sub-folders")
    ap.add_argument("--out", required=True, help="Output .pkl file path")
    ap.add_argument("--split-file", default=None,
                    help="Path to split JSON produced by create_split.py (e.g. splits/melody_split.json)")
    ap.add_argument("--split-partition", choices=["train", "test"], default=None,
                    help="Which partition to extract: 'train' or 'test'. Requires --split-file.")
    ap.add_argument("--model-id", default=None,
                    help="Override model ID (defaults: mert→m-a-p/MERT-v1-330M, clap→laion/clap-htsat-fused, clap3→laion/larger_clap_music_and_speech)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Audio clips per forward pass (reduce if OOM)")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Process only shard I of N (0-indexed, e.g. '0/3'). "
                         "Splits the unique wav list evenly. Used to parallelize "
                         "a single factor across multiple GPUs; merge with "
                         "merge_pkl.py afterwards.")
    args = ap.parse_args()

    if args.split_file and not args.split_partition:
        raise SystemExit("--split-file requires --split-partition (train or test)")
    if args.split_partition and not args.split_file:
        raise SystemExit("--split-partition requires --split-file")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    triplets_dir = Path(args.triplets_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_folders = _load_split_set(args.split_file, args.split_partition)
    if allowed_folders is not None:
        print(f"Using split partition '{args.split_partition}' ({len(allowed_folders)} folders) from {args.split_file}")

    print(f"Scanning {triplets_dir} ...")
    records = _collect_triplets(triplets_dir, allowed_folders)
    if not records:
        raise SystemExit(f"No triplets found in {triplets_dir}")
    print(f"Found {len(records)} triplets")

    paths = _unique_paths(records)

    # Optional sharding: split wav list across N workers
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        chunk = len(paths) // shard_n
        lo = shard_i * chunk
        hi = lo + chunk if shard_i < shard_n - 1 else len(paths)
        paths = paths[lo:hi]
        print(f"Shard {shard_i}/{shard_n}: processing wavs {lo}–{hi-1} ({len(paths)} files)")
    else:
        print(f"Unique wav files to embed: {len(paths)}")

    if args.encoder == "mert":
        model_id = args.model_id or "m-a-p/MERT-v1-330M"
        embeddings, embed_dim = _extract_mert(paths, model_id, args.batch_size, device)
    elif args.encoder == "clap3":
        model_id = args.model_id or "laion/larger_clap_music_and_speech"
        embeddings, embed_dim = _extract_clap(paths, model_id, args.batch_size, device)
    else:
        model_id = args.model_id or "laion/clap-htsat-fused"
        embeddings, embed_dim = _extract_clap(paths, model_id, args.batch_size, device)

    print(f"Embedded {len(embeddings)} files ({embed_dim}-dim)")

    data = {
        "encoder": args.encoder,
        "model_id": model_id,
        "embed_dim": embed_dim,
        "embeddings": embeddings,
        "split_partition": args.split_partition,
        # Only include full triplet metadata when not sharding; shards are
        # merged by merge_pkl.py which reconstructs this from the full scan.
        "triplets": records if not args.shard else [],
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
