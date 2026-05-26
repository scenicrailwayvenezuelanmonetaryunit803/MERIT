"""encode_folder.py — Encode all audio files in a folder with the MERT multi-layer
backbone and save {relative_path: np.ndarray(5120,)} to a pkl file.

Usage:
    python encode_folder.py --audio-dir /path/to/audio --out /path/to/out.pkl
    python encode_folder.py --audio-dir /path/to/audio --out out.pkl --no-recursive
    python encode_folder.py --audio-dir /path/to/audio --out out.pkl --batch-size 4

    # 10-second windowed analysis: produce one embedding per 10-s segment
    python encode_folder.py --audio-dir /path/to/audio --out out_seg.pkl --segment-sec 10

    # 10-second windowed analysis with mean-pooling into one embedding per file
    python encode_folder.py --audio-dir /path/to/audio --out out_seg.pkl \\
        --segment-sec 10 --segment-agg mean

Encoder: MERT-v1-330M, layers 3,4,5,6,23, mean-pool each → concatenate → 5120-dim.
Audio:   Resampled to 24 kHz mono.  Default (no --segment-sec): padded/truncated to 30 s.
         With --segment-sec N: split into N-second non-overlapping windows.

Output pkl format (no --segment-sec OR --segment-agg mean):
    dict[str, np.ndarray]  — keys are relative paths, values are (5120,) arrays.

Output pkl format (--segment-sec N without --segment-agg):
    dict[str, np.ndarray]  — keys are "rel_path:seg_00", values are (5120,) arrays.
    Useful for segment-level Probe C analysis (encode each 10-s window independently).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

EXTRACT_LAYERS = (3, 4, 5, 6, 23)
MODEL_ID = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24_000
MAX_SAMPLES = SAMPLE_RATE * 30   # 30 seconds
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def load_audio(path: str) -> torch.Tensor:
    """Load, resample to 24 kHz mono, pad/truncate to exactly 30 s."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0)   # mono — (N,)
    if wav.shape[0] > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    else:
        wav = torch.nn.functional.pad(wav, (0, MAX_SAMPLES - wav.shape[0]))
    return wav   # (720000,)


def load_audio_segments(path: str, segment_sec: int) -> list:
    """
    Load audio, resample to 24 kHz mono, then split into non-overlapping
    windows of `segment_sec` seconds.  Each window is padded to exactly
    `segment_sec` seconds if the last window is shorter.

    Returns:
        List of (torch.Tensor of shape (segment_sec * SAMPLE_RATE,), segment_index)
    """
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0)   # mono
    seg_samples = SAMPLE_RATE * segment_sec
    segments = []
    i = 0
    seg_idx = 0
    while i < wav.shape[0]:
        chunk = wav[i : i + seg_samples]
        if chunk.shape[0] < seg_samples:
            chunk = torch.nn.functional.pad(chunk, (0, seg_samples - chunk.shape[0]))
        segments.append((chunk, seg_idx))
        i += seg_samples
        seg_idx += 1
    return segments   # [(tensor(seg_samples,), idx), ...]


@torch.no_grad()
def encode_batch(
    wavs: list,
    processor: Wav2Vec2FeatureExtractor,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Run one batch through MERT and return (B, 5120) float32 numpy array."""
    inputs = processor(
        [w.numpy() for w in wavs],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    # Mean-pool each extracted layer over time dimension, then concatenate.
    parts = [out.hidden_states[layer].mean(dim=1) for layer in EXTRACT_LAYERS]
    emb = torch.cat(parts, dim=-1)   # (B, 5120)
    return emb.cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode an audio folder with MERT multi-layer features."
    )
    parser.add_argument("--audio-dir", required=True, help="Root folder to scan.")
    parser.add_argument("--out", required=True, help="Output pkl file path.")
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        default=True,
        help="If set, scan only the top-level directory (not subdirectories).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Audio files per forward pass."
    )
    parser.add_argument("--device", default="cuda", help="torch device string.")
    parser.add_argument(
        "--segment-sec",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, split each audio file into non-overlapping N-second windows "
            "before encoding.  Output keys become 'rel_path:seg_NN'.  "
            "Covers80 songs are ~4 min on average; --segment-sec 10 gives ~24 "
            "segments per song, enabling fine-grained temporal analysis."
        ),
    )
    parser.add_argument(
        "--segment-agg",
        choices=["none", "mean", "max"],
        default="none",
        help=(
            "How to aggregate per-segment embeddings back to a single file embedding. "
            "'none' (default): keep all segments as separate keys ('rel_path:seg_NN'). "
            "'mean': L2-normalised mean over segments — one key per file (same format "
            "as full-song encoding, but pooled from N-second windows). "
            "'max': element-wise max over segments then L2-normalise."
        ),
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        raise ValueError(f"--audio-dir does not exist: {audio_dir}")

    if args.recursive:
        paths = [p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    else:
        paths = [p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    paths = sorted(paths)
    print(f"Found {len(paths)} audio files under {audio_dir}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, output_hidden_states=True
    )
    model = model.to(args.device).eval()
    print(f"MERT loaded on {args.device}. Embed dim = {len(EXTRACT_LAYERS) * model.config.hidden_size}")

    results: dict = {}
    failed: list = []
    log_every = max(1, len(paths) // 20)   # log ~20 times

    if args.segment_sec:
        # ---- Segment mode: split each song into N-second windows ----
        seg_sec = args.segment_sec
        agg = args.segment_agg
        print(
            f"Segment mode: {seg_sec}s windows, aggregation='{agg}'.  "
            f"~{int(251 / seg_sec)} segments per song (Covers80 mean 251s)."
        )
        for file_idx, p in enumerate(paths):
            rel = str(p.relative_to(audio_dir))
            try:
                segs = load_audio_segments(str(p), seg_sec)
            except Exception as exc:
                print(f"  SKIP {p.name}: {exc}")
                failed.append(str(p))
                continue

            # Encode segments in mini-batches
            seg_embs = []
            for seg_start in range(0, len(segs), args.batch_size):
                batch_segs = segs[seg_start : seg_start + args.batch_size]
                wavs = [s[0] for s in batch_segs]
                embs = encode_batch(wavs, processor, model, args.device)
                seg_embs.extend(embs)

            if agg == "none":
                for (_, seg_idx), emb in zip(segs, seg_embs):
                    results[f"{rel}:seg_{seg_idx:03d}"] = emb
            else:
                mat = np.stack(seg_embs, axis=0)  # (n_segs, 5120)
                if agg == "mean":
                    vec = mat.mean(axis=0)
                else:  # max
                    vec = mat.max(axis=0)
                norm = np.linalg.norm(vec)
                results[rel] = vec / (norm + 1e-9)

            if file_idx % log_every == 0:
                print(f"  {file_idx + 1}/{len(paths)} files  ({len(segs)} segs last)")

    else:
        # ---- Full-song mode (original behaviour) ----
        log_every = max(1, (len(paths) // args.batch_size) // 20)
        for batch_idx, i in enumerate(range(0, len(paths), args.batch_size)):
            batch_paths = paths[i : i + args.batch_size]
            wavs, valid_paths = [], []
            for p in batch_paths:
                try:
                    wavs.append(load_audio(str(p)))
                    valid_paths.append(str(p.relative_to(audio_dir)))
                except Exception as exc:
                    print(f"  SKIP {p.name}: {exc}")
                    failed.append(str(p))

            if not wavs:
                continue

            embs = encode_batch(wavs, processor, model, args.device)
            for rel_path, emb in zip(valid_paths, embs):
                results[rel_path] = emb

            if batch_idx % log_every == 0:
                done = i + len(wavs)
                print(f"  {done}/{len(paths)} encoded ({100*done//len(paths)}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh, protocol=4)
    print(f"Saved {len(results)} embeddings → {out_path}")

    if failed:
        print(f"WARNING: {len(failed)} files failed to encode:")
        for p in failed[:10]:
            print(f"  {p}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
