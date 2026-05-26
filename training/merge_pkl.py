#!/usr/bin/env python3
"""
Merge multiple shard .pkl files (produced by extract_embeddings.py --shard I/N)
into a single .pkl that also contains the full triplet metadata.

Usage:
  python merge_pkl.py \
    --shards $EMBEDDINGS_ROOT/melody_shard_*.pkl \
    --triplets-dir $TRIPLETS_ROOT/melody_triplets \
    --out $EMBEDDINGS_ROOT/melody.pkl
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

# Allow running as `python training/merge_pkl.py` from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from extract_embeddings import _collect_triplets   # reuse scanner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Shard .pkl files (glob OK if quoted)")
    ap.add_argument("--triplets-dir", required=True,
                    help="Original triplets root dir (to rebuild triplet metadata)")
    ap.add_argument("--out", required=True, help="Output merged .pkl")
    args = ap.parse_args()

    # Expand globs
    shard_paths = []
    for pat in args.shards:
        expanded = glob.glob(pat)
        if not expanded:
            raise SystemExit(f"No files matched: {pat}")
        shard_paths += expanded
    shard_paths = sorted(shard_paths)
    print(f"Merging {len(shard_paths)} shards:")
    for p in shard_paths:
        print(f"  {p}")

    merged_embeddings: dict = {}
    encoder = model_id = embed_dim = None

    for p in shard_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if encoder is None:
            encoder, model_id, embed_dim = data["encoder"], data["model_id"], data["embed_dim"]
        else:
            assert data["encoder"] == encoder, "Shard encoder mismatch"
        merged_embeddings.update(data["embeddings"])

    print(f"Total unique embeddings: {len(merged_embeddings)}")

    # Rebuild full triplet metadata from the original directory
    records = _collect_triplets(Path(args.triplets_dir))
    print(f"Triplet records: {len(records)}")

    # Verify all wav paths have embeddings
    missing = []
    for r in records:
        for path in [r["anchor"]] + r["positives"] + [r["negative"]]:
            if path not in merged_embeddings:
                missing.append(path)
    if missing:
        print(f"WARNING: {len(missing)} wav paths have no embedding (first 5):")
        for m in missing[:5]:
            print(f"  {m}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "encoder": encoder,
            "model_id": model_id,
            "embed_dim": embed_dim,
            "embeddings": merged_embeddings,
            "triplets": records,
        }, f, protocol=4)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved merged pkl: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
