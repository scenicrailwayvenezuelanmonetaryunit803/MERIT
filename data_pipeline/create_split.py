#!/usr/bin/env python3
"""
Create deterministic 80/20 train/test splits at folder level for each factor.

Splits are stratified by folder name (sorted alphabetically) so the split is
reproducible regardless of filesystem order. The same folder never appears in
both train and test.

Usage:
  python create_split.py \
    --melody-dir  $TRIPLETS_ROOT/melody_triplets \
    --rhythm-dir  $TRIPLETS_ROOT/rhythm_triplets \
    --timbre-dir  $TRIPLETS_ROOT/timbre_triplets \
    --out-dir     splits/ \
    --test-frac   0.20 \
    --seed        42

Outputs:
  splits/melody_split.json
  splits/rhythm_split.json
  splits/timbre_split.json

Each file:
  {"train": ["triplets_1", "triplets_2", ...], "test": ["triplets_X", ...]}

Verify:
  python -c "
  import json
  for f in ['melody','rhythm','timbre']:
      d = json.load(open(f'splits/{f}_split.json'))
      print(f'{f}: {len(d[\"train\"])} train, {len(d[\"test\"])} test')
  "
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def make_split(triplets_dir: Path, test_frac: float, seed: int) -> dict:
    folders = sorted(
        d.name for d in triplets_dir.iterdir()
        if d.is_dir() and d.name.startswith("triplets_")
    )
    if not folders:
        raise SystemExit(f"No triplets_* folders found in {triplets_dir}")

    rng = random.Random(seed)
    shuffled = folders[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_frac))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])

    return {"train": train, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create 80/20 train/test splits for each factor.")
    ap.add_argument("--melody-dir", required=True, type=Path)
    ap.add_argument("--rhythm-dir", required=True, type=Path)
    ap.add_argument("--timbre-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    factors = {
        "melody": args.melody_dir,
        "rhythm": args.rhythm_dir,
        "timbre": args.timbre_dir,
    }

    for factor, triplets_dir in factors.items():
        if not triplets_dir.exists():
            raise SystemExit(f"Directory not found: {triplets_dir}")
        split = make_split(triplets_dir, args.test_frac, args.seed)
        out_path = args.out_dir / f"{factor}_split.json"
        out_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{factor}: {len(split['train'])} train folders, "
            f"{len(split['test'])} test folders  →  {out_path}"
        )


if __name__ == "__main__":
    main()
