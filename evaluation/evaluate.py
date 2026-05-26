#!/usr/bin/env python3
"""
Evaluate a trained projection head on a pre-extracted test embedding pkl.

Computes:
  - Triplet accuracy (fraction of triplets where d(a,p) < d(a,n) after projection)
  - Raw cosine baseline (same metric applied to the un-projected embeddings)
  - Per-batch breakdown for confidence checking

Usage:
  python evaluate.py \\
    --head    $MODELS_ROOT/head_rhy/best_head.pt \\
    --test    $EMBEDDINGS_ROOT/rhy_mert_test.pkl

  # Also compute raw cosine baseline only (no head):
  python evaluate.py --test $EMBEDDINGS_ROOT/rhy_mert_test.pkl

Output:
  Prints a table of metrics to stdout.
  If --out is given, writes a JSON results file.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Shared model definition (must match train_head.py)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletEmbeddingDataset(Dataset):
    def __init__(self, records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        a, p, n = self.records[idx]
        return (
            torch.from_numpy(a).float(),
            torch.from_numpy(p).float(),
            torch.from_numpy(n).float(),
        )


# ---------------------------------------------------------------------------
# Load test pkl → flat list of (anchor, pos, neg) numpy triplets
# ---------------------------------------------------------------------------

def load_test_records(pkl_path: Path) -> Tuple[List[Tuple], int, str, str]:
    """
    Load test pkl and expand to k² triplets (same logic as train_head.py).
    Returns flat list of (anchor_emb, pos_emb, neg_emb) tuples.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    embeddings: dict = data["embeddings"]
    triplets: list = data["triplets"]
    embed_dim: int = data["embed_dim"]
    encoder: str = data["encoder"]
    model_id: str = data["model_id"]

    records = []
    missing = 0
    for t in triplets:
        a_emb = embeddings.get(t["anchor"])
        n_emb = embeddings.get(t["negative"])
        if a_emb is None or n_emb is None:
            missing += 1
            continue

        pos_embs = [embeddings[p] for p in t["positives"] if p in embeddings]
        if not pos_embs:
            missing += 1
            continue

        # k original: (anchor, pos_i, neg)
        for p_emb in pos_embs:
            records.append((a_emb, p_emb, n_emb))

        # k(k-1) cross-positive: (pos_i, pos_j, neg) for i≠j
        for i, p_i in enumerate(pos_embs):
            for j, p_j in enumerate(pos_embs):
                if i != j:
                    records.append((p_i, p_j, n_emb))

    if missing:
        print(f"[WARN] {missing} triplets skipped (missing embeddings).")
    print(f"  Loaded {len(records)} k² triplets from {len(triplets)} folders")
    return records, embed_dim, encoder, model_id


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_with_head(
    model: ProjectionHead,
    records: List[Tuple],
    batch_size: int,
    device: str,
) -> dict:
    """Run the projection head over all triplets and return accuracy stats."""
    model.eval()
    ds = TripletEmbeddingDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    d_ap_all, d_an_all = [], []

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        e_a = model(a)
        e_p = model(p)
        e_n = model(n)
        d_ap = 1.0 - (e_a * e_p).sum(dim=-1)
        d_an = 1.0 - (e_a * e_n).sum(dim=-1)
        correct += (d_ap < d_an).sum().item()
        total += a.size(0)
        d_ap_all.append(d_ap.cpu())
        d_an_all.append(d_an.cpu())

    d_ap_all = torch.cat(d_ap_all)
    d_an_all = torch.cat(d_an_all)
    margin = (d_an_all - d_ap_all).mean().item()  # avg gap: positive if correct ordering

    return {
        "triplet_accuracy": correct / total if total > 0 else 0.0,
        "n_triplets": total,
        "mean_d_ap": d_ap_all.mean().item(),
        "mean_d_an": d_an_all.mean().item(),
        "mean_margin": margin,  # positive = model separates pos from neg on average
    }


@torch.no_grad()
def eval_raw_cosine(records: List[Tuple], batch_size: int, device: str) -> dict:
    """Triplet accuracy of raw (un-projected) L2-normalised embeddings."""
    ds = TripletEmbeddingDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    d_ap_all, d_an_all = [], []

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        a = F.normalize(a, dim=-1)
        p = F.normalize(p, dim=-1)
        n = F.normalize(n, dim=-1)
        d_ap = 1.0 - (a * p).sum(dim=-1)
        d_an = 1.0 - (a * n).sum(dim=-1)
        correct += (d_ap < d_an).sum().item()
        total += a.size(0)
        d_ap_all.append(d_ap.cpu())
        d_an_all.append(d_an.cpu())

    d_ap_all = torch.cat(d_ap_all)
    d_an_all = torch.cat(d_an_all)

    return {
        "triplet_accuracy": correct / total if total > 0 else 0.0,
        "n_triplets": total,
        "mean_d_ap": d_ap_all.mean().item(),
        "mean_d_an": d_an_all.mean().item(),
        "mean_margin": (d_an_all - d_ap_all).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(stats: dict) -> str:
    acc = stats["triplet_accuracy"]
    n   = stats["n_triplets"]
    margin = stats["mean_margin"]
    d_ap = stats["mean_d_ap"]
    d_an = stats["mean_d_an"]
    return (
        f"acc={acc:.4f}  margin={margin:+.4f}  "
        f"mean_d(a,p)={d_ap:.4f}  mean_d(a,n)={d_an:.4f}  "
        f"n={n:,}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a trained projection head on a test embedding pkl."
    )
    ap.add_argument("--head", default=None,
                    help="Path to best_head.pt checkpoint (omit to only run raw cosine baseline).")
    ap.add_argument("--test", required=True,
                    help=".pkl file produced by extract_embeddings.py (test partition).")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", default=None,
                    help="Optional path to write JSON results file.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load test data ---
    print(f"\nLoading test embeddings from {args.test} ...")
    records, embed_dim, encoder, model_id = load_test_records(Path(args.test))
    print(f"  Encoder: {encoder} ({model_id})  embed_dim={embed_dim}")

    results = {
        "test_pkl": args.test,
        "encoder": encoder,
        "model_id": model_id,
        "embed_dim": embed_dim,
    }

    # --- Raw cosine baseline ---
    print("\n[1] Raw cosine baseline (no head, L2-normalised embeddings):")
    raw_stats = eval_raw_cosine(records, args.batch_size, device)
    print(f"    {fmt(raw_stats)}")
    results["raw_cosine"] = raw_stats

    # --- Projection head ---
    if args.head is not None:
        ckpt = torch.load(args.head, map_location=device)
        in_dim     = ckpt["in_dim"]
        hidden_dim = ckpt["hidden_dim"]
        out_dim    = ckpt["out_dim"]
        saved_enc  = ckpt.get("encoder", "?")
        saved_ep   = ckpt.get("epoch", "?")
        saved_val  = ckpt.get("val_loss", float("nan"))
        saved_acc  = ckpt.get("val_acc",  float("nan"))

        print(f"\n[2] Projection head: {args.head}")
        print(f"    Architecture : Linear({in_dim}→{hidden_dim}) → ReLU → Linear({hidden_dim}→{out_dim}) → L2")
        print(f"    Saved at     : epoch {saved_ep}  val_loss={saved_val:.4f}  val_acc={saved_acc:.4f}")
        print(f"    Trained on   : encoder={saved_enc}")

        if in_dim != embed_dim:
            print(f"\n[ERROR] Head expects in_dim={in_dim} but test pkl has embed_dim={embed_dim}.")
            print("        Make sure the test pkl was extracted with the same encoder used for training.")
            raise SystemExit(1)

        model = ProjectionHead(in_dim, hidden_dim, out_dim).to(device)
        model.load_state_dict(ckpt["state_dict"])

        print("\n    Evaluating head on test set ...")
        head_stats = eval_with_head(model, records, args.batch_size, device)
        print(f"    {fmt(head_stats)}")
        results["head"] = {
            "checkpoint": args.head,
            "in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
            **head_stats,
        }

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"  Test set  : {Path(args.test).name}  ({records.__len__()} triplets)")
    print(f"  Raw cosine acc : {raw_stats['triplet_accuracy']:.4f}")
    if args.head:
        delta = head_stats["triplet_accuracy"] - raw_stats["triplet_accuracy"]
        print(f"  Head acc       : {head_stats['triplet_accuracy']:.4f}  (Δ {delta:+.4f} vs raw cosine)")
    print("="*70)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
