#!/usr/bin/env python3
"""
Train a small linear projection head on pre-extracted embeddings.

This is the FAST path — the encoder never runs during training because
extract_embeddings.py already cached all embeddings. Each factor trains
in a few minutes on a single GPU (or even CPU).

Use this for rapid iteration. For paper-quality results, also run
train_factor_similarity.py which fine-tunes the encoder too.

The head architecture:
  Linear(D → hidden_dim) → ReLU → Linear(hidden_dim → out_dim, bias=False) → L2-normalize

With the MERT multi-layer backbone, D = 5 × 1024 = 5120 (layers 3, 4, 5, 6, 23 concatenated).
(MERT-v1-330M has hidden_size=1024; the 95M model has 768 — do not confuse them.)
The MLP learns to weight and combine information from different MERT layers, routing
each factor to the dimensions most diagnostic for that specific factor.

Similarity between two clips = cosine similarity of their projections.
Loss: max(0, margin + cosine_dist(anchor, pos) - cosine_dist(anchor, neg))

Usage:
  # From cached embeddings produced by extract_embeddings.py:

  python train_head.py \\
    --embeddings $EMBEDDINGS_ROOT/melody.pkl \\
    --out $MODELS_ROOT/head_mel

  python train_head.py \\
    --embeddings $EMBEDDINGS_ROOT/rhythm.pkl \\
    --out $MODELS_ROOT/head_rhy

  python train_head.py \\
    --embeddings $EMBEDDINGS_ROOT/timbre.pkl \\
    --out $MODELS_ROOT/head_tim

  # All three in parallel (each is CPU/single-GPU, very fast):
  python train_head.py --embeddings $EMBEDDINGS_ROOT/melody.pkl --out $MODELS_ROOT/head_mel &
  python train_head.py --embeddings $EMBEDDINGS_ROOT/rhythm.pkl --out $MODELS_ROOT/head_rhy &
  python train_head.py --embeddings $EMBEDDINGS_ROOT/timbre.pkl --out $MODELS_ROOT/head_tim &
  wait

Output (saved to --out dir):
  best_head.pt  — checkpoint with state_dict + metadata (encoder info, dims)
  history.json  — per-epoch train/val loss and triplet accuracy
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletEmbeddingDataset(Dataset):
    """Dataset that returns pre-extracted (anchor, positive, negative) embedding tuples."""

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


def _load_records(
    pkl_path: Path,
) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]], int, str, str]:
    """
    Load embeddings and expand to k² triplets, grouped by folder (anchor path).

    Returns a list of *groups*, where each group is a list of (anchor, pos, neg)
    tuples belonging to the same source folder. Callers must split at the group
    level to avoid data leakage (the same anchor embedding must never appear in
    both train and val).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    embeddings: dict = data["embeddings"]   # {str(path): np.array shape (D,)}
    triplets: list = data["triplets"]       # [{anchor, positives, negative}]
    embed_dim: int = data["embed_dim"]
    encoder: str = data["encoder"]
    model_id: str = data["model_id"]

    groups: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    missing = 0
    for t in triplets:
        a_emb = embeddings.get(t["anchor"])
        n_emb = embeddings.get(t["negative"])
        if a_emb is None or n_emb is None:
            missing += 1
            continue

        # Gather all valid positive embeddings for this folder.
        pos_embs = []
        for pos_path in t["positives"]:
            p_emb = embeddings.get(pos_path)
            if p_emb is not None:
                pos_embs.append(p_emb)

        if not pos_embs:
            missing += 1
            continue

        folder_records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # --- k² triplet expansion ---
        # Original k triplets: (anchor, pos_i, neg)
        for p_emb in pos_embs:
            folder_records.append((a_emb, p_emb, n_emb))

        # Cross-positive k*(k-1) triplets: (pos_i, pos_j, neg) for i≠j.
        # Valid because all positives in a folder share the same factor property
        # (same melody contour / same beat pattern / same instrument type),
        # so each positive is a valid anchor for any other positive.
        for i, p_emb_i in enumerate(pos_embs):
            for j, p_emb_j in enumerate(pos_embs):
                if i != j:
                    folder_records.append((p_emb_i, p_emb_j, n_emb))

        groups.append(folder_records)

    if missing:
        print(f"[WARN] {missing} triplets skipped (missing embeddings).")
    return groups, embed_dim, encoder, model_id


def _raw_cosine_accuracy(records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Triplet accuracy of raw cosine similarity (no head). Baseline before any training."""
    correct = 0
    for a, p, n in records:
        a_t = torch.from_numpy(a).float()
        p_t = torch.from_numpy(p).float()
        n_t = torch.from_numpy(n).float()
        # Normalize to unit vectors for cosine similarity
        a_t = F.normalize(a_t, dim=-1)
        p_t = F.normalize(p_t, dim=-1)
        n_t = F.normalize(n_t, dim=-1)
        d_ap = 1.0 - (a_t * p_t).sum()
        d_an = 1.0 - (a_t * n_t).sum()
        if d_ap < d_an:
            correct += 1
    return correct / len(records) if records else 0.0


# ---------------------------------------------------------------------------
# Model: single linear layer + L2-norm
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection + L2 normalisation.

    in_dim → hidden_dim (ReLU) → out_dim (L2-normalised unit vector).

    The shallow MLP allows the head to selectively combine information from different
    MERT layers (early layers capture rhythm/timbre, later layers capture melody/pitch).
        Each factor's head learns a different routing through the 5120-dim multi-layer input.
    """

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
# Loss
# ---------------------------------------------------------------------------

class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al., CVPR 2020) adapted for triplet inputs.

    For each (anchor, pos, neg) triple of L2-normalised unit vectors:
        sp = cosine_sim(anchor, pos)
        sn = cosine_sim(anchor, neg)
        αp = max(0, Op − sp)   ← softens gradient for already-close positives
        αn = max(0, sn − On)   ← softens gradient for already-far negatives
        L  = softplus(γ · [αn·(sn − On) − αp·(sp − Op)])
    where Op = 1 − m (target lower bound for positives)
          On = m     (target upper bound for negatives)

    Compared to triplet margin loss, the per-pair re-weighting prevents the
    common failure mode where loss hits zero on easy pairs while hard pairs
    still have wrong ordering — Circle Loss keeps gradients alive on those.
    Typical hyper-parameters from the paper: gamma=80, m=0.25.
    """

    def __init__(self, gamma: float = 80.0, m: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.Op = 1.0 - m   # push positive cosine sim toward 1
        self.On = m          # push negative cosine sim toward 0

    def forward(
        self,
        anchor: torch.Tensor,   # [B, D] unit vectors
        pos: torch.Tensor,      # [B, D] unit vectors
        neg: torch.Tensor,      # [B, D] unit vectors
    ) -> torch.Tensor:
        sp = (anchor * pos).sum(dim=-1)   # [B]
        sn = (anchor * neg).sum(dim=-1)   # [B]

        alpha_p = (self.Op - sp.detach()).clamp(min=0.0)
        alpha_n = (sn.detach() - self.On).clamp(min=0.0)

        loss = F.softplus(
            self.gamma * (alpha_n * (sn - self.On) - alpha_p * (sp - self.Op))
        )
        return loss.sum()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading embeddings from {args.embeddings} ...")
    groups, in_dim, encoder, model_id = _load_records(Path(args.embeddings))
    print(f"  Encoder: {encoder} ({model_id})  embed_dim={in_dim}")
    print(f"  Folders (anchor groups): {len(groups)}")
    total_triplets = sum(len(g) for g in groups)
    print(f"  Total k² triplets: {total_triplets}")

    # --- Folder-level train/val split ---
    # CRITICAL: split at the folder (group) level, never at the triplet level.
    # The same anchor embedding appears in all k² triplets of one folder.
    # If we split at the triplet level, the model sees the same anchor in both
    # train and val, making val accuracy meaningless (data leakage).
    random.shuffle(groups)
    n_val_groups = max(1, int(len(groups) * args.val_split))
    val_groups = groups[:n_val_groups]
    train_groups = groups[n_val_groups:]
    val_records = [r for g in val_groups for r in g]
    train_records = [r for g in train_groups for r in g]
    print(f"  Train: {len(train_groups)} folders, {len(train_records)} triplets")
    print(f"  Val:   {len(val_groups)} folders, {len(val_records)} triplets  (folder-level split, no leakage)")

    # --- Raw cosine baseline (no head, no training) ---
    # Sample up to 5000 val records for a fast estimate.
    baseline_sample = val_records[:5000]
    raw_acc = _raw_cosine_accuracy(baseline_sample)
    print(f"  Raw cosine baseline accuracy (val, no head): {raw_acc:.3f}")
    if raw_acc > 0.95:
        print("  [NOTE] Raw MERT cosine already scores >95% on this factor's val set.")
        print("         The head may converge very fast. Off-diagonal suppression on the")
        print("         TEST set (other factors) is what matters for the paper's claim.")

    train_ds = TripletEmbeddingDataset(train_records)
    val_ds = TripletEmbeddingDataset(val_records)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0,  # embeddings already in RAM, no I/O bottleneck
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ProjectionHead(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Head: Linear({in_dim}→{args.hidden_dim}) → ReLU → Linear({args.hidden_dim}→{args.out_dim})  params={n_params:,}")

    # Triplet margin loss (commented out — kept for reference)
    # loss_fn = nn.TripletMarginWithDistanceLoss(
    #     distance_function=lambda a, b: 1.0 - (a * b).sum(dim=-1),  # cosine distance
    #     margin=args.margin,
    #     reduction="sum",
    # )

    # Circle Loss: per-pair re-weighting based on current similarity.
    # Equivalent sigma/margin semantics to triplet loss but with smoother gradients.
    # gamma=80, m=0.25 are the recommended defaults from the original paper.
    loss_fn = CircleLoss(gamma=10.0, m=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_steps), eta_min=args.lr * 0.01
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}", end="")
    if args.patience > 0:
        print(f", early-stop patience={args.patience}", end="")
    print()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        tr_losses, tr_accs = [], []
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            e_a, e_p, e_n = model(a), model(p), model(n)
            loss = loss_fn(e_a, e_p, e_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            d_ap = 1.0 - (e_a * e_p).sum(dim=-1)
            d_an = 1.0 - (e_a * e_n).sum(dim=-1)
            tr_losses.append(loss.item())
            tr_accs.append((d_ap < d_an).float().mean().item())

        # --- Validate ---
        model.eval()
        vl_losses, vl_accs = [], []
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                e_a, e_p, e_n = model(a), model(p), model(n)
                loss = loss_fn(e_a, e_p, e_n)
                d_ap = 1.0 - (e_a * e_p).sum(dim=-1)
                d_an = 1.0 - (e_a * e_n).sum(dim=-1)
                vl_losses.append(loss.item())
                vl_accs.append((d_ap < d_an).float().mean().item())

        tr_l, tr_a = float(np.mean(tr_losses)), float(np.mean(tr_accs))
        vl_l, vl_a = float(np.mean(vl_losses)), float(np.mean(vl_accs))
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={tr_l:.4f} acc={tr_a:.3f}  "
            f"val loss={vl_l:.4f} acc={vl_a:.3f}  ({elapsed:.1f}s)"
        )

        rec = {
            "epoch": epoch,
            "train_loss": tr_l, "val_loss": vl_l,
            "train_acc": tr_a, "val_acc": vl_a,
        }
        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if vl_l < best_val_loss:
            best_val_loss = vl_l
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden_dim": args.hidden_dim,
                    "out_dim": args.out_dim,
                    "encoder": encoder,
                    "model_id": model_id,
                    "epoch": epoch,
                    "val_loss": vl_l,
                    "val_acc": vl_a,
                },
                out_dir / "best_head.pt",
            )
            print(f"  *** Saved best_head.pt (val_loss={vl_l:.4f}, val_acc={vl_a:.3f}) ***")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  Early stop: val_loss did not improve for {args.patience} epochs.")
                break

    print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {out_dir}/best_head.pt")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a linear projection head on pre-extracted embeddings (fast)."
    )
    ap.add_argument("--embeddings", required=True,
                    help=".pkl file produced by extract_embeddings.py")
    ap.add_argument("--out", required=True,
                    help="Output directory for checkpoints")
    ap.add_argument("--out-dim", type=int, default=128,
                    help="Projected embedding dimension (default 128)")
    ap.add_argument("--hidden-dim", type=int, default=512,
                    help="Hidden dim of the MLP head (default 512; input is 5120-dim for MERT-v1-330M multi-layer)")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Training epochs (fast since no audio I/O; default 200)")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="Batch size (can be large since inputs are just vectors)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.2,
                    help="Triplet cosine-distance margin")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=0,
                    help="Early stopping patience (epochs without val_loss improvement; 0=disabled)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
