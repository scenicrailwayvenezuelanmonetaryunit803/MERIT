"""probe_eval.py — Evaluate trained MERIT heads on external probe datasets.

Supports four probe experiment types:

  --meta-format mtgjamendo
        TSV with columns: TRACKID <tab> RELPATH <tab> TAG1 <tab> TAG2 ...
        (format produced by MTG-Jamendo autotagging_instrument-test.tsv)
        Metric: per-tag mAP@10; aggregated mean mAP@10 across all tags.

  --meta-format ballroom
        Labels inferred from immediate parent directory name of each audio file.
        Works with the mirdata Ballroom layout:  <class>/<file>.mp3
        Metric: triplet accuracy per class; overall mAP@10.

  --meta-format musdb18
        Labels inferred from filename (vocals.wav → "vocals", drums.wav → "drums", etc.)
        Works with MUSDB18-HQ layout: test/<song>/vocals.wav, test/<song>/drums.wav, etc.
        Metric: triplet accuracy per stem type (vocals/drums/bass/other); overall accuracy.

  --meta-format covers80
        Each subdirectory of the root folder contains exactly 2 audio files
        forming a cover pair.  No extra metadata file needed.
        Metric: triplet accuracy (original vs cover vs random-other-pair).
        Also writes per-pair factor profile to --out JSON and --visualize PDF.

Usage:
  python probe_eval.py \
    --embeddings $PROBES_ROOT/ballroom_mert.pkl \
    --meta-format ballroom \
    --heads-dir $MODELS_ROOT \
    --out results/probe_B.json

  python probe_eval.py \
    --embeddings $PROBES_ROOT/musdb18hq_test_mert.pkl \
    --meta-format musdb18 \
    --heads-dir $MODELS_ROOT \
    --out results/probe_A.json

  python probe_eval.py \
    --embeddings $PROBES_ROOT/covers80_mert.pkl \
    --meta-format covers80 \
    --heads-dir $MODELS_ROOT \
    --out results/probe_C.json \
    --visualize results/probe_C_profile.pdf

  python probe_eval.py \
    --embeddings $PROBES_ROOT/mtgjamendo_mert.pkl \
    --metadata $PROBES_ROOT/mtgjamendo_repo/data/splits/split-0/autotagging_instrument-test.tsv \
    --meta-format mtgjamendo \
    --heads-dir $MODELS_ROOT \
    --out results/probe_A.json

Output JSON:
  {
    "raw_mert": {"metric": value, ...},
    "H_mel":    {"metric": value, ...},
    "H_rhy":    {"metric": value, ...},
    "H_tim":    {"metric": value, ...}
  }
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model (must match train_head.py exactly)
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
# Load embeddings pkl (output of encode_folder.py)
# ---------------------------------------------------------------------------


def load_embeddings(pkl_path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """Return (embeddings dict, embed_dim)."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        # Direct {rel_path: ndarray} dict — produced by encode_folder.py
        embeddings = data
        embed_dim = next(iter(embeddings.values())).shape[0]
    else:
        raise ValueError(f"Unexpected pkl format in {pkl_path}")
    return embeddings, embed_dim


# ---------------------------------------------------------------------------
# Head loading
# ---------------------------------------------------------------------------


def load_heads(
    heads_dir: Path, embed_dim: int, device: str
) -> Dict[str, Optional[ProjectionHead]]:
    """Load best_head.pt for H_mel, H_rhy, H_tim from heads_dir."""
    heads: Dict[str, Optional[ProjectionHead]] = {}
    for name in ("head_mel", "head_rhy", "head_tim"):
        ckpt = heads_dir / name / "best_head.pt"
        if not ckpt.exists():
            print(f"[WARN] Head not found: {ckpt} — skipping {name}")
            heads[name] = None
            continue
        ckpt_data = torch.load(ckpt, map_location="cpu")
        # Checkpoint is a metadata dict with a nested "state_dict" key
        state = ckpt_data["state_dict"] if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data else ckpt_data

        # Detect architecture from state dict keys
        if "net.0.weight" in state:
            # Current two-layer MLP: Linear(in_dim→512)→ReLU→Linear(512→128)
            saved_in_dim = state["net.0.weight"].shape[1]
            if saved_in_dim != embed_dim:
                print(f"  [SKIP] {name}: checkpoint in_dim={saved_in_dim} != embeddings embed_dim={embed_dim}. Re-train this head.")
                heads[name] = None
                continue
            head = ProjectionHead(in_dim=embed_dim)
        elif "linear.weight" in state:
            # Old single-layer head — architecture mismatch, cannot use
            saved_in_dim = state["linear.weight"].shape[1]
            print(f"  [SKIP] {name}: old single-layer checkpoint (in_dim={saved_in_dim}). Re-train with train_head.py.")
            heads[name] = None
            continue
        else:
            print(f"  [SKIP] {name}: unrecognised state_dict keys: {list(state.keys())}")
            heads[name] = None
            continue

        head.load_state_dict(state)
        head = head.to(device).eval()
        heads[name] = head
        print(f"  Loaded {name} from {ckpt}")
    return heads


# ---------------------------------------------------------------------------
# Projection utility
# ---------------------------------------------------------------------------


@torch.no_grad()
def project_all(
    embeddings: Dict[str, np.ndarray],
    heads: Dict[str, Optional[ProjectionHead]],
    device: str,
    batch_size: int = 512,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Project all embeddings through each head (and also L2-normalise raw for baseline).
    Also adds a "combined" key = L2-normalised mean of all available head projections.

    Returns:
        projected["raw"]           -> {rel_path: unit_vec (5120,) float32}
        projected["head_mel"]      -> {rel_path: unit_vec (128,) float32}
        projected["head_rhy"]      -> {rel_path: unit_vec (128,) float32}
        projected["head_tim"]      -> {rel_path: unit_vec (128,) float32}
        projected["combined"]      -> {rel_path: unit_vec (128,) float32}  mean of available heads
    """
    keys = list(embeddings.keys())
    raw_matrix = np.stack([embeddings[k] for k in keys], axis=0)  # (N, D)

    # Raw L2-normalised baseline
    raw_norm = raw_matrix / (np.linalg.norm(raw_matrix, axis=1, keepdims=True) + 1e-9)
    projected: Dict[str, Dict[str, np.ndarray]] = {
        "raw": {k: raw_norm[i] for i, k in enumerate(keys)}
    }

    for head_name, head in heads.items():
        if head is None:
            projected[head_name] = {}
            continue
        out_vecs = []
        for start in range(0, len(keys), batch_size):
            batch_np = raw_matrix[start : start + batch_size]
            batch_t = torch.from_numpy(batch_np).float().to(device)
            out_vecs.append(head(batch_t).cpu().numpy())
        out_matrix = np.concatenate(out_vecs, axis=0)  # (N, 128)
        projected[head_name] = {k: out_matrix[i] for i, k in enumerate(keys)}

    # ---- Combination strategies (ablation over four fusion methods) ----
    active_names = [h for h in ("head_mel", "head_rhy", "head_tim") if projected.get(h)]
    active = [projected[h] for h in active_names]

    if len(active) >= 2:
        # --- Strategy 1: Arithmetic mean → L2-norm (default "combined") ---
        combined: Dict[str, np.ndarray] = {}
        for k in keys:
            vecs = [d[k] for d in active if k in d]
            if vecs:
                avg = np.mean(np.stack(vecs, axis=0), axis=0)
                combined[k] = avg / (np.linalg.norm(avg) + 1e-9)
        projected["combined"] = combined

        # --- Strategy 2: Concatenation → L2-norm  (dim = n_active × 128) ---
        concat_dict: Dict[str, np.ndarray] = {}
        for k in keys:
            vecs = [d[k] for d in active if k in d]
            if vecs:
                cat = np.concatenate(vecs, axis=0)
                concat_dict[k] = cat / (np.linalg.norm(cat) + 1e-9)
        projected["combined_concat"] = concat_dict

        # --- Strategy 3: Weighted mean (cover-tuned: H_mel=0.40, H_rhy=0.40, H_tim=0.20) ---
        # Weights reflect that cover songs share melody+rhythm; timbre changes more.
        # Normalised so available heads always sum to 1.
        _COVER_W = {"head_mel": 0.40, "head_rhy": 0.40, "head_tim": 0.20}
        w_raw = {h: _COVER_W.get(h, 1.0 / 3) for h in active_names}
        w_total = sum(w_raw.values())
        w_norm = {h: v / w_total for h, v in w_raw.items()}
        ref_dim = next(iter(active[0].values())).shape[0]  # 128
        wmean_dict: Dict[str, np.ndarray] = {}
        for k in keys:
            wvec = np.zeros(ref_dim, dtype=np.float32)
            for h, weight in w_norm.items():
                if k in projected[h]:
                    wvec = wvec + weight * projected[h][k]
            nrm = np.linalg.norm(wvec)
            if nrm > 1e-9:
                wmean_dict[k] = wvec / nrm
        projected["combined_wmean"] = wmean_dict

        # --- Strategy 4: Product of Experts — element-wise product → L2-norm ---
        # cos(a_prod, b_prod) ≈ f(Π cos(a_i, b_i)): only high if ALL heads agree.
        # Recent adoption: ByteCover3 (ICASSP 2023) uses PoE-style multi-space fusion.
        prod_dict: Dict[str, np.ndarray] = {}
        for k in keys:
            vecs = [d[k] for d in active if k in d]
            if vecs:
                prod = vecs[0].copy()
                for v in vecs[1:]:
                    prod = prod * v
                nrm = np.linalg.norm(prod)
                if nrm > 1e-9:
                    prod_dict[k] = prod / nrm
        projected["combined_prod"] = prod_dict

        # --- Strategy 5: Element-wise max → L2-norm (max pooling fusion) ---
        # Takes strongest activation from any head per dimension. Common in multi-modal fusion.
        max_dict: Dict[str, np.ndarray] = {}
        for k in keys:
            vecs = [d[k] for d in active if k in d]
            if vecs:
                max_vec = np.maximum.reduce(np.stack(vecs, axis=0))
                nrm = np.linalg.norm(max_vec)
                if nrm > 1e-9:
                    max_dict[k] = max_vec / nrm
        projected["combined_max"] = max_dict

        print(
            f"  Combinations built from {len(active_names)} heads "
            f"({len(combined)} clips): "
            f"mean | concat({ref_dim * len(active_names)}-d) | wmean | prod-exp | max-pool"
        )
    else:
        projected["combined"] = {}
        projected["combined_concat"] = {}
        projected["combined_wmean"] = {}
        projected["combined_prod"] = {}
        projected["combined_max"] = {}

    return projected


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _ap_at_k(query_vec: np.ndarray, gallery_vecs: list, gallery_labels: list,
             query_labels: set, k: int = 10) -> float:
    """Average precision@k for a single query."""
    sims = [cosine_sim(query_vec, g) for g in gallery_vecs]
    ranked = sorted(zip(sims, gallery_labels), reverse=True)[:k]
    hits, ap, num_rel = 0, 0.0, 0
    for rank, (_, lbl) in enumerate(ranked, 1):
        if lbl in query_labels:
            hits += 1
            ap += hits / rank
            num_rel += 1
    if num_rel == 0:
        return 0.0
    return ap / min(k, sum(1 for l in gallery_labels if l in query_labels) + 1e-9)


def triplet_accuracy(
    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> float:
    """Fraction of triplets where cosine(anchor, pos) > cosine(anchor, neg)."""
    if not triplets:
        return float("nan")
    correct = sum(
        1 for a, p, n in triplets
        if cosine_sim(a, p) > cosine_sim(a, n)
    )
    return correct / len(triplets)


# ---------------------------------------------------------------------------
# Metadata loaders
# ---------------------------------------------------------------------------


def load_meta_mtgjamendo(tsv_path: Path) -> Dict[str, set]:
    """Parse autotagging_instrument TSV → {track_id: set_of_tags}."""
    labels: Dict[str, set] = {}
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 1:
                continue
            track_id = parts[0].strip()
            tags = {t.strip() for t in parts[1:] if t.strip()}
            labels[track_id] = tags
    return labels


def _ballroom_class_from_path(rel_path: str) -> str:
    """Infer Ballroom dance class from immediate parent directory name."""
    return Path(rel_path).parent.name


def _musdb18_stem_from_path(rel_path: str) -> str:
    """Infer MUSDB18 stem type from filename (vocals.wav → vocals)."""
    return Path(rel_path).stem


def _covers80_pairs(embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
    """
    Parse Covers80 structure: each immediate subdirectory holds exactly 2 files.
    Return list of (file_a_rel, file_b_rel) pairs.
    """
    dir_to_files: Dict[str, List[str]] = defaultdict(list)
    for rel in embeddings:
        parent = Path(rel).parent.name
        if parent and parent != ".":
            dir_to_files[parent].append(rel)
        else:
            # Flat layout fallback: can't form pairs
            pass
    pairs = []
    for parent, files in dir_to_files.items():
        if len(files) == 2:
            pairs.append((sorted(files)[0], sorted(files)[1]))
        elif len(files) > 2:
            # Take first two alphabetically
            sf = sorted(files)
            pairs.append((sf[0], sf[1]))
    return pairs


# ---------------------------------------------------------------------------
# Probe A — MTG-Jamendo instrument retrieval
# ---------------------------------------------------------------------------


def run_probe_A(
    projected: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, np.ndarray],
    metadata_path: Path,
    k: int = 10,
) -> Dict:
    """Compute per-tag and mean mAP@10 for each projection."""
    print("\n=== Probe A: Instrument Retrieval (MTG-Jamendo) ===")
    # Build track_id→labels map
    track_labels = load_meta_mtgjamendo(metadata_path)

    # Match rel-path keys in embeddings to track IDs.
    # MTG-Jamendo paths look like: "00/000002.mp3" — track_id = "000002"
    key_to_trackid: Dict[str, str] = {}
    for rel in embeddings:
        stem = Path(rel).stem   # e.g. "000002"
        tid = stem.lstrip("0") or "0"   # strip leading zeros for lookup
        # Try both zero-padded and stripped
        if stem in track_labels:
            key_to_trackid[rel] = stem
        elif tid in track_labels:
            key_to_trackid[rel] = tid

    matched = list(key_to_trackid.keys())
    if not matched:
        return {"error": "No embedding keys matched MTG-Jamendo track IDs. Check path format."}
    print(f"  Matched {len(matched)}/{len(embeddings)} files to MTG-Jamendo metadata.")

    results: Dict = {}
    for proj_name, proj_dict in projected.items():
        if not proj_dict:
            results[proj_name] = {"mean_map10": None}
            continue

        # Collect all tags that have at least 2 examples
        tag_counter: Dict[str, int] = defaultdict(int)
        for rel in matched:
            tid = key_to_trackid[rel]
            for tag in track_labels.get(tid, set()):
                tag_counter[tag] += 1
        valid_tags = [t for t, c in tag_counter.items() if c >= 2]

        per_tag_ap: Dict[str, float] = {}
        for tag in valid_tags:
            query_keys = [r for r in matched if tag in track_labels.get(key_to_trackid[r], set())]
            gallery_keys = matched

            aps = []
            for qk in query_keys:
                q_vec = proj_dict.get(qk)
                if q_vec is None:
                    continue
                gallery_vecs = [proj_dict[g] for g in gallery_keys if g != qk and g in proj_dict]
                gallery_labels_list = [
                    track_labels.get(key_to_trackid[g], set()) for g in gallery_keys if g != qk and g in proj_dict
                ]
                # Convert multi-label to "relevant if shares ANY query tag"
                q_tags = track_labels.get(key_to_trackid[qk], set())
                gl_binary = [1 if len(ql & q_tags) > 0 else 0 for ql in gallery_labels_list]
                # mAP@k (single label: the tag we're querying by)
                sims = [cosine_sim(q_vec, gv) for gv in gallery_vecs]
                ranked_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
                hits, ap, n_rel = 0, 0.0, sum(gl_binary)
                for r, idx in enumerate(ranked_idx, 1):
                    if gl_binary[idx]:
                        hits += 1
                        ap += hits / r
                aps.append(ap / min(k, max(n_rel, 1)))
            per_tag_ap[tag] = float(np.mean(aps)) if aps else 0.0

        mean_map = float(np.mean(list(per_tag_ap.values()))) if per_tag_ap else 0.0
        results[proj_name] = {
            "mean_map10": round(mean_map, 4),
            "per_tag_map10": {t: round(v, 4) for t, v in sorted(per_tag_ap.items(), key=lambda x: -x[1])},
        }
        print(f"  {proj_name:12s}  mAP@10 = {mean_map:.4f}")

    return results


# ---------------------------------------------------------------------------
# Probe B — Ballroom groove retrieval
# ---------------------------------------------------------------------------


def run_probe_B(
    projected: Dict[str, Dict[str, np.ndarray]],
    n_neg_per_query: int = 10,
    k: int = 10,
    seed: int = 42,
    label_extractor=None,
    exclude_classes: List[str] = None,
) -> Dict:
    """Compute per-class and overall triplet accuracy + mAP@10 for each projection.
    
    Args:
        exclude_classes: List of class names to exclude (e.g., ['mixture'] for MUSDB18).
    """
    print("\n=== Probe B: Groove Class Retrieval (Ballroom/IRMAS/MUSDB18) ===")
    rng = random.Random(seed)
    if exclude_classes is None:
        exclude_classes = []

    # Build class→[rel_paths] mapping from all projected keys
    if label_extractor is None:
        label_extractor = _ballroom_class_from_path
    all_keys = list(next(iter(projected.values())).keys())
    class_to_keys: Dict[str, List[str]] = defaultdict(list)
    for rel in all_keys:
        cls = label_extractor(rel)
        if cls and cls not in exclude_classes:
            class_to_keys[cls].append(rel)

    print(f"  Classes found: {sorted(class_to_keys.keys())}")
    if exclude_classes:
        print(f"  Excluded classes: {sorted(exclude_classes)}")

    results: Dict = {}
    for proj_name, proj_dict in projected.items():
        if not proj_dict:
            results[proj_name] = {"triplet_accuracy": None}
            continue

        all_triplets: List[Tuple] = []
        per_class_acc: Dict[str, float] = {}

        for cls, class_keys in class_to_keys.items():
            if len(class_keys) < 2:
                continue
            other_keys = [k for c, ks in class_to_keys.items() for k in ks if c != cls]
            if not other_keys:
                continue
            cls_triplets: List[Tuple] = []
            for qk in class_keys:
                qv = proj_dict.get(qk)
                if qv is None:
                    continue
                # Positive: another key from the same class
                pos_keys = [k for k in class_keys if k != qk]
                # Negative: sampled from other classes
                neg_keys = rng.sample(other_keys, min(n_neg_per_query, len(other_keys)))
                for pk in pos_keys[:n_neg_per_query]:
                    pv = proj_dict.get(pk)
                    if pv is None:
                        continue
                    for nk in neg_keys:
                        nv = proj_dict.get(nk)
                        if nv is None:
                            continue
                        cls_triplets.append((qv, pv, nv))
            per_class_acc[cls] = round(triplet_accuracy(cls_triplets), 4)
            all_triplets.extend(cls_triplets)

        overall_acc = triplet_accuracy(all_triplets)
        results[proj_name] = {
            "triplet_accuracy": round(overall_acc, 4),
            "n_triplets": len(all_triplets),
            "per_class_accuracy": per_class_acc,
        }
        print(
            f"  {proj_name:12s}  triplet_acc = {overall_acc:.4f}"
            f"  (n={len(all_triplets)})"
        )
        # Print per-class breakdown
        for cls in sorted(per_class_acc.keys()):
            print(f"    {cls:15s} = {per_class_acc[cls]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Probe C — Covers80 cover song factor profiling
# ---------------------------------------------------------------------------


def run_probe_C(
    projected: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, np.ndarray],
    n_neg: int = 50,
    seed: int = 42,
    n_profile_pairs: int = 6,
) -> Dict:
    """
    Probe C — Covers80 cover song factor profiling.

    Experiment C1: Triplet accuracy per head (+ combined).
    Experiment C2: Per-pair factor profile {S_mel, S_rhy, S_tim} for selected pairs.

    Cover songs are a *multi-factor* signal: they share harmonic/melodic content but
    differ in timbre and sometimes rhythm.  The combined head should outperform any
    single factor head — and both should outperform raw MERT — demonstrating that
    MERIT's disentangled factors are complementary and jointly useful.
    """
    print("\n=== Probe C: Cover Song Factor Profiling (Covers80) ===")
    rng = random.Random(seed)

    pairs = _covers80_pairs(embeddings)
    if not pairs:
        return {"error": "No cover pairs detected. Check audio folder structure."}
    print(f"  Cover pairs found: {len(pairs)}")

    all_keys = list(embeddings.keys())

    # Ordered list of projections to evaluate (individual heads + 5 combination ablations)
    eval_order = [
        "raw", "head_mel", "head_rhy", "head_tim",
        "combined", "combined_wmean", "combined_concat", "combined_prod", "combined_max",
    ]

    results: Dict = {}
    # pair_profiles: list of dicts, one per cover pair, filled across all projections
    pair_profiles: List[Dict] = [{"pair": (fa, fb)} for fa, fb in pairs]

    for proj_name in eval_order:
        proj_dict = projected.get(proj_name, {})
        if not proj_dict:
            results[proj_name] = {"triplet_accuracy": None, "n_triplets": 0}
            continue

        triplets: List[Tuple] = []

        for idx, (fa, fb) in enumerate(pairs):
            va = proj_dict.get(fa)
            vb = proj_dict.get(fb)
            if va is None or vb is None:
                continue

            # Per-pair similarity (for factor profile chart)
            pair_profiles[idx][proj_name] = round(cosine_sim(va, vb), 4)

            # C1 triplets: (a, b, neg) and symmetric (b, a, neg)
            non_pair_keys = [k for k in all_keys if k != fa and k != fb]
            neg_sample = rng.sample(non_pair_keys, min(n_neg, len(non_pair_keys)))
            for nk in neg_sample:
                nv = proj_dict.get(nk)
                if nv is not None:
                    triplets.append((va, vb, nv))
                    triplets.append((vb, va, nv))

        acc = triplet_accuracy(triplets)
        results[proj_name] = {
            "triplet_accuracy": round(acc, 4),
            "n_triplets": len(triplets),
        }
        print(f"  {proj_name:18s}  triplet_acc = {acc:.4f}  (n={len(triplets)})")

    # Print ablation summary: best combination strategy
    comb_keys = ["combined", "combined_wmean", "combined_concat", "combined_prod", "combined_max"]
    best_comb = max(
        comb_keys,
        key=lambda k: results.get(k, {}).get("triplet_accuracy") or 0.0,
    )
    best_acc = results.get(best_comb, {}).get("triplet_accuracy", 0.0)
    print(f"\n  ★ Best combination: {best_comb}  ({best_acc:.4f})")

    # Select top-N profile pairs by highest combined similarity
    # (most clearly recognised cover pairs = most visually compelling)
    scoreable = [p for p in pair_profiles if "combined" in p or "head_mel" in p]
    top_pairs = sorted(
        scoreable,
        key=lambda p: p.get("combined", p.get("head_mel", 0)),
        reverse=True,
    )[:n_profile_pairs]

    results["cover_factor_profiles"] = top_pairs
    print(f"\n  Top {len(top_pairs)} profile pairs (by combined similarity):")
    for p in top_pairs:
        s_mel = p.get("head_mel", "—")
        s_rhy = p.get("head_rhy", "—")
        s_tim = p.get("head_tim", "—")
        s_com = p.get("combined", "—")
        print(f"    {Path(p['pair'][0]).parent.name:30s}  "
              f"S_mel={s_mel}  S_rhy={s_rhy}  S_tim={s_tim}  combined={s_com}")

    return results


# ---------------------------------------------------------------------------
# Visualisation (Probe C — two-panel figure)
# ---------------------------------------------------------------------------

# Display labels / colours for each projection key
_PROJ_LABEL = {
    "raw":              "Raw\nMERT",
    "head_mel":         "H_mel",
    "head_rhy":         "H_rhy",
    "head_tim":         "H_tim",
    "combined":         "MERIT\nmean",
    "combined_wmean":   "MERIT\nwt-mean",
    "combined_concat":  "MERIT\nconcat",
    "combined_prod":    "MERIT\nprod-exp",
    "combined_max":     "MERIT\nmax-pool",
}
_PROJ_COLOR = {
    "raw":              "#AAAAAA",
    "head_mel":         "#4878CF",
    "head_rhy":         "#E87B31",
    "head_tim":         "#6ACC65",
    "combined":         "#C44E52",
    "combined_wmean":   "#9467BD",
    "combined_concat":  "#8C564B",
    "combined_prod":    "#D62728",
    "combined_max":     "#E377C2",  # pink
}


def plot_probe_C(results: Dict, out_pdf: Path) -> None:
    """
    Two-panel figure:
      Panel A (top): triplet accuracy bar chart — raw / H_mel / H_rhy / H_tim / combined
      Panel B (bottom): factor profile grid for selected cover pairs
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not installed — skipping figure generation.")
        return

    profiles = results.get("cover_factor_profiles", [])
    n_profiles = len(profiles)
    n_prof_cols = 2
    n_prof_rows = (n_profiles + n_prof_cols - 1) // n_prof_cols

    # Figure layout: panel A (accuracy bar chart) + panel B (profile grid)
    fig = plt.figure(figsize=(12, 4 + n_prof_rows * 2.0))
    gs = gridspec.GridSpec(
        1 + n_prof_rows, n_prof_cols,
        height_ratios=[2.5] + [1.8] * n_prof_rows,
        hspace=0.55, wspace=0.35,
    )

    # ---- Panel A: triplet accuracy comparison bar chart ----
    ax_bar = fig.add_subplot(gs[0, :])   # spans both columns

    order = ["raw", "head_mel", "head_rhy", "head_tim", "combined"]
    accs = [results.get(k, {}).get("triplet_accuracy") for k in order]
    labels = [_PROJ_LABEL[k] for k in order]
    colors = [_PROJ_COLOR[k] for k in order]

    x = list(range(len(order)))
    bars = ax_bar.bar(x, [a if a is not None else 0 for a in accs],
                      color=colors, width=0.55, zorder=3)
    ax_bar.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="chance (0.50)", zorder=2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylim(0.45, 1.0)
    ax_bar.set_ylabel("Triplet Accuracy", fontsize=9)
    ax_bar.set_title("Probe C — Cover Song Detection (Covers80, 80 pairs)\n"
                     "Cover songs share harmonic/melodic content but differ in timbre and arrangement",
                     fontsize=9, loc="left")
    ax_bar.legend(fontsize=8)
    ax_bar.yaxis.grid(True, linewidth=0.4, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    for bar, acc in zip(bars, accs):
        if acc is not None:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{acc:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ---- Panel B: per-pair factor profile grid ----
    factor_keys = ["head_mel", "head_rhy", "head_tim"]
    factor_labels = ["S_mel", "S_rhy", "S_tim"]
    factor_colors = [_PROJ_COLOR[k] for k in factor_keys]

    for idx, profile in enumerate(profiles):
        row = 1 + idx // n_prof_cols
        col = idx % n_prof_cols
        ax = fig.add_subplot(gs[row, col])

        pair_name = Path(profile["pair"][0]).parent.name
        values = [profile.get(k, 0.0) for k in factor_keys]

        ax.barh(factor_labels[::-1], values[::-1],
                color=factor_colors[::-1], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_title(pair_name, fontsize=7.5, loc="left", pad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=7)
        for bar, val in zip(ax.patches, values[::-1]):
            ax.text(min(val + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7)

    # Hide empty subplot cells
    for idx in range(n_profiles, n_prof_rows * n_prof_cols):
        row = 1 + idx // n_prof_cols
        col = idx % n_prof_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out_pdf}")


# Legacy alias kept for backward compatibility
def plot_cover_profiles(profiles: List[Dict], out_pdf: Path) -> None:
    plot_probe_C({"cover_factor_profiles": profiles}, out_pdf)


# ---------------------------------------------------------------------------
# Probe C — Ablation figure (all 8 projections compared)
# ---------------------------------------------------------------------------


def plot_ablation_C(results: Dict, out_pdf: Path) -> None:
    """
    Grouped bar chart for Probe C ablation:
      Group 1 (left):  individual heads (raw, H_mel, H_rhy, H_tim)
      Group 2 (right): combination strategies (mean, wt-mean, concat, prod-exp)

    Best for an ISMIR paper appendix or dedicated ablation section.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping ablation figure.")
        return

    group1 = ["raw", "head_mel", "head_rhy", "head_tim"]
    group2 = ["combined", "combined_wmean", "combined_concat", "combined_prod", "combined_max"]
    all_keys = group1 + group2
    labels = [_PROJ_LABEL[k] for k in all_keys]
    colors = [_PROJ_COLOR[k] for k in all_keys]
    accs = [results.get(k, {}).get("triplet_accuracy") for k in all_keys]

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(all_keys))
    bars = ax.bar(x, [a if a is not None else 0 for a in accs],
                  color=colors, width=0.6, zorder=3)

    # Vertical separator between the two groups
    ax.axvline(3.5, color="gray", linewidth=1.0, linestyle=":", alpha=0.6)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="chance (0.50)", zorder=2)

    # Group annotation text
    yann = ax.get_ylim()[0] + 0.005
    ax.annotate("← Individual heads", xy=(1.5, yann), fontsize=8, color="dimgray",
                ha="center", va="bottom")
    ax.annotate("← Combination strategies (ablation) →", xy=(5.5, yann), fontsize=8,
                color="dimgray", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Triplet Accuracy (Covers80)", fontsize=9)
    ax.set_title(
        "Probe C — Ablation: Combination Strategy vs. Individual Heads\n"
        "wt-mean=cover-tuned weights  |  concat=concatenation  |  prod-exp=product-of-experts",
        fontsize=9, loc="left",
    )
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, acc in zip(bars, accs):
        if acc is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=7.5,
            )
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Ablation figure saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Probe C — Cosine similarity heatmap (block-diagonal cover structure)
# ---------------------------------------------------------------------------


def plot_similarity_heatmap(
    projected: Dict,
    embeddings: Dict,
    out_pdf: Path,
    proj_names: Optional[List[str]] = None,
) -> None:
    """
    N×N cosine similarity heatmap with songs sorted so cover pairs are adjacent.
    A good head should show bright 2×2 blocks along the diagonal.
    Generates one panel per proj_name side-by-side.

    Best practices (ISMIR-level figures):
      - Use a perceptually uniform colormap (viridis / magma).
      - Mark ground-truth cover pair blocks with a thin red border.
      - Side-by-side comparison highlights what each head captures.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[WARN] matplotlib not installed — skipping heatmap.")
        return

    if proj_names is None:
        proj_names = ["raw", "head_mel", "head_rhy", "head_tim", "combined"]

    pairs = _covers80_pairs(embeddings)
    if not pairs:
        return

    # Sort songs so cover pairs sit adjacent (pair 0: idx 0,1 | pair 1: idx 2,3 | ...)
    ordered_keys: List[str] = []
    pair_boundaries: List[Tuple[int, int]] = []
    ref_proj = projected.get("raw", projected.get("combined", {}))
    for fa, fb in pairs:
        if fa in ref_proj and fb in ref_proj:
            start = len(ordered_keys)
            ordered_keys.extend([fa, fb])
            pair_boundaries.append((start, start + 2))

    if not ordered_keys:
        return

    # Filter to projections that have data
    show = [p for p in proj_names if projected.get(p)]
    n = len(show)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 0.5, 5), squeeze=False)
    for col, pname in enumerate(show):
        proj = projected[pname]
        vecs_list = []
        valid_keys = []
        for k in ordered_keys:
            if k in proj:
                vecs_list.append(proj[k])
                valid_keys.append(k)
        if not vecs_list:
            continue
        vecs = np.stack(vecs_list, axis=0)
        sim_mat = vecs @ vecs.T  # cosine similarity (unit vectors)

        ax = axes[0, col]
        im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="viridis",
                       aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(_PROJ_LABEL.get(pname, pname).replace("\n", " "), fontsize=10)
        ax.tick_params(labelbottom=False, labelleft=False)

        # Red 2×2 borders on diagonal cover pair blocks
        for start, _ in pair_boundaries:
            if start < len(valid_keys):
                rect = Rectangle(
                    (start - 0.5, start - 0.5), 2, 2,
                    linewidth=0.8, edgecolor="#E74C3C", facecolor="none",
                )
                ax.add_patch(rect)

    fig.suptitle(
        "Cosine Similarity Matrix — Covers80  (songs sorted by pair)\n"
        "Red squares = ground-truth cover pairs  (bright blocks = good cover detection)",
        fontsize=9,
    )
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Heatmap saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Probe C — Violin plot (within-pair vs. between-pair similarity distribution)
# ---------------------------------------------------------------------------


def plot_violin_cover(
    projected: Dict,
    embeddings: Dict,
    out_pdf: Path,
    seed: int = 42,
) -> None:
    """
    For each projection: two violins — within-pair (cover) vs. between-pair (random).
    The gap between the distributions quantifies discriminability.
    The combined head should show the widest gap.

    Best practices (ISMIR-level figures):
      - Red = within-pair (covers), Blue = between-pair (non-covers).
      - Show median line.  Add mean markers.
      - sharey=True so axes are directly comparable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping violin plot.")
        return

    rng = random.Random(seed)
    pairs = _covers80_pairs(embeddings)
    if not pairs:
        return
    pair_set = {frozenset([fa, fb]) for fa, fb in pairs}
    all_keys_e = list(embeddings.keys())

    show_projs = ["raw", "head_mel", "head_rhy", "head_tim",
                  "combined", "combined_wmean", "combined_concat", "combined_prod", "combined_max"]
    show_projs = [p for p in show_projs if projected.get(p)]

    fig, axes = plt.subplots(
        1, len(show_projs),
        figsize=(2.4 * len(show_projs), 4.2),
        sharey=True, squeeze=False,
    )

    for col, pname in enumerate(show_projs):
        proj = projected[pname]
        within, between = [], []

        for fa, fb in pairs:
            va = proj.get(fa)
            vb = proj.get(fb)
            if va is not None and vb is not None:
                within.append(float(np.dot(va, vb)))

        # Sample ~2000 random between-pair similarities
        pool = [k for k in all_keys_e if k in proj]
        for _ in range(min(2000, len(pool) * len(pool))):
            a, b = rng.sample(pool, 2)
            if frozenset([a, b]) not in pair_set:
                between.append(float(np.dot(proj[a], proj[b])))

        ax = axes[0, col]
        COLOR_W = "#E74C3C"   # red  — within (covers)
        COLOR_B = "#5B9BD5"   # blue — between (non-covers)

        try:
            vp = ax.violinplot(
                [within, between], positions=[0, 1],
                showmedians=True, showextrema=True,
            )
            for pc, c in zip(vp["bodies"], [COLOR_W, COLOR_B]):
                pc.set_facecolor(c)
                pc.set_alpha(0.75)
            # Mean markers
            for i, (vals, c) in enumerate([(within, COLOR_W), (between, COLOR_B)]):
                ax.scatter([i], [np.mean(vals)], color=c, s=30, zorder=5,
                           marker="D", label="mean" if (col == 0 and i == 0) else "")
        except Exception:
            ax.boxplot([within, between], positions=[0, 1])

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["covers\n(within)", "random\n(between)"], fontsize=7.5)
        ax.set_title(_PROJ_LABEL.get(pname, pname).replace("\n", " "), fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if col == 0:
            ax.set_ylabel("Cosine Similarity", fontsize=9)

    fig.suptitle(
        "Covers80 — Within-Pair vs. Between-Pair Cosine Similarity\n"
        "(wider gap + higher within = better cover detection)",
        fontsize=9,
    )
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Violin plot saved → {out_pdf}")


def run_probe_M(
    pkl_path: Path,
    heads: Dict[str, Optional[ProjectionHead]],
    device: str,
    batch_size: int = 512,
) -> Dict:
    """
    Probe M: evaluate all heads on the held-out melody triplet test set.

    Input pkl format (produced by extract_embeddings.py):
        {"embeddings": {clip_path: ndarray}, "triplets": [{anchor, positives, negative}],
         "embed_dim": N, "encoder": "mert", "model_id": "..."}

    This is a stronger test than Probe C because:
      - Triplets were constructed by JASCO conditioned on CQT pitch salience → ground truth
        is purely melodic identity, not cover-version agreements
      - H_mel should dominate; H_rhy and H_tim should be near-random (≈50%)
    """
    print(f"\n=== Probe M: Pure Melody Similarity ({pkl_path.name}) ===")

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, dict) or "embeddings" not in data or "triplets" not in data:
        return {"error": f"Expected MERIT triplet pkl format in {pkl_path}"}

    embeddings: Dict[str, np.ndarray] = data["embeddings"]
    raw_triplets: list = data["triplets"]
    embed_dim: int = data["embed_dim"]
    print(f"  {len(embeddings)} embeddings, {len(raw_triplets)} folders, embed_dim={embed_dim}")

    # k² expansion (same as train_head.py / evaluate.py)
    expanded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    missing = 0
    for t in raw_triplets:
        a_emb = embeddings.get(t["anchor"])
        n_emb = embeddings.get(t["negative"])
        if a_emb is None or n_emb is None:
            missing += 1
            continue
        pos_embs = [embeddings[p] for p in t["positives"] if p in embeddings]
        if not pos_embs:
            missing += 1
            continue
        for p_emb in pos_embs:
            expanded.append((a_emb, p_emb, n_emb))
        for i, pi in enumerate(pos_embs):
            for j, pj in enumerate(pos_embs):
                if i != j:
                    expanded.append((pi, pj, n_emb))
    if missing:
        print(f"  [WARN] {missing} folders skipped (missing embeddings)")
    print(f"  k² expansion → {len(expanded)} triplets")

    # Build per-head projected triplets
    all_embs_list = list(embeddings.keys())
    all_embs_mat = np.stack([embeddings[k] for k in all_embs_list], axis=0)

    # Helper: project entire embedding matrix through a head
    def _project_matrix(head: ProjectionHead) -> Dict[str, np.ndarray]:
        out_vecs = []
        for start in range(0, len(all_embs_list), batch_size):
            batch = torch.from_numpy(all_embs_mat[start : start + batch_size]).float().to(device)
            out_vecs.append(head(batch).detach().cpu().numpy())
        mat = np.concatenate(out_vecs, axis=0)
        return {k: mat[i] for i, k in enumerate(all_embs_list)}

    # Raw baseline (L2-normalised)
    raw_norm = all_embs_mat / (np.linalg.norm(all_embs_mat, axis=1, keepdims=True) + 1e-9)
    proj_raw = {k: raw_norm[i] for i, k in enumerate(all_embs_list)}

    results: Dict = {}
    head_projs: Dict[str, Dict] = {}
    order = [("raw", None)] + [(n, h) for n, h in heads.items()]

    for proj_name, head in order:
        proj_dict = proj_raw if head is None else (_project_matrix(head) if head is not None else {})
        if not proj_dict:
            results[proj_name] = {"triplet_accuracy": None}
            continue

        triplets_proj = []
        for a_e, p_e, n_e in expanded:
            # Map the raw embedding back to a key via object identity isn't reliable;
            # project each embedding on the fly using a lookup by value address is unwieldy.
            # Instead we use the pre-built projection dict keyed on clip paths.
            # expanded contains raw np arrays — we need to look up by path.
            # Rebuild expansion using projected embeddings instead.
            pass

        # --- Cleaner: treat expanded as raw np triplets and project inline ---
        # Re-project each triplet element through this head
        if head is not None:
            @torch.no_grad()
            def _proj_arr(arr: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                return head(t).cpu().numpy()[0]
        else:
            def _proj_arr(arr: np.ndarray) -> np.ndarray:  # type: ignore[misc]
                norm = np.linalg.norm(arr)
                return arr / (norm + 1e-9)

        correct, total = 0, 0
        batch_a, batch_p, batch_n = [], [], []
        BSIZE = batch_size

        def _flush(ba, bp, bn):
            nonlocal correct, total
            if not ba:
                return
            if head is not None:
                with torch.no_grad():
                    ta = head(torch.from_numpy(np.stack(ba)).float().to(device)).cpu().numpy()
                    tp = head(torch.from_numpy(np.stack(bp)).float().to(device)).cpu().numpy()
                    tn = head(torch.from_numpy(np.stack(bn)).float().to(device)).cpu().numpy()
            else:
                def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
                ta, tp, tn = l2(np.stack(ba)), l2(np.stack(bp)), l2(np.stack(bn))
            s_ap = (ta * tp).sum(axis=1)
            s_an = (ta * tn).sum(axis=1)
            correct += int((s_ap > s_an).sum())
            total += len(ba)

        for a_e, p_e, n_e in expanded:
            batch_a.append(a_e)
            batch_p.append(p_e)
            batch_n.append(n_e)
            if len(batch_a) >= BSIZE:
                _flush(batch_a, batch_p, batch_n)
                batch_a, batch_p, batch_n = [], [], []
        _flush(batch_a, batch_p, batch_n)

        acc = correct / total if total else 0.0
        results[proj_name] = {"triplet_accuracy": round(acc, 4), "n_triplets": total}
        print(f"  {proj_name:12s}  triplet_acc = {acc:.4f}  (n={total})")

    return results


# ---------------------------------------------------------------------------
# Visualisation — Probe M bar chart
# ---------------------------------------------------------------------------


def plot_probe_M(results: Dict, out_pdf: Path) -> None:
    """Bar chart: triplet accuracy for raw / H_mel / H_rhy / H_tim on melody test set."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping Probe M figure.")
        return

    order = ["raw", "head_mel", "head_rhy", "head_tim"]
    accs = [results.get(k, {}).get("triplet_accuracy") for k in order]
    labels = [_PROJ_LABEL[k] for k in order]
    colors = [_PROJ_COLOR[k] for k in order]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = list(range(len(order)))
    bars = ax.bar(x, [a if a is not None else 0 for a in accs],
                  color=colors, width=0.5, zorder=3)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Triplet Accuracy", fontsize=9)
    ax.set_title("Probe M — Pure Melody Similarity\n(held-out JASCO melody triplets)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, acc in zip(bars, accs):
        if acc is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Probe M figure saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MERIT heads on external probe datasets."
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to MERT embeddings pkl. For merit_triplets format, path to MERIT triplet pkl "
             "(e.g. mel_mert_test.pkl). For other formats, output of encode_folder.py.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata file (required for --meta-format mtgjamendo).",
    )
    parser.add_argument(
        "--meta-format",
        required=True,
        choices=["mtgjamendo", "ballroom", "irmas", "musdb18", "covers80", "merit_triplets"],
        help=(
            "Dataset format. "
            "merit_triplets: MERIT pkl with embeddings+triplets fields (e.g. mel_mert_test.pkl). "
            "irmas/ballroom: directory-name class labels. "
            "musdb18: stem type from filename (vocals.wav/drums.wav/bass.wav/other.wav). "
            "covers80: two files per subdirectory = cover pair."
        ),
    )
    parser.add_argument(
        "--heads-dir",
        required=True,
        help="Directory containing head_mel/, head_rhy/, head_tim/ subdirs.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--visualize",
        default=None,
        help="Path for output figure PDF (covers80: two-panel; merit_triplets: bar chart).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--map-k",
        type=int,
        default=10,
        help="k value for mAP@k computation (Probe A) and n_neg_per_query (Probe B). Default: 10",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default=None,
        help="Comma-separated class names to exclude from evaluation (e.g., 'mixture' for MUSDB18)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    # --- Probe M (merit_triplets) uses a different loading path ---
    if args.meta_format == "merit_triplets":
        print(f"Loading embeddings from {args.embeddings} ...")
        # Peek at embed_dim for head loading
        with open(args.embeddings, "rb") as fh:
            _peek = pickle.load(fh)
        embed_dim = _peek["embed_dim"]
        print(f"  embed_dim = {embed_dim}")
        print(f"Loading heads from {args.heads_dir} ...")
        heads = load_heads(Path(args.heads_dir), embed_dim, args.device)
        results = run_probe_M(Path(args.embeddings), heads, args.device, args.batch_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved → {out_path}")
        if args.visualize:
            plot_probe_M(results, Path(args.visualize))
        return

    # --- All other probes: load flat embeddings from encode_folder.py ---
    print(f"Loading embeddings from {args.embeddings} ...")
    embeddings, embed_dim = load_embeddings(Path(args.embeddings))
    print(f"  {len(embeddings)} files, embed_dim = {embed_dim}")

    print(f"Loading heads from {args.heads_dir} ...")
    heads = load_heads(Path(args.heads_dir), embed_dim, args.device)

    print("Projecting embeddings through all heads ...")
    projected = project_all(embeddings, heads, args.device, batch_size=args.batch_size)

    # Parse exclude_classes
    exclude_classes = []
    if args.exclude_classes:
        exclude_classes = [x.strip() for x in args.exclude_classes.split(',')]

    if args.meta_format == "mtgjamendo":
        if not args.metadata:
            parser.error("--metadata is required for --meta-format mtgjamendo")
        results = run_probe_A(projected, embeddings, Path(args.metadata), k=args.map_k)

    elif args.meta_format in ("ballroom", "irmas"):
        results = run_probe_B(projected, n_neg_per_query=args.map_k, k=args.map_k, seed=args.seed, exclude_classes=exclude_classes)

    elif args.meta_format == "musdb18":
        results = run_probe_B(projected, n_neg_per_query=args.map_k, k=args.map_k, seed=args.seed, label_extractor=_musdb18_stem_from_path, exclude_classes=exclude_classes)

    elif args.meta_format == "covers80":
        results = run_probe_C(projected, embeddings, seed=args.seed)
        if args.visualize:
            vis = Path(args.visualize)
            plot_probe_C(results, vis)
            # Derive sibling paths for extra figures
            vis_stem = str(vis.parent / vis.stem)
            plot_ablation_C(results, Path(vis_stem + "_ablation.pdf"))
            plot_similarity_heatmap(projected, embeddings, Path(vis_stem + "_heatmap.pdf"))
            plot_violin_cover(projected, embeddings, Path(vis_stem + "_violin.pdf"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
