#!/usr/bin/env python3
"""
Generate N *timbre* triplets from MoisesDB without using any generative model.

Goal (用户目标A): 同音色不同旋律（做相似度数据）

Triplet definition (finest granularity):
  - Anchor: a 10s segment from a single MoisesDB track with a specific `trackType`
  - Positive: same `trackType` but from a different song (different track_id)
  - Negative: different instrument from the *same song* (different `stemName`), ideally at the same offset

Usage:
  python pipeline.py <num_triplets>

Outputs:
  outputs/timbre_triplets/triplets_1/, outputs/timbre_triplets/triplets_2/, ...
Each triplets_* folder contains:
  - input_A_segment.wav (raw segment at original sr/channels)
  - input_B01_segment.wav ... input_B05_segment.wav (raw positives)
  - input_N_segment.wav (raw segment at original sr/channels)
  - run_meta.json
  - manifest.jsonl
  - triplet/
      - anchor.wav (mono, resampled to 32k)
      - positive_01.wav ... positive_05.wav (mono, resampled to 32k)
      - negative.wav (mono, resampled to 32k)
      - triplet_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triplets_input_index.index_builder import build_timbre_index

_MOISESDB_ROOT_ENV = os.environ.get("MOISESDB_ROOT", "").strip()
if not _MOISESDB_ROOT_ENV:
    raise SystemExit(
        "Error: MOISESDB_ROOT environment variable is not set.\n"
        "Set it to the parent directory of moisesdb_v0.1/:  "
        "export MOISESDB_ROOT=/path/to/moisesdb"
    )
MOISESDB_ROOT = Path(_MOISESDB_ROOT_ENV)
MOISESDB_VERSION_DIRNAME = "moisesdb_v0.1"

CACHE_DIR = SCRIPT_DIR / "cache"
SEGMENT_CACHE_PATH = CACHE_DIR / "segment_quality_cache.json"


def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _load_index_from_env(env_name: str, expected_generator: str) -> dict | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise SystemExit(f"{env_name} points to a missing JSON: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read {env_name}={path}: {e}")
    if str(data.get("generator", "")).strip() != expected_generator:
        raise SystemExit(f"{env_name}={path} is not a valid {expected_generator} index JSON.")
    print(f"Using custom {expected_generator} index from {path}")
    return data


def _next_triplets_index(out_base: Path) -> int:
    if not out_base.exists():
        return 1
    best = 0
    for p in out_base.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("triplets_"):
            continue
        tail = p.name[len("triplets_") :]
        if tail.isdigit():
            best = max(best, int(tail))
    return best + 1


def _collect_existing_anchor_track_uuids(out_base: Path) -> set[str]:
    """
    Avoid repeating anchors across multiple invocations by scanning existing outputs.
    """
    used: set[str] = set()
    if not out_base.exists():
        return used
    for run_dir in out_base.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("triplets_"):
            continue
        meta = run_dir / "triplet" / "triplet_meta.json"
        if not meta.exists():
            continue
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        anchor = (obj.get("inputs") or {}).get("anchor") or {}
        track_uuid = anchor.get("track_uuid")
        if isinstance(track_uuid, str) and track_uuid.strip():
            used.add(track_uuid.strip())
    return used


@dataclass(frozen=True)
class TrackInfo:
    song_id: str
    stem: str
    track_uuid: str
    track_type: str
    has_bleed: Optional[bool]
    wav_rel: str


@dataclass(frozen=True)
class QualityGate:
    segment_duration: float = 10.0
    search_hop_sec: float = 1.0
    frame_sec: float = 0.05
    frame_hop_sec: float = 0.025
    top_db: float = 40.0
    min_non_silent_ratio: float = 0.90
    min_rms: float = 0.003
    out_sample_rate: int = 32000


def _scan_moisesdb_tracks(moises_root: Path) -> List[TrackInfo]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base} (expected extracted DB at {moises_root})")

    out: List[TrackInfo] = []
    for song_dir in sorted(base.iterdir()):
        if not song_dir.is_dir():
            continue
        data_json = song_dir / "data.json"
        if not data_json.exists():
            continue
        try:
            obj = json.loads(data_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        stems = obj.get("stems") or []
        if not isinstance(stems, list):
            continue
        for stem in stems:
            if not isinstance(stem, dict):
                continue
            stem_name = str(stem.get("stemName") or "").strip()
            if not stem_name:
                continue
            tracks = stem.get("tracks") or []
            if not isinstance(tracks, list):
                continue
            for tr in tracks:
                if not isinstance(tr, dict):
                    continue
                track_uuid = str(tr.get("id") or "").strip()
                track_type = str(tr.get("trackType") or "").strip()
                if not track_uuid or not track_type:
                    continue
                has_bleed = tr.get("has_bleed")
                if has_bleed is None and "hasBleed" in tr:
                    has_bleed = tr.get("hasBleed")
                if has_bleed is not None:
                    has_bleed = bool(has_bleed)

                wav_path = song_dir / stem_name / f"{track_uuid}.wav"
                if not wav_path.exists():
                    continue
                wav_rel = (Path(MOISESDB_VERSION_DIRNAME) / song_dir.name / stem_name / wav_path.name).as_posix()
                out.append(
                    TrackInfo(
                        song_id=song_dir.name,
                        stem=stem_name,
                        track_uuid=track_uuid,
                        track_type=track_type,
                        has_bleed=has_bleed,
                        wav_rel=wav_rel,
                    )
                )
    return out


def _load_quality_cache(path: Path, expected_config: dict) -> dict:
    if not path.exists():
        return {"schema_version": 1, "config": expected_config, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": 1, "config": expected_config, "entries": {}}

    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        return {"schema_version": 1, "config": expected_config, "entries": {}}
    if data.get("config") != expected_config or not isinstance(data.get("entries"), dict):
        print(f"[WARN] Segment quality cache config changed; ignoring old cache: {path}")
        return {"schema_version": 1, "config": expected_config, "entries": {}}
    return data


def _save_quality_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _segment_stats_from_mono(mono: torch.Tensor, sr: int, gate: QualityGate) -> dict:
    """
    Compute basic stats for a mono segment: RMS and a non-silent frame ratio.
    The non-silent ratio is computed w.r.t. the segment's peak frame-RMS (top_db threshold).
    """
    mono = mono.float()
    if mono.numel() == 0:
        return {"rms": 0.0, "non_silent_ratio": 0.0, "max_frame_rms": 0.0, "threshold_rms": 0.0}

    rms = float(mono.pow(2).mean().sqrt().item())

    frame_len = max(8, int(round(gate.frame_sec * sr)))
    frame_hop = max(1, int(round(gate.frame_hop_sec * sr)))
    x = mono.pow(2).view(1, 1, -1)
    if x.shape[-1] < frame_len:
        x = F.pad(x, (0, frame_len - x.shape[-1]))
    frame_ms = F.avg_pool1d(x, kernel_size=frame_len, stride=frame_hop).squeeze(0).squeeze(0)
    frame_rms = frame_ms.clamp_min(0).sqrt()
    if frame_rms.numel() == 0:
        return {"rms": rms, "non_silent_ratio": 0.0, "max_frame_rms": 0.0, "threshold_rms": 0.0}

    max_frame_rms = float(frame_rms.max().item())
    if max_frame_rms <= 0.0:
        return {"rms": rms, "non_silent_ratio": 0.0, "max_frame_rms": 0.0, "threshold_rms": 0.0}

    threshold = float(max_frame_rms) * (10.0 ** (-float(gate.top_db) / 20.0))
    non_silent_ratio = float((frame_rms >= float(threshold)).float().mean().item())
    return {
        "rms": float(rms),
        "non_silent_ratio": float(non_silent_ratio),
        "max_frame_rms": float(max_frame_rms),
        "threshold_rms": float(threshold),
    }


def _pick_best_offset_for_track(mono: torch.Tensor, sr: int, gate: QualityGate) -> Optional[dict]:
    """
    Pick the best 10s window by:
      1) maximizing non_silent_ratio (computed using a *track-level* threshold)
      2) tie-break with window RMS
    """
    mono = mono.float()
    length = int(round(gate.segment_duration * sr))
    if length <= 0:
        raise ValueError("segment_duration must be > 0")
    if mono.numel() < length:
        # Too short; allow but quality will likely be low.
        mono = F.pad(mono, (0, length - mono.numel()))

    total = int(mono.numel())
    max_start = max(0, total - length)
    hop = max(1, int(round(gate.search_hop_sec * sr)))
    candidates = list(range(0, max_start + 1, hop))
    if candidates and candidates[-1] != max_start:
        candidates.append(max_start)
    if not candidates:
        candidates = [0]

    # Track-level frame RMS for non-silent ratio (top_db threshold).
    frame_len = max(8, int(round(gate.frame_sec * sr)))
    frame_hop = max(1, int(round(gate.frame_hop_sec * sr)))
    x = mono.pow(2).view(1, 1, -1)
    if x.shape[-1] < frame_len:
        x = F.pad(x, (0, frame_len - x.shape[-1]))
    frame_ms = F.avg_pool1d(x, kernel_size=frame_len, stride=frame_hop).squeeze(0).squeeze(0)
    frame_rms = frame_ms.clamp_min(0).sqrt()
    if frame_rms.numel() == 0:
        return None
    track_max_frame_rms = float(frame_rms.max().item())
    if track_max_frame_rms <= 0.0:
        return None
    threshold = float(track_max_frame_rms) * (10.0 ** (-float(gate.top_db) / 20.0))

    active = (frame_rms >= float(threshold)).to(torch.int32)
    csum_active = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(active, dim=0)], dim=0)

    # Sample-level cumulative sum for exact RMS per window.
    sq = mono.pow(2)
    csum_sq = torch.cat([torch.zeros(1, dtype=sq.dtype), torch.cumsum(sq, dim=0)], dim=0)

    n_frames = int(frame_rms.numel())

    def _frame_index(sample_idx: int) -> int:
        # Map sample index to frame index (approx).
        return int(sample_idx // frame_hop)

    best: Optional[dict] = None
    for s in candidates:
        e = int(s + length)

        # Window RMS.
        mean_sq = float((csum_sq[e] - csum_sq[s]).item()) / float(max(1, length))
        win_rms = float(math.sqrt(max(0.0, mean_sq)))

        # Window non-silent ratio using track-level threshold.
        sf = _frame_index(int(s))
        ef = _frame_index(int(e + frame_hop - 1))
        sf = max(0, min(sf, n_frames))
        ef = max(sf + 1, min(ef, n_frames))
        denom = int(ef - sf)
        active_count = int((csum_active[ef] - csum_active[sf]).item())
        ratio = float(active_count) / float(max(1, denom))

        cand = {
            "offset_sec": float(s) / float(sr),
            "sr": int(sr),
            "ns": int(length),
            "non_silent_ratio": float(ratio),
            "rms": float(win_rms),
            "track_max_frame_rms": float(track_max_frame_rms),
            "threshold_rms": float(threshold),
        }
        if best is None:
            best = cand
        else:
            if (cand["non_silent_ratio"], cand["rms"]) > (best["non_silent_ratio"], best["rms"]):
                best = cand
    return best


def _score_track_best_segment(
    wav_path: Path,
    wav_rel: str,
    cache_entries: dict,
    gate: QualityGate,
) -> Tuple[Optional[dict], bool]:
    """
    Return best segment stats for this track (auto-picked offset). Uses cache.
    Returns (best, cache_hit).
    """
    ent = cache_entries.get(wav_rel)
    if isinstance(ent, dict):
        best = ent.get("best")
        if isinstance(best, dict) and ent.get("status") == "ok":
            return best, True
        if ent.get("status") in ("too_silent", "error"):
            return None, True

    try:
        wav, sr = torchaudio.load(str(wav_path))
        mono = wav.float().mean(dim=0)
        best = _pick_best_offset_for_track(mono, int(sr), gate)
        if best is None:
            cache_entries[wav_rel] = {
                "status": "too_silent",
                "best": None,
                "last_error": "no_frames_or_silent",
                "updated_at": time.time(),
            }
            return None, False
        if float(best["rms"]) < float(gate.min_rms) or float(best["non_silent_ratio"]) < float(gate.min_non_silent_ratio):
            cache_entries[wav_rel] = {
                "status": "too_silent",
                "best": best,
                "last_error": "below_gate",
                "updated_at": time.time(),
            }
            return None, False
        cache_entries[wav_rel] = {"status": "ok", "best": best, "last_error": None, "updated_at": time.time()}
        return best, False
    except Exception as e:
        cache_entries[wav_rel] = {"status": "error", "best": None, "last_error": str(e), "updated_at": time.time()}
        return None, False


def _extract_segment_at_offset(
    wav: torch.Tensor, sr: int, offset_sec: float, segment_duration: float
) -> Tuple[torch.Tensor, float]:
    """
    Extract [C, T] segment at offset_sec, padding if needed.
    Returns (segment, used_offset_sec).
    """
    length = int(round(segment_duration * sr))
    if length <= 0:
        raise ValueError("segment_duration must be > 0")
    if wav.ndim != 2:
        raise ValueError("Expected wav of shape [C, T].")
    total = int(wav.shape[1])

    start = int(round(max(0.0, float(offset_sec)) * sr))
    if start >= total:
        raise ValueError(f"offset_sec={offset_sec:.3f} is beyond audio length ({total/sr:.3f}s).")

    end = start + length
    seg = wav[:, start:end]
    if seg.shape[1] < length:
        seg = F.pad(seg, (0, length - seg.shape[1]))
    used_offset_sec = float(start) / float(sr)
    return seg, used_offset_sec


def _save_raw_segment(wav_path: Path, out_path: Path, offset_sec: float, gate: QualityGate) -> dict:
    wav, sr = torchaudio.load(str(wav_path))  # [C, T]
    seg, used_off = _extract_segment_at_offset(wav, int(sr), float(offset_sec), gate.segment_duration)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), seg, int(sr), encoding="PCM_S", bits_per_sample=16)
    return {"sr": int(sr), "ns": int(seg.shape[1]), "offset_sec": float(used_off)}


def _write_mono_resampled(src_segment_path: Path, out_wav_path: Path, out_sr: int) -> None:
    wav, sr = torchaudio.load(str(src_segment_path))  # [C, T]
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(out_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(out_sr))
    wav = wav.contiguous()
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav_path), wav, int(out_sr), encoding="PCM_S", bits_per_sample=16)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate N timbre triplets (no model).")
    ap.add_argument("num_triplets", type=int, help="How many triplets to generate.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to save triplets (default: <script_dir>/outputs/timbre_triplets).")
    ap.add_argument("--num-positives", type=int, default=5,
                    help="Number of positives per anchor (default: 5).")
    ap.add_argument("--min-positives", type=int, default=3,
                    help="Skip anchors with fewer than this many available positives (default: 3).")
    args = ap.parse_args()

    if args.num_triplets <= 0:
        raise SystemExit("num_triplets must be > 0")

    if not MOISESDB_ROOT.exists():
        raise SystemExit(
            f"MoisesDB not found: {MOISESDB_ROOT}\n"
            "Ensure MOISESDB_ROOT points to the directory containing moisesdb_v0.1/.\n"
            "  export MOISESDB_ROOT=/path/to/moisesdb"
        )

    gate = QualityGate()
    print("Loading prefiltered timbre index...")
    timbre_index = _load_index_from_env("TRIPLETS_TIMBRE_INDEX_JSON", "timbre")
    if timbre_index is None:
        timbre_index = build_timbre_index(force=False, verbose=True)
    index_entries = list(timbre_index.get("entries") or [])
    index_anchors = list(timbre_index.get("anchors") or [])
    if not index_entries or not index_anchors:
        raise SystemExit(
            "Timbre index is empty. "
            f"Please rebuild it: python {REPO_ROOT / 'triplets_input_index' / 'build_index.py'} timbre --force"
        )

    tracks_by_type: Dict[str, List[dict]] = {}
    for entry in index_entries:
        tracks_by_type.setdefault(str(entry["trackType"]), []).append(entry)
    for k in tracks_by_type:
        tracks_by_type[k].sort(key=lambda x: (str(x["song_id"]), str(x["stem"]), str(x["track_uuid"])))

    allowed_types = set(str(x) for x in (timbre_index.get("allowed_track_types") or []))
    print(
        f"MoisesDB: valid_entries={len(index_entries)}, "
        f"anchors={len(index_anchors)}, allowed_trackTypes(>=2 songs)={len(allowed_types)}"
    )

    out_base = args.output_dir if args.output_dir is not None else SCRIPT_DIR / "outputs" / "timbre_triplets"
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    start_idx = _next_triplets_index(out_base)
    end_idx = start_idx + args.num_triplets - 1
    print(f"Output base: {out_base}")
    print(f"Generating triplets_{start_idx} .. triplets_{end_idx}")

    used_anchor_uuids = _collect_existing_anchor_track_uuids(out_base)
    if used_anchor_uuids:
        print(f"Found {len(used_anchor_uuids)} existing anchors; will avoid reusing their track_uuid for anchor.")

    def _get_wav_path(entry: dict) -> Path:
        return MOISESDB_ROOT / str(entry["wav_rel"])

    def _find_positives(anchor: dict, k: int) -> List[dict]:
        """Find up to k positives: same trackType, each from a different song."""
        lst = list(tracks_by_type.get(str(anchor["trackType"])) or [])
        if len(lst) < 2:
            return []
        # Shuffle deterministically so different anchors pick different subsets.
        rng_local = random.Random(str(anchor["track_uuid"]))
        rng_local.shuffle(lst)
        result = []
        used_song_ids = {str(anchor["song_id"])}
        for cand in lst:
            if str(cand["track_uuid"]) == str(anchor["track_uuid"]):
                continue
            if str(cand["song_id"]) in used_song_ids:
                continue
            cand_best = dict(cand.get("quality_best_window") or {})
            if not cand_best:
                continue
            result.append(cand)
            used_song_ids.add(str(cand["song_id"]))
            if len(result) >= k:
                break
        return result

    anchor_cursor = 0
    for trip_idx in range(start_idx, end_idx + 1):
        run_dir = out_base / f"triplets_{trip_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        triplet_dir = run_dir / "triplet"
        triplet_dir.mkdir(parents=True, exist_ok=True)

        # Find a valid (anchor, positive, negative) triple.
        chosen = None
        scanned = 0
        while anchor_cursor < len(index_anchors):
            anchor_rec = index_anchors[anchor_cursor]
            anchor_cursor += 1
            scanned += 1
            anchor = dict(anchor_rec.get("anchor") or {})
            negative_candidates = list(anchor_rec.get("negative_candidates") or [])

            if str(anchor.get("track_uuid", "")) in used_anchor_uuids:
                continue
            if str(anchor.get("trackType", "")) not in allowed_types:
                continue

            # Positives: same trackType, each from a different song.
            positives = _find_positives(anchor, args.num_positives)
            if len(positives) < args.min_positives:
                continue

            if not negative_candidates:
                continue
            negative = dict(negative_candidates[0])
            a_best = dict(anchor.get("quality_best_window") or {})
            n_stats_at_a = dict(negative.get("quality_at_anchor_offset") or {})
            if not a_best or not n_stats_at_a:
                continue
            positives_bests = [dict(p.get("quality_best_window") or {}) for p in positives]
            if not all(positives_bests):
                continue

            chosen = (anchor, a_best, positives, positives_bests, negative, n_stats_at_a)
            break

        if chosen is None:
            raise SystemExit(
                f"Could not find a valid triplet for triplets_{trip_idx}. "
                f"scanned={scanned}, cursor={anchor_cursor}/{len(index_anchors)}. "
                "Try rebuilding the timbre index or generating fewer triplets."
            )

        anchor, a_best, positives, positives_bests, negative, n_stats_at_a = chosen
        used_anchor_uuids.add(str(anchor["track_uuid"]))

        # Save raw segments (original sr/channels).
        a_seg_path = run_dir / "input_A_segment.wav"
        n_seg_path = run_dir / "input_N_segment.wav"

        a_raw = _save_raw_segment(_get_wav_path(anchor), a_seg_path, float(a_best["offset_sec"]), gate)
        # Negative uses the same offset as anchor (or the closest valid offset from extraction).
        n_raw = _save_raw_segment(_get_wav_path(negative), n_seg_path, float(a_best["offset_sec"]), gate)

        # Save raw segments for each positive.
        b_seg_paths = []
        b_raws = []
        for pi, (pos, p_best) in enumerate(zip(positives, positives_bests)):
            b_seg_path = run_dir / f"input_B{pi+1:02d}_segment.wav"
            b_raw_i = _save_raw_segment(_get_wav_path(pos), b_seg_path, float(p_best["offset_sec"]), gate)
            b_seg_paths.append(b_seg_path)
            b_raws.append(b_raw_i)

        # Write final triplet audio (mono 32k).
        anchor_out = triplet_dir / "anchor.wav"
        neg_out = triplet_dir / "negative.wav"
        _write_mono_resampled(a_seg_path, anchor_out, gate.out_sample_rate)
        _write_mono_resampled(n_seg_path, neg_out, gate.out_sample_rate)
        pos_outs = []
        for pi, b_seg_path in enumerate(b_seg_paths):
            pos_out = triplet_dir / f"positive_{pi+1:02d}.wav"
            _write_mono_resampled(b_seg_path, pos_out, gate.out_sample_rate)
            pos_outs.append(pos_out)

        # Manifest (minimal, consistent with other generators).
        manifest_path = run_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        def _append_manifest(obj: dict) -> None:
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        _append_manifest(
            {"type": "anchor", "output_wav": _relpath(anchor_out), "prompt": None, "seed": None, "stats": {"source": "input_A_segment"}}
        )
        for pi, pos_out in enumerate(pos_outs):
            _append_manifest(
                {
                    "type": "positive",
                    "positive_index": pi + 1,
                    "output_wav": _relpath(pos_out),
                    "prompt": None,
                    "seed": None,
                    "stats": {"source": f"input_B{pi+1:02d}_segment"},
                }
            )
        _append_manifest(
            {
                "type": "negative",
                "output_wav": _relpath(neg_out),
                "prompt": None,
                "seed": None,
                "stats": {"source": "input_N_segment", "same_song_as_anchor": True, "offset_matches_anchor": True},
            }
        )

        # Metadata.
        triplet_meta = {
            "triplet_index": int(trip_idx),
            "triplet_dir": _relpath(triplet_dir),
            "inputs": {
                "dataset": "moisesdb",
                "moisesdb_root": str(MOISESDB_ROOT),
                "anchor": {
                    "song_id": anchor["song_id"],
                    "stem": anchor["stem"],
                    "track_uuid": anchor["track_uuid"],
                    "trackType": anchor["trackType"],
                    "has_bleed": anchor.get("has_bleed"),
                    "wav": str(_get_wav_path(anchor)),
                    "wav_rel": anchor["wav_rel"],
                    "offset_sec": float(a_raw["offset_sec"]),
                    "segment_saved": _relpath(a_seg_path),
                    "quality_best_window": a_best,
                },
                "positives": [
                    {
                        "song_id": pos["song_id"],
                        "stem": pos["stem"],
                        "track_uuid": pos["track_uuid"],
                        "trackType": pos["trackType"],
                        "has_bleed": pos.get("has_bleed"),
                        "wav": str(_get_wav_path(pos)),
                        "wav_rel": pos["wav_rel"],
                        "offset_sec": float(b_raw_i["offset_sec"]),
                        "segment_saved": _relpath(b_seg_path_i),
                        "quality_best_window": p_best_i,
                    }
                    for pos, b_raw_i, b_seg_path_i, p_best_i in zip(positives, b_raws, b_seg_paths, positives_bests)
                ],
                "negative": {
                    "song_id": negative["song_id"],
                    "stem": negative["stem"],
                    "track_uuid": negative["track_uuid"],
                    "trackType": negative["trackType"],
                    "has_bleed": negative.get("has_bleed"),
                    "wav": str(_get_wav_path(negative)),
                    "wav_rel": negative["wav_rel"],
                    "offset_sec": float(n_raw["offset_sec"]),
                    "segment_saved": _relpath(n_seg_path),
                    "quality_at_anchor_offset": n_stats_at_a,
                },
            },
            "quality_gate": asdict(gate),
            "outputs": {
                "anchor_wav": _relpath(anchor_out),
                "negative_wav": _relpath(neg_out),
                "positives": [_relpath(p) for p in pos_outs],
            },
        }
        (triplet_dir / "triplet_meta.json").write_text(
            json.dumps(triplet_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        run_meta = {
            "note": "No-model timbre triplet. Anchor/Positive share MoisesDB trackType; Negative is a different stem from the same song.",
            "quality_gate": asdict(gate),
            "inputs": {
                "dataset": "moisesdb",
                "moisesdb_root": str(MOISESDB_ROOT),
                "anchor_wav": str(_get_wav_path(anchor)),
                "positive_wavs": [str(_get_wav_path(pos)) for pos in positives],
                "negative_wav": str(_get_wav_path(negative)),
                "A_segment_saved": _relpath(a_seg_path),
                "B_segments_saved": [_relpath(p) for p in b_seg_paths],
                "N_segment_saved": _relpath(n_seg_path),
            },
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        pos_summary = ", ".join(
            f"P{i+1}={pos['song_id']}/{pos['stem']} off={b_raw_i['offset_sec']:.2f}s"
            for i, (pos, b_raw_i) in enumerate(zip(positives, b_raws))
        )
        print(
            f"[triplets_{trip_idx}] trackType={anchor['trackType']} | "
            f"A={anchor['song_id']}/{anchor['stem']} off={a_raw['offset_sec']:.2f}s | "
            f"{pos_summary} | "
            f"N={negative['song_id']}/{negative['stem']} off={n_raw['offset_sec']:.2f}s"
        )


if __name__ == "__main__":
    main()
