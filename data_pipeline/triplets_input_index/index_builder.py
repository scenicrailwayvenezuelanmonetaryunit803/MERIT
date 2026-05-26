#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Threads per build function. GIL is released for torch/librosa C-extension work,
# so threads outperform processes here (zero serialisation overhead).
# Override with TRIMUS_NUM_WORKERS env var.
NUM_WORKERS: int = int(os.environ.get("TRIMUS_NUM_WORKERS", str(min(os.cpu_count() or 8, 32))))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
_MOISESDB_ROOT_ENV = os.environ.get("MOISESDB_ROOT", "").strip()
# MOISESDB_ROOT should point to the *parent* of moisesdb_v0.1/
# e.g. $MOISESDB_ROOT   (not .../moisesdb/moisesdb_v0.1)
if not _MOISESDB_ROOT_ENV:
    raise SystemExit(
        "Error: MOISESDB_ROOT environment variable is not set.\n"
        "Set it to the parent directory of moisesdb_v0.1/:  "
        "export MOISESDB_ROOT=/path/to/moisesdb"
    )
MOISESDB_ROOT = Path(_MOISESDB_ROOT_ENV)
MOISESDB_VERSION_DIRNAME = "moisesdb_v0.1"

INDEX_DIR = SCRIPT_DIR / "indexes"
SAMPLE_INDEX_DIR = SCRIPT_DIR / "samples"
MELODY_INDEX_PATH = INDEX_DIR / "melody_index.json"
RHYTHM_INDEX_PATH = INDEX_DIR / "rhythm_index.json"
TIMBRE_INDEX_PATH = INDEX_DIR / "timbre_index.json"

TMP_ROOT = Path("/tmp/triplets_input_index")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MELODY_GENERATOR_DIR = REPO_ROOT / "melody_triplets_generator"
if str(MELODY_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(MELODY_GENERATOR_DIR))


def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _resolve_index_path(default_path: Path, index_path: Optional[Path]) -> Path:
    return Path(index_path) if index_path is not None else default_path


def _load_existing_index(path: Path, generator: str, expected_config: dict) -> Optional[dict]:
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    if int(data.get("schema_version", 0)) != 1:
        return None
    if str(data.get("generator", "")) != generator:
        return None
    if data.get("config") != expected_config:
        return None
    return data


def _all_song_ids(moises_root: Path) -> List[str]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base}")
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def _resolve_song_subset(
    moises_root: Path,
    song_ids: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
) -> Optional[List[str]]:
    all_song_ids = _all_song_ids(moises_root)
    subset_requested = bool(song_ids) or (max_songs is not None)
    if song_ids:
        wanted = {str(x).strip() for x in song_ids if str(x).strip()}
        resolved = [sid for sid in all_song_ids if sid in wanted]
    else:
        resolved = list(all_song_ids)
    if max_songs is not None:
        resolved = resolved[: max(0, int(max_songs))]
    if subset_requested and not resolved:
        return []
    return resolved if len(resolved) < len(all_song_ids) else None


def _scan_moisesdb_tracks(moises_root: Path, stems: List[str], song_ids: Optional[List[str]] = None) -> List[dict]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base}")

    tracks: List[dict] = []
    stems = [s.strip() for s in stems if s.strip()]
    allowed = set(song_ids or [])
    for track_dir in sorted(base.iterdir()):
        if not track_dir.is_dir():
            continue
        if allowed and track_dir.name not in allowed:
            continue
        files: List[str] = []
        for stem in stems:
            stem_dir = track_dir / stem
            if not stem_dir.is_dir():
                continue
            for wav in sorted(stem_dir.glob("*.wav")):
                rel = (Path(MOISESDB_VERSION_DIRNAME) / track_dir.name / stem / wav.name).as_posix()
                files.append(rel)
        tracks.append({"track_id": track_dir.name, "files": files})
    return tracks


def _stem_from_relpath(relpath: str) -> Optional[str]:
    parts = relpath.split("/")
    if len(parts) < 3:
        return None
    return parts[-2]


def _scan_moisesdb_drums_tracks(moises_root: Path, song_ids: Optional[List[str]] = None) -> List[dict]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base}")

    out: List[dict] = []
    allowed = set(song_ids or [])
    for track_dir in sorted(base.iterdir()):
        if not track_dir.is_dir():
            continue
        if allowed and track_dir.name not in allowed:
            continue
        stem_dir = track_dir / "drums"
        if not stem_dir.is_dir():
            continue
        files: List[str] = []
        for wav in sorted(stem_dir.glob("*.wav")):
            rel = (Path(MOISESDB_VERSION_DIRNAME) / track_dir.name / "drums" / wav.name).as_posix()
            files.append(rel)
        if files:
            out.append({"track_id": track_dir.name, "stem": "drums", "files": files})
    return out


@dataclass(frozen=True)
class TrackInfo:
    song_id: str
    stem: str
    track_uuid: str
    track_type: str
    has_bleed: Optional[bool]
    wav_rel: str


def _scan_moisesdb_trackinfo(moises_root: Path, song_ids: Optional[List[str]] = None) -> List[TrackInfo]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base}")

    out: List[TrackInfo] = []
    allowed = set(song_ids or [])
    for song_dir in sorted(base.iterdir()):
        if not song_dir.is_dir():
            continue
        if allowed and song_dir.name not in allowed:
            continue
        data_json = song_dir / "data.json"
        if not data_json.exists():
            continue
        try:
            obj = json.loads(data_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        for stem_obj in obj.get("stems") or []:
            if not isinstance(stem_obj, dict):
                continue
            stem_name = str(stem_obj.get("stemName") or "").strip()
            if not stem_name:
                continue
            for track_obj in stem_obj.get("tracks") or []:
                if not isinstance(track_obj, dict):
                    continue
                track_uuid = str(track_obj.get("id") or "").strip()
                track_type = str(track_obj.get("trackType") or "").strip()
                if not track_uuid or not track_type:
                    continue
                has_bleed = track_obj.get("has_bleed")
                if has_bleed is None and "hasBleed" in track_obj:
                    has_bleed = track_obj.get("hasBleed")
                if has_bleed is not None:
                    has_bleed = bool(has_bleed)
                wav_path = song_dir / stem_name / f"{track_uuid}.wav"
                if not wav_path.exists():
                    continue
                out.append(
                    TrackInfo(
                        song_id=song_dir.name,
                        stem=stem_name,
                        track_uuid=track_uuid,
                        track_type=track_type,
                        has_bleed=has_bleed,
                        wav_rel=(Path(MOISESDB_VERSION_DIRNAME) / song_dir.name / stem_name / wav_path.name).as_posix(),
                    )
                )
    return out


def _save_input_segment(
    input_wav: Path,
    out_path: Path,
    offset_sec: float,
    segment_duration: float,
    quiet: bool = False,
) -> Tuple[int, int, float]:
    y, sr = torchaudio.load(str(input_wav))
    length = int(round(segment_duration * sr))
    max_start = max(0, y.shape[1] - length)

    offset_sec_f = float(offset_sec)
    auto_pick = offset_sec_f < 0.0
    start = int(round(max(0.0, offset_sec_f) * sr))
    if start >= y.shape[1]:
        raise ValueError(f"offset_sec={offset_sec} is beyond input audio length for {input_wav}.")

    def _extract_segment(start_sample: int) -> torch.Tensor:
        end = start_sample + length
        seg_ = y[:, start_sample:end]
        if seg_.shape[1] < length:
            seg_ = F.pad(seg_, (0, length - seg_.shape[1]))
        return seg_

    def _rms_mono(seg_: torch.Tensor) -> float:
        mono = seg_.float().mean(dim=0)
        return float(mono.pow(2).mean().sqrt().item())

    seg = _extract_segment(start)
    used_offset_sec = float(start) / float(sr)
    silence_rms_thres = 1e-4
    search_hop_sec = 1.0
    seg_rms = _rms_mono(seg)
    if auto_pick or (seg_rms < silence_rms_thres and max_start > 0):
        mono_full = y.float().mean(dim=0)
        sq = mono_full.pow(2)
        csum = torch.cat([torch.zeros(1, dtype=sq.dtype), torch.cumsum(sq, dim=0)], dim=0)

        hop = max(1, int(round(search_hop_sec * sr)))
        candidates = list(range(0, max_start + 1, hop))
        if candidates and candidates[-1] != max_start:
            candidates.append(max_start)
        if not candidates:
            candidates = [0]

        ms = []
        for s in candidates:
            e = s + length
            ms.append((csum[e] - csum[s]).item() / float(length))
        best_i = int(np.argmax(ms))
        best_start = int(candidates[best_i])
        best_rms = float(math.sqrt(max(0.0, float(ms[best_i]))))
        if best_rms < silence_rms_thres:
            raise ValueError(
                f"Could not find a non-silent {segment_duration:.1f}s window in {input_wav} "
                f"(best_rms={best_rms:.6f} < {silence_rms_thres})."
            )
        if best_start != start or auto_pick:
            used_offset_sec = float(best_start) / float(sr)
            if not quiet:
                print(f"[info] auto-picked non-silent segment in {input_wav}: offset={used_offset_sec:.2f}s")
        start = best_start
        seg = _extract_segment(start)
    elif seg_rms < silence_rms_thres:
        raise ValueError(
            f"Input segment at offset={offset_sec:.2f}s looks silent and no alternative window exists: {input_wav}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), seg, sr)
    return int(sr), int(seg.shape[1]), float(used_offset_sec)


def _drums_rhythm_stats(seg_wav_path: Path, segment_duration: float, sr: int = 22050, hop: int = 512) -> dict:
    try:
        import librosa
    except Exception as e:
        raise SystemExit(f"librosa is required for rhythm stats: {e}")

    y, _ = librosa.load(str(seg_wav_path), sr=sr, mono=True)
    if y.size == 0:
        return {"rms": 0.0, "onset_count": 0, "onset_rate": 0.0}

    target_len = int(round(segment_duration * sr))
    if y.shape[0] < target_len:
        y = librosa.util.fix_length(y, size=target_len)
    else:
        y = y[:target_len]

    rms = float(np.sqrt(np.mean(y.astype(np.float32, copy=False) ** 2)))
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="frames")
    onset_count = int(len(onset_frames))
    onset_rate = float(onset_count) / float(max(1e-8, segment_duration))
    return {"rms": float(rms), "onset_count": int(onset_count), "onset_rate": float(onset_rate)}


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


def _segment_stats_from_mono(mono: torch.Tensor, sr: int, gate: QualityGate) -> dict:
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
    mono = mono.float()
    length = int(round(gate.segment_duration * sr))
    if length <= 0:
        raise ValueError("segment_duration must be > 0")
    if mono.numel() < length:
        mono = F.pad(mono, (0, length - mono.numel()))

    total = int(mono.numel())
    max_start = max(0, total - length)
    hop = max(1, int(round(gate.search_hop_sec * sr)))
    candidates = list(range(0, max_start + 1, hop))
    if candidates and candidates[-1] != max_start:
        candidates.append(max_start)
    if not candidates:
        candidates = [0]

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
    sq = mono.pow(2)
    csum_sq = torch.cat([torch.zeros(1, dtype=sq.dtype), torch.cumsum(sq, dim=0)], dim=0)
    n_frames = int(frame_rms.numel())

    def _frame_index(sample_idx: int) -> int:
        return int(sample_idx // frame_hop)

    best: Optional[dict] = None
    for s in candidates:
        e = int(s + length)
        mean_sq = float((csum_sq[e] - csum_sq[s]).item()) / float(max(1, length))
        win_rms = float(math.sqrt(max(0.0, mean_sq)))

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
        if best is None or (cand["non_silent_ratio"], cand["rms"]) > (best["non_silent_ratio"], best["rms"]):
            best = cand
    return best


def _extract_segment_at_offset(
    wav: torch.Tensor, sr: int, offset_sec: float, segment_duration: float
) -> Tuple[torch.Tensor, float]:
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


def _score_track_best_segment(wav_path: Path, gate: QualityGate) -> Optional[dict]:
    """Returns None if track fails quality gate. Raises on IO errors (let callers decide)."""
    wav, sr = torchaudio.load(str(wav_path))
    mono = wav.float().mean(dim=0)
    best = _pick_best_offset_for_track(mono, int(sr), gate)
    if best is None:
        return None
    if float(best["rms"]) < float(gate.min_rms) or float(best["non_silent_ratio"]) < float(gate.min_non_silent_ratio):
        return None
    return best


def build_melody_index(
    force: bool = False,
    verbose: bool = False,
    index_path: Optional[Path] = None,
    song_ids: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
) -> dict:
    import jasco_ravel_80s as gen

    preferred_stems = ["piano", "guitar", "vocals", "other_keys", "bowed_strings", "wind"]
    segment_duration = 10.0
    frame_rate = 50.0
    melody_bins = 53
    pyin_sr = 22050
    voiced_prob_thres = 0.2
    f0_min_hz = 65.0
    f0_max_hz = 2000.0
    smooth_win = 5
    fill_gaps_frames = 3
    min_anchor_nonzero_ratio_after_fill = 0.90
    selected_song_ids = _resolve_song_subset(MOISESDB_ROOT, song_ids=song_ids, max_songs=max_songs)
    index_path_final = _resolve_index_path(MELODY_INDEX_PATH, index_path)
    checkpoint_path = index_path_final.with_suffix(".ckpt.jsonl")

    config = {
        "dataset": "moisesdb",
        "moisesdb_root": _relpath(MOISESDB_ROOT),
        "song_ids": list(selected_song_ids) if selected_song_ids is not None else None,
        "preferred_stems": preferred_stems,
        "segment_duration": float(segment_duration),
        "frame_rate": float(frame_rate),
        "melody_bins": int(melody_bins),
        "pyin_sr": int(pyin_sr),
        "voiced_prob_thres": float(voiced_prob_thres),
        "f0_min_hz": float(f0_min_hz),
        "f0_max_hz": float(f0_max_hz),
        "smooth_win": int(smooth_win),
        "fill_gaps_frames": int(fill_gaps_frames),
        "min_anchor_nonzero_ratio_after_fill": float(min_anchor_nonzero_ratio_after_fill),
    }
    if not force:
        existing = _load_existing_index(index_path_final, "melody", config)
        if existing is not None:
            if verbose:
                print(f"[melody] reuse existing index: {index_path_final}")
            return existing

    # Load checkpoint for resuming
    done_wav_rels: set = set()
    entries_by_stem: Dict[str, List[dict]] = {stem: [] for stem in preferred_stems}
    if not force and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                stem = str(entry.get("stem", ""))
                if stem in entries_by_stem:
                    entries_by_stem[stem].append(entry)
                done_wav_rels.add(str(entry["wav_rel"]))
            except Exception:
                pass
        loaded = sum(len(v) for v in entries_by_stem.values())
        if verbose:
            print(f"[melody] resuming from checkpoint: {loaded} entries done")
    elif force and checkpoint_path.exists():
        checkpoint_path.unlink()
        entries_by_stem = {stem: [] for stem in preferred_stems}

    moises_tracks = _scan_moisesdb_tracks(MOISESDB_ROOT, preferred_stems, song_ids=selected_song_ids)
    total = len(moises_tracks)
    tmp_dir = TMP_ROOT / "melody"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checked_files = len(done_wav_rels)

    # Build flat list of (track_id, stem, rel_files) tasks for parallel processing
    melody_tasks: List[Tuple[str, str, List[str]]] = []
    for track in moises_tracks:
        track_id = str(track.get("track_id", "")).strip()
        files = track.get("files") or []
        if not track_id or not isinstance(files, list):
            continue
        files_by_stem: Dict[str, List[str]] = {}
        for rel in files:
            if not isinstance(rel, str) or not rel.endswith(".wav"):
                continue
            sname = _stem_from_relpath(rel)
            if sname:
                files_by_stem.setdefault(sname, []).append(rel)
        for stem in preferred_stems:
            rel_files = files_by_stem.get(stem) or []
            if not rel_files:
                continue
            if any(str(r) in done_wav_rels for r in rel_files):
                continue
            melody_tasks.append((track_id, stem, rel_files))

    if verbose:
        print(f"[melody] {len(done_wav_rels)} already done, {len(melody_tasks)} stem-track pairs | {num_workers} workers")

    def _score_melody_stem(task: Tuple[str, str, List[str]]) -> Tuple[str, str, Optional[dict], Optional[str]]:
        t_id, stem, rel_files = task
        tmp_path = tmp_dir / f"melody_{threading.get_ident()}.wav"
        best: Optional[dict] = None
        for rel in rel_files:
            wav_path = MOISESDB_ROOT / rel
            try:
                sr, ns, off = gen._save_input_segment(wav_path, tmp_path, -1.0, segment_duration, quiet=True)
                sal = gen.build_salience_from_wav_pyin(
                    wav_path=tmp_path,
                    segment_duration=segment_duration,
                    frame_rate=frame_rate,
                    melody_bins=melody_bins,
                    melody_sr=pyin_sr,
                    offset_sec=0.0,
                    voiced_prob_thres=voiced_prob_thres,
                    f0_min_hz=f0_min_hz,
                    f0_max_hz=f0_max_hz,
                    smooth_win=smooth_win,
                )
                r0 = float(gen.nonzero_frame_ratio(sal))
                if fill_gaps_frames > 0:
                    sal = gen.fill_short_gaps(sal, fill_gaps_frames)
                r1 = float(gen.nonzero_frame_ratio(sal))
            except Exception as e:
                continue
            cand = {
                "track_id": t_id, "stem": stem, "wav_rel": str(rel),
                "offset_sec": float(off), "sr": int(sr), "ns": int(ns),
                "ratio_before_fill": float(r0), "ratio_after_fill": float(r1),
            }
            if best is None or float(cand["ratio_after_fill"]) > float(best["ratio_after_fill"]):
                best = cand
        return t_id, stem, best, None

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_score_melody_stem, t): t for t in melody_tasks}
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                if verbose and done_count % 50 == 0:
                    total_valid = sum(len(v) for v in entries_by_stem.values())
                    print(f"[melody] scored {done_count}/{len(melody_tasks)} | valid: {total_valid}")
                try:
                    t_id, stem, best, err = fut.result()
                except Exception as exc:
                    print(f"[melody] worker error: {exc}", file=sys.stderr)
                    checked_files += 1
                    continue
                checked_files += 1
                if err or best is None:
                    continue
                if float(best["ratio_after_fill"]) >= float(min_anchor_nonzero_ratio_after_fill):
                    entries_by_stem[stem].append(best)
                    ckpt_f.write(json.dumps(best) + "\n")
                    ckpt_f.flush()

    for stem in preferred_stems:
        entries_by_stem[stem].sort(key=lambda x: (str(x["track_id"]), str(x["wav_rel"])))

    stems_ok_for_b = {
        stem
        for stem, entries in entries_by_stem.items()
        if len({str(x["track_id"]) for x in entries}) >= 2
    }
    per_track_valid: Dict[str, Dict[str, dict]] = {}
    for stem, entries in entries_by_stem.items():
        for entry in entries:
            per_track_valid.setdefault(str(entry["track_id"]), {})[stem] = entry

    anchors: List[dict] = []
    for track in moises_tracks:
        track_id = str(track.get("track_id", "")).strip()
        valid_this_track = per_track_valid.get(track_id) or {}
        for stem in preferred_stems:
            rec = valid_this_track.get(stem)
            if rec is not None and stem in stems_ok_for_b:
                anchors.append(rec)
                break

    data = {
        "schema_version": 1,
        "generator": "melody",
        "built_at": time.time(),
        "config": config,
        "stats": {
            "tracks_total": total,
            "songs_selected": len(selected_song_ids) if selected_song_ids is not None else len(_all_song_ids(MOISESDB_ROOT)),
            "checked_files": int(checked_files),
            "anchors_available": len(anchors),
            "entries_per_stem": {stem: len(entries_by_stem.get(stem) or []) for stem in preferred_stems},
        },
        "anchors": anchors,
        "entries_by_stem": entries_by_stem,
    }
    _save_json(index_path_final, data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if verbose:
        print(f"[melody] done: anchors={len(anchors)} checked={checked_files} -> {index_path_final}")
    return data


def build_rhythm_index(
    force: bool = False,
    verbose: bool = False,
    index_path: Optional[Path] = None,
    song_ids: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
) -> dict:
    segment_duration = 10.0
    onset_sr = 22050
    onset_hop = 512
    min_anchor_onset_count = 20
    min_anchor_rms = 0.005
    selected_song_ids = _resolve_song_subset(MOISESDB_ROOT, song_ids=song_ids, max_songs=max_songs)
    index_path_final = _resolve_index_path(RHYTHM_INDEX_PATH, index_path)
    checkpoint_path = index_path_final.with_suffix(".ckpt.jsonl")
    config = {
        "dataset": "moisesdb",
        "moisesdb_root": _relpath(MOISESDB_ROOT),
        "song_ids": list(selected_song_ids) if selected_song_ids is not None else None,
        "stem": "drums",
        "segment_duration": float(segment_duration),
        "onset_sr": int(onset_sr),
        "onset_hop": int(onset_hop),
        "min_anchor_onset_count": int(min_anchor_onset_count),
        "min_anchor_rms": float(min_anchor_rms),
    }
    if not force:
        existing = _load_existing_index(index_path_final, "rhythm", config)
        if existing is not None:
            if verbose:
                print(f"[rhythm] reuse existing index: {index_path_final}")
            return existing

    # Load checkpoint for resuming
    done_track_ids: set = set()
    entries: List[dict] = []
    if not force and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
                done_track_ids.add(str(entry["track_id"]))
            except Exception:
                pass
        if verbose:
            print(f"[rhythm] resuming from checkpoint: {len(entries)} tracks done")
    elif force and checkpoint_path.exists():
        checkpoint_path.unlink()

    moises_tracks = _scan_moisesdb_drums_tracks(MOISESDB_ROOT, song_ids=selected_song_ids)
    total = len(moises_tracks)
    tmp_dir = TMP_ROOT / "rhythm"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checked_files = len(done_track_ids)

    pending_tracks = [
        t for t in moises_tracks
        if str(t.get("track_id", "")).strip() and str(t.get("track_id", "")).strip() not in done_track_ids
    ]
    if verbose:
        print(f"[rhythm] {len(done_track_ids)} already done, {len(pending_tracks)} to score | {num_workers} workers")

    def _score_rhythm_track(track: dict) -> Tuple[str, Optional[dict], Optional[str]]:
        track_id = str(track.get("track_id", "")).strip()
        files = list(track.get("files") or [])
        tmp_path = tmp_dir / f"rhythm_{threading.get_ident()}.wav"
        best: Optional[dict] = None
        for rel in files:
            wav_path = MOISESDB_ROOT / str(rel)
            try:
                sr, ns, off = _save_input_segment(wav_path, tmp_path, -1.0, segment_duration, quiet=True)
                st = _drums_rhythm_stats(tmp_path, segment_duration, sr=onset_sr, hop=onset_hop)
            except Exception:
                continue
            cand = {
                "track_id": track_id, "stem": "drums", "wav_rel": str(rel),
                "offset_sec": float(off), "sr": int(sr), "ns": int(ns),
                "rms": float(st["rms"]), "onset_count": int(st["onset_count"]), "onset_rate": float(st["onset_rate"]),
            }
            if best is None or (cand["onset_count"], cand["rms"]) > (best["onset_count"], best["rms"]):
                best = cand
        return track_id, best, None

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_score_rhythm_track, t): t for t in pending_tracks}
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                if verbose and done_count % 25 == 0:
                    print(f"[rhythm] scored {done_count}/{len(pending_tracks)} | valid: {len(entries)}")
                try:
                    track_id, best, err = fut.result()
                except Exception as exc:
                    print(f"[rhythm] worker error: {exc}", file=sys.stderr)
                    checked_files += 1
                    continue
                checked_files += 1
                if err or best is None:
                    continue
                if int(best["onset_count"]) < int(min_anchor_onset_count) or float(best["rms"]) < float(min_anchor_rms):
                    continue
                entries.append(best)
                ckpt_f.write(json.dumps(best) + "\n")
                ckpt_f.flush()

    entries.sort(key=lambda x: (str(x["track_id"]), str(x["wav_rel"])))
    data = {
        "schema_version": 1,
        "generator": "rhythm",
        "built_at": time.time(),
        "config": config,
        "stats": {
            "tracks_total": total,
            "songs_selected": len(selected_song_ids) if selected_song_ids is not None else len(_all_song_ids(MOISESDB_ROOT)),
            "checked_files": int(checked_files),
            "entries_available": len(entries),
        },
        "anchors": entries,
        "entries": entries,
    }
    _save_json(index_path_final, data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if verbose:
        print(f"[rhythm] done: entries={len(entries)} checked={checked_files} -> {index_path_final}")
    return data


def build_timbre_index(
    force: bool = False,
    verbose: bool = False,
    index_path: Optional[Path] = None,
    song_ids: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
) -> dict:
    gate = QualityGate()
    selected_song_ids = _resolve_song_subset(MOISESDB_ROOT, song_ids=song_ids, max_songs=max_songs)
    index_path_final = _resolve_index_path(TIMBRE_INDEX_PATH, index_path)
    checkpoint_path = index_path_final.with_suffix(".ckpt.jsonl")
    config = {
        "dataset": "moisesdb",
        "moisesdb_root": _relpath(MOISESDB_ROOT),
        "song_ids": list(selected_song_ids) if selected_song_ids is not None else None,
        "quality_gate": asdict(gate),
    }
    if not force:
        existing = _load_existing_index(index_path_final, "timbre", config)
        if existing is not None:
            if verbose:
                print(f"[timbre] reuse existing index: {index_path_final}")
            return existing

    # Load checkpoint for resuming
    done_wav_rels: set = set()
    valid_entries: List[dict] = []
    if not force and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                valid_entries.append(entry)
                done_wav_rels.add(str(entry["wav_rel"]))
            except Exception:
                pass
        if verbose:
            print(f"[timbre] resuming from checkpoint: {len(valid_entries)} entries done")
    elif force and checkpoint_path.exists():
        checkpoint_path.unlink()

    tracks = _scan_moisesdb_trackinfo(MOISESDB_ROOT, song_ids=selected_song_ids)
    sorted_tracks = sorted(tracks, key=lambda x: (x.song_id, x.stem, x.track_type, x.track_uuid))
    total = len(sorted_tracks)
    checked_tracks = len(done_wav_rels)

    pending = [info for info in sorted_tracks if str(info.wav_rel) not in done_wav_rels]
    if verbose:
        print(f"[timbre] {len(done_wav_rels)} already done, {len(pending)} to score | {num_workers} workers")

    def _score_one(info: TrackInfo) -> Tuple[TrackInfo, Optional[dict], Optional[str]]:
        wav_path = MOISESDB_ROOT / info.wav_rel
        try:
            return info, _score_track_best_segment(wav_path, gate), None
        except Exception as e:
            return info, None, f"{type(e).__name__}: {e}"

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_score_one, info): info for info in pending}
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                if verbose and done_count % 200 == 0:
                    print(f"[timbre] scored {done_count}/{len(pending)} | valid: {len(valid_entries)}")
                try:
                    info, best, err = fut.result()
                except Exception as exc:
                    print(f"[timbre] unexpected worker error: {exc}", file=sys.stderr)
                    checked_tracks += 1
                    continue
                checked_tracks += 1
                if err:
                    print(f"[timbre] error {info.wav_rel}: {err}", file=sys.stderr)
                    continue
                if best is None:
                    continue
                entry = {
                    "song_id": info.song_id,
                    "stem": info.stem,
                    "track_uuid": info.track_uuid,
                    "trackType": info.track_type,
                    "has_bleed": info.has_bleed,
                    "wav_rel": info.wav_rel,
                    "quality_best_window": best,
                }
                valid_entries.append(entry)
                ckpt_f.write(json.dumps(entry) + "\n")
                ckpt_f.flush()

    if verbose:
        print(f"[timbre] scoring done: {len(valid_entries)}/{total} valid. Building negatives...")

    valid_entries.sort(key=lambda x: (str(x["song_id"]), str(x["stem"]), str(x["trackType"]), str(x["track_uuid"])))
    entries_by_song: Dict[str, List[dict]] = {}
    type_song_counts: Dict[str, set] = {}
    for entry in valid_entries:
        entries_by_song.setdefault(str(entry["song_id"]), []).append(entry)
        type_song_counts.setdefault(str(entry["trackType"]), set()).add(str(entry["song_id"]))
    allowed_track_types = sorted([tt for tt, songs in type_song_counts.items() if len(songs) >= 2])
    allowed_set = set(allowed_track_types)

    anchors: List[dict] = []
    all_song_ids_sorted = sorted(entries_by_song)
    for song_idx, song_id in enumerate(all_song_ids_sorted, start=1):
        if verbose and song_idx % 25 == 0:
            print(f"[timbre] negatives {song_idx}/{len(all_song_ids_sorted)} songs | anchors: {len(anchors)}")
        song_entries = entries_by_song[song_id]
        wav_cache: Dict[str, Tuple[torch.Tensor, int]] = {}
        for entry in song_entries:
            try:
                wav, sr = torchaudio.load(str(MOISESDB_ROOT / str(entry["wav_rel"])))
                wav_cache[str(entry["track_uuid"])] = (wav, int(sr))
            except Exception as e:
                print(f"[timbre] cache load error {entry['wav_rel']}: {e}", file=sys.stderr)

        for anchor in song_entries:
            if str(anchor["trackType"]) not in allowed_set:
                continue
            negative_candidates: List[dict] = []
            anchor_offset = float(anchor["quality_best_window"]["offset_sec"])
            for cand in song_entries:
                if str(cand["track_uuid"]) == str(anchor["track_uuid"]):
                    continue
                if str(cand["stem"]) == str(anchor["stem"]):
                    continue
                if str(cand["trackType"]) == str(anchor["trackType"]):
                    continue
                pair = wav_cache.get(str(cand["track_uuid"]))
                if pair is None:
                    continue
                wav, sr = pair
                try:
                    seg, used_off = _extract_segment_at_offset(wav, int(sr), anchor_offset, gate.segment_duration)
                except Exception:
                    continue
                mono = seg.float().mean(dim=0)
                st = _segment_stats_from_mono(mono, int(sr), gate)
                st["offset_sec"] = float(used_off)
                if float(st["rms"]) < float(gate.min_rms) or float(st["non_silent_ratio"]) < float(gate.min_non_silent_ratio):
                    continue
                negative_candidates.append({
                    "song_id": cand["song_id"],
                    "stem": cand["stem"],
                    "track_uuid": cand["track_uuid"],
                    "trackType": cand["trackType"],
                    "has_bleed": cand["has_bleed"],
                    "wav_rel": cand["wav_rel"],
                    "offset_sec": float(used_off),
                    "quality_at_anchor_offset": st,
                })
            negative_candidates.sort(
                key=lambda x: (float(x["quality_at_anchor_offset"]["non_silent_ratio"]), float(x["quality_at_anchor_offset"]["rms"])),
                reverse=True,
            )
            if not negative_candidates:
                continue
            anchors.append({"anchor": anchor, "negative_candidates": negative_candidates})

    data = {
        "schema_version": 1,
        "generator": "timbre",
        "built_at": time.time(),
        "config": config,
        "stats": {
            "tracks_total": total,
            "songs_selected": len(selected_song_ids) if selected_song_ids is not None else len(_all_song_ids(MOISESDB_ROOT)),
            "checked_tracks": int(checked_tracks),
            "valid_entries": len(valid_entries),
            "anchors_available": len(anchors),
            "allowed_track_types": len(allowed_track_types),
        },
        "allowed_track_types": allowed_track_types,
        "entries": valid_entries,
        "anchors": anchors,
    }
    _save_json(index_path_final, data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if verbose:
        print(f"[timbre] done: valid_entries={len(valid_entries)} anchors={len(anchors)} -> {index_path_final}")
    return data
