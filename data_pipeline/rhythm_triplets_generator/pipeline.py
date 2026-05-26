#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triplets_input_index.index_builder import build_rhythm_index

# -------------------------------------------------
# JASCO / audiocraft path bootstrap
# -------------------------------------------------
_JASCO_ROOT_ENV = os.environ.get("JASCO_ROOT", "").strip()
# PROJECT_ROOT = root of the cloned JASCO audiocraft repo (contains audiocraft/ and assets/)
PROJECT_ROOT: Path | None = Path(_JASCO_ROOT_ENV) if _JASCO_ROOT_ENV else None
AUDIOCRAFT_REPO_ROOT: Path | None = (PROJECT_ROOT / "audiocraft") if PROJECT_ROOT else None

if PROJECT_ROOT is not None:
    sys.path.insert(0, str(AUDIOCRAFT_REPO_ROOT))
    sys.path.insert(1, str(PROJECT_ROOT))


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
ANCHOR_QUALITY_CACHE_PATH = CACHE_DIR / "anchor_quality_cache.json"

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

def _resolve_root_from_config(raw: object, fallback: Path) -> Path:
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip()).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p
    return fallback.resolve()

def _resolve_index_audio_path(entry: dict, audio_root: Path) -> Path:
    rel = str(entry.get("wav_rel", "")).strip()
    if not rel:
        raise SystemExit(f"Invalid rhythm index entry without wav_rel: {entry}")
    p = Path(rel)
    if p.is_absolute():
        return p
    return audio_root / p

def _resolve_index_drums_path(entry: dict, audio_root: Path, drums_root: Path) -> Path:
    rel = str(entry.get("drums_wav_rel", "")).strip()
    if rel:
        p = Path(rel)
        if p.is_absolute():
            return p
        return drums_root / p
    return _resolve_index_audio_path(entry, audio_root)

def _next_triplets_index(out_base: Path) -> int:
    if not out_base.exists():
        return 1
    best = 0
    for p in out_base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("triplets_"):
            continue
        tail = name[len("triplets_") :]
        if not tail.isdigit():
            continue
        best = max(best, int(tail))
    return best + 1

def _collect_existing_anchor_track_ids(out_base: Path) -> set[str]:
    used: set[str] = set()
    if not out_base.exists():
        return used
    for run_dir in out_base.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("triplets_"):
            continue
        meta_paths = [
            run_dir / "triplet" / "triplet_meta.json",
            run_dir / "triplet_0001" / "triplet_meta.json",
        ]
        meta_path = next((p for p in meta_paths if p.exists()), None)
        if meta_path is None:
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        track_id = data.get("inputs", {}).get("A_track_id")
        if isinstance(track_id, str) and track_id.strip():
            used.add(track_id.strip())
    return used

def _runtime_diag() -> None:
    try:
        import transformers
    except Exception as e:
        raise SystemExit(f"Missing dependency: transformers ({e}). Please run with the project venv.")

    print(f"python = {sys.executable}")
    print(f"torch = {torch.__version__}")
    print(f"transformers = {getattr(transformers, '__version__', 'unknown')}")

    is_torch_available = None
    fn = getattr(transformers, "is_torch_available", None)
    if callable(fn):
        is_torch_available = fn
    if callable(is_torch_available) and not bool(is_torch_available()):
        raise SystemExit(
            "Environment error: transformers reports torch backend is unavailable.\n"
            "Ensure you are running in the correct conda/venv environment with "
            "matching torch and transformers versions. See README.md for setup instructions."
        )

def _load_positive_prompts(positive_prompts_file: Path) -> List[str]:
    if not positive_prompts_file.exists():
        raise SystemExit(f"prompts.txt not found: {positive_prompts_file}")
    lines = positive_prompts_file.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if not out:
        raise SystemExit(f"prompts.txt is empty: {positive_prompts_file}")
    return out

def _scan_moisesdb_drums_tracks(moises_root: Path) -> list[dict]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base} (expected extracted DB at {moises_root})")

    out: list[dict] = []
    for track_dir in sorted(base.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        stem_dir = track_dir / "drums"
        if not stem_dir.is_dir():
            continue
        files: list[str] = []
        for wav in sorted(stem_dir.glob("*.wav")):
            rel = (Path(MOISESDB_VERSION_DIRNAME) / track_id / "drums" / wav.name).as_posix()
            files.append(rel)
        if files:
            out.append({"track_id": track_id, "stem": "drums", "files": files})
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
            seg_ = torch.nn.functional.pad(seg_, (0, length - seg_.shape[1]))
        return seg_

    def _rms_mono(seg_: torch.Tensor) -> float:
        mono = seg_.float().mean(dim=0)
        return float(mono.pow(2).mean().sqrt().item())

    seg = _extract_segment(start)
    used_offset_sec = float(start) / float(sr)

    SILENCE_RMS_THRES = 1e-4
    SEARCH_HOP_SEC = 1.0
    seg_rms = _rms_mono(seg)
    if auto_pick or (seg_rms < SILENCE_RMS_THRES and max_start > 0):
        mono_full = y.float().mean(dim=0)
        sq = mono_full.pow(2)
        csum = torch.cat([torch.zeros(1, dtype=sq.dtype), torch.cumsum(sq, dim=0)], dim=0)

        hop = max(1, int(round(SEARCH_HOP_SEC * sr)))
        candidates = list(range(0, max_start + 1, hop))
        if candidates[-1] != max_start:
            candidates.append(max_start)

        ms = []
        for s in candidates:
            e = s + length
            ms.append((csum[e] - csum[s]).item() / float(length))
        best_i = int(np.argmax(ms))
        best_start = int(candidates[best_i])
        best_rms = float(math.sqrt(max(0.0, float(ms[best_i]))))
        if best_rms < SILENCE_RMS_THRES:
            raise ValueError(
                f"Could not find a non-silent {segment_duration:.1f}s window in {input_wav} "
                f"(best_rms={best_rms:.6f} < {SILENCE_RMS_THRES})."
            )
        if best_start != start or auto_pick:
            used_offset_sec = float(best_start) / float(sr)
            level = "INFO" if auto_pick else "WARN"
            if not quiet:
                print(
                    f"[{level}] Auto-picked non-silent segment in {input_wav}: "
                    f"offset={used_offset_sec:.2f}s (rms={best_rms:.6f})."
                )
        start = best_start
        seg = _extract_segment(start)
    elif seg_rms < SILENCE_RMS_THRES:
        raise ValueError(
            f"Input segment at offset={offset_sec:.2f}s looks silent and no alternative window exists: "
            f"{input_wav} (rms={seg_rms:.6f} < {SILENCE_RMS_THRES})."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), seg, sr)
    return sr, seg.shape[1], used_offset_sec

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

def _load_anchor_quality_cache(path: Path, expected_config: dict) -> dict:
    if not path.exists():
        return {"schema_version": 1, "config": expected_config, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": 1, "config": expected_config, "entries": {}}

    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        return {"schema_version": 1, "config": expected_config, "entries": {}}

    cfg = data.get("config")
    entries = data.get("entries")
    if cfg != expected_config:
        print(f"[WARN] Anchor quality cache config changed; ignoring old cache: {path}")
        return {"schema_version": 1, "config": expected_config, "entries": {}}
    if not isinstance(entries, dict):
        entries = {}
    return {"schema_version": 1, "config": expected_config, "entries": entries}

def _save_anchor_quality_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def _pick_device(mode: str = "auto") -> str:
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _set_seed(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

def _filter_supported_generate_kwargs(model, kwargs: Dict) -> Tuple[Dict, List[str]]:
    sig = inspect.signature(model.lm.generate)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    filtered: Dict = {}
    skipped: List[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if k in allowed:
            filtered[k] = v
        else:
            skipped.append(k)
    return filtered, skipped

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate N rhythm triplets (pipeline).")
    ap.add_argument("num_triplets", type=int, help="How many triplets to generate.")
    ap.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="Root output directory where triplet folders will be written.",
    )
    ap.add_argument(
        "--index", type=Path, default=None,
        help="Path to pre-built rhythm_index.json. Overrides env var and auto-build.",
    )
    ap.add_argument(
        "--num-positives", type=int, default=5,
        help="Number of positive samples per triplet (default: 5).",
    )
    ap.add_argument(
        "--prompts-file", type=Path, default=None,
        help="Prompts file. Defaults to melody's prompts_5000.txt then local prompts.txt.",
    )
    ap.add_argument(
        "--allow-anchor-reuse", action="store_true",
        help="Cycle through anchors when num_triplets > unique anchors available.",
    )
    ap.add_argument(
        "--model", type=str,
        default="facebook/jasco-chords-drums-melody-1B",
        help="JASCO model ID (default: facebook/jasco-chords-drums-melody-1B).",
    )
    ap.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index (sets CUDA_VISIBLE_DEVICES before any CUDA init).",
    )
    ap.add_argument(
        "--start-at", type=int, default=None,
        help="Override starting triplet number (for splitting work across GPUs).",
    )
    args = ap.parse_args()

    if args.num_triplets <= 0:
        raise SystemExit("num_triplets must be > 0")

    # Set GPU BEFORE any CUDA / torch init
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")

    _runtime_diag()

    model_id = args.model
    segment_duration = 10.0
    frame_rate = 50.0
    melody_bins = 53

    onset_sr = 22050
    onset_hop = 512
    min_anchor_onset_count = 20
    min_anchor_rms = 0.005

    cfg_all = 1.25
    cfg_txt = 2.5
    ode_rtol = None
    ode_atol = None
    euler = False
    euler_steps = None

    base_seed = 1234
    positive_seed_mode = "same"

    print("Loading prefiltered rhythm index...")
    if args.index is not None:
        if not args.index.exists():
            raise SystemExit(f"--index path not found: {args.index}")
        rhythm_index = json.loads(args.index.read_text(encoding="utf-8"))
        if str(rhythm_index.get("generator", "")).strip() != "rhythm":
            raise SystemExit(f"--index JSON is not a rhythm index (generator != 'rhythm'): {args.index}")
        print(f"Using rhythm index from --index: {args.index}")
    else:
        rhythm_index = _load_index_from_env("TRIPLETS_RHYTHM_INDEX_JSON", "rhythm")
        if rhythm_index is None:
            if not MOISESDB_ROOT.exists():
                raise SystemExit(
                    f"MoisesDB not found: {MOISESDB_ROOT}\n"
                    "Set --index to a pre-built rhythm_index.json or rebuild via build_index.py rhythm"
                )
            rhythm_index = build_rhythm_index(force=False, verbose=True)
    index_config = dict(rhythm_index.get("config") or {})
    dataset_name = str(index_config.get("dataset") or "moisesdb")
    audio_root = _resolve_root_from_config(
        index_config.get("audio_root") or index_config.get("moisesdb_root"),
        MOISESDB_ROOT,
    )
    drums_cache_root = _resolve_root_from_config(index_config.get("drums_cache_root"), audio_root)
    source_type = str(index_config.get("source_type") or ("drums_stem" if dataset_name == "moisesdb" else "fullmix"))
    candidates = list(rhythm_index.get("entries") or [])
    if len(candidates) < 2:
        raise SystemExit(
            "Rhythm index does not contain enough valid entries. "
            f"Please rebuild it: python {REPO_ROOT / 'triplets_input_index' / 'build_index.py'} rhythm --force"
        )
    print(
        f"Rhythm dataset: {dataset_name} "
        f"(audio_root={audio_root}, drums_root={drums_cache_root}, source_type={source_type}, "
        f"prefiltered drums_entries={len(candidates)})"
    )

    out_base = args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)
    if args.start_at is not None:
        if args.start_at < 1:
            raise SystemExit("--start-at must be >= 1")
        start_idx = args.start_at
    else:
        start_idx = _next_triplets_index(out_base)
    end_idx = start_idx + args.num_triplets - 1
    print(f"Output base: {out_base}")
    print(f"Generating triplets_{start_idx} .. triplets_{end_idx}")

    used_a_track_ids = _collect_existing_anchor_track_ids(out_base)
    if used_a_track_ids:
        print(f"Found {len(used_a_track_ids)} existing anchors; will avoid reusing their track_id for A.")

    # Resolve prompts file: prefer melody's prompts_5000.txt for richer diversity
    if args.prompts_file is not None:
        _prompts_path = args.prompts_file
    else:
        _melody_prompts = REPO_ROOT / "melody_triplets_generator" / "prompts_5000.txt"
        _local_prompts = SCRIPT_DIR / "prompts.txt"
        _prompts_path = _melody_prompts if _melody_prompts.exists() else _local_prompts
    all_prompts = _load_positive_prompts(_prompts_path)
    if len(all_prompts) < args.num_positives + 1:
        raise SystemExit(
            f"Need at least num_positives+1={args.num_positives + 1} prompts "
            f"in {_prompts_path} (got {len(all_prompts)})"
        )
    print(f"Loaded {len(all_prompts)} prompts from {_prompts_path}")

    anchor_plan: list[dict] = []
    selected_track_ids: set[str] = set()
    scanned_tracks = 0

    for rec in list(rhythm_index.get("anchors") or []):
        track_id = str(rec.get("track_id", "")).strip()
        if not track_id:
            continue
        if track_id in used_a_track_ids or track_id in selected_track_ids:
            continue

        scanned_tracks += 1
        b_choices = [x for x in candidates if str(x.get("track_id", "")) != track_id]
        if not b_choices:
            continue
        anchor_plan.append(rec)
        selected_track_ids.add(track_id)
        print(
            f"[anchor] selected (index): track={track_id} onset_count={int(rec.get('onset_count', 0))} "
            f"rms={float(rec.get('rms', 0.0)):.4f} wav={rec.get('wav_rel')}"
        )

        if len(anchor_plan) >= args.num_triplets:
            break

    if len(anchor_plan) < args.num_triplets:
        if not args.allow_anchor_reuse:
            raise SystemExit(
                f"Not enough unique anchors (have {len(anchor_plan)}, need {args.num_triplets}).\n"
                "Pass --allow-anchor-reuse to cycle anchors with different random prompts per triplet.\n"
                f"(scanned={scanned_tracks}, selected={len(anchor_plan)})"
            )
        # Cycle the valid pool
        valid_pool = [
            rec for rec in list(rhythm_index.get("anchors") or [])
            if str(rec.get("track_id", "")).strip()
            and any(str(x.get("track_id", "")) != str(rec.get("track_id", "")) for x in candidates)
        ]
        if not valid_pool:
            raise SystemExit("No valid rhythm anchors with >=2 tracks found in index.")
        while len(anchor_plan) < args.num_triplets:
            anchor_plan.append(valid_pool[len(anchor_plan) % len(valid_pool)])
        print(
            f"[anchor-reuse] Cycled {len(valid_pool)} valid anchors to fill {len(anchor_plan)} slots. "
            "Each reuse will use different random prompts."
        )

    print(f"Selected anchors: drums={len(anchor_plan)}")

    if PROJECT_ROOT is None:
        raise SystemExit(
            "Error: JASCO_ROOT environment variable is not set.\n"
            "Set it to the root of the cloned JASCO audiocraft repository:\n"
            "  export JASCO_ROOT=/path/to/jasco-audiocraft\n"
            "See README.md for installation instructions."
        )

    from audiocraft.models import JASCO
    from audiocraft.data.audio import audio_write
    from audiocraft.data.audio_utils import normalize_audio

    chord_map = Path(PROJECT_ROOT) / "assets" / "chord_to_index_mapping.pkl"
    if not chord_map.exists():
        raise SystemExit(f"Chord map not found: {chord_map}")

    device = _pick_device("auto")
    print("device =", device)
    if device == "cuda":
        print("cuda device:", torch.cuda.get_device_name(0))

    print("Loading JASCO model...")
    model = JASCO.get_pretrained(model_id, device=device, chords_mapping_path=str(chord_map))

    gen_kwargs = {
        "cfg_coef_all": float(cfg_all),
        "cfg_coef_txt": float(cfg_txt),
        "ode_rtol": ode_rtol,
        "ode_atol": ode_atol,
        "euler": bool(euler),
        "euler_steps": euler_steps,
    }
    filtered, skipped = _filter_supported_generate_kwargs(model, gen_kwargs)
    model.set_generation_params(**filtered)
    print("Applied generation params:", filtered)
    if skipped:
        print("Skipped unsupported generation params:", skipped)

    def _append_manifest(manifest_path: Path, obj: dict) -> None:
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _write_segment_as(triplet_dir: Path, name: str, seg_wav_path: Path) -> str:
        wav, sr = torchaudio.load(str(seg_wav_path))
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(model.sample_rate):
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(model.sample_rate))
        wav = wav.squeeze(0).contiguous()
        audio_write(
            str(triplet_dir / name),
            wav,
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
            add_suffix=True,
        )
        return _relpath(triplet_dir / f"{name}.wav")

    def _generate_one(prompt: str, drums_wav: torch.Tensor, drums_sr: int, seed: int) -> Tuple[torch.Tensor, dict]:
        _set_seed(seed, device)
        stats = {"seed": int(seed), "device": device}

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        wav = model.generate_music(
            descriptions=[prompt],
            chords=None,
            drums_wav=drums_wav,
            drums_sample_rate=int(drums_sr),
            melody_salience_matrix=None,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            progress=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        stats["duration_sec"] = float(t1 - t0)
        if device == "cuda":
            stats["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        return wav, stats

    for plan_idx, trip_idx in enumerate(range(start_idx, end_idx + 1), start=0):
        run_dir = out_base / f"triplets_{trip_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        triplet_dir = run_dir / "triplet"
        triplet_dir.mkdir(parents=True, exist_ok=True)

        run_seed = base_seed + int(trip_idx)
        rng = np.random.default_rng(run_seed)

        # Per-triplet random prompt selection for diversity across cycles
        _n_pick = args.num_positives + 1
        _prompt_idxs = rng.choice(len(all_prompts), size=min(_n_pick, len(all_prompts)), replace=False)
        _anchor_prompt = all_prompts[int(_prompt_idxs[0])]
        positive_prompts = [all_prompts[int(i)] for i in _prompt_idxs[1:]]

        a_plan = anchor_plan[plan_idx]

        a_seg_path = run_dir / "input_A_segment.wav"
        b_seg_path = run_dir / "input_B_segment.wav"
        a_drums_seg_path = run_dir / "input_A_drums_segment.wav"

        input_a = _resolve_index_audio_path(a_plan, audio_root)
        input_a_drums = _resolve_index_drums_path(a_plan, audio_root, drums_cache_root)
        a_track_id = str(a_plan["track_id"])
        used_offset_a = float(a_plan["offset_sec"])
        sr_a, ns_a, used_offset_a = _save_input_segment(input_a, a_seg_path, used_offset_a, segment_duration)
        if input_a_drums.resolve() != input_a.resolve() or source_type == "fullmix":
            sr_a_drums, ns_a_drums, used_offset_a_drums = _save_input_segment(
                input_a_drums, a_drums_seg_path, used_offset_a, segment_duration
            )
            drums_segment_for_condition = a_drums_seg_path
        else:
            sr_a_drums, ns_a_drums, used_offset_a_drums = sr_a, ns_a, used_offset_a
            drums_segment_for_condition = a_seg_path

        b_choices = [t for t in candidates if str(t.get("track_id", "")) != a_track_id]
        if not b_choices:
            raise SystemExit("Need >=2 tracks to pick B != A, but only found 1.")
        b = b_choices[int(rng.integers(0, len(b_choices)))]
        input_b = _resolve_index_audio_path(b, audio_root)
        b_track_id = str(b["track_id"])
        sr_b, ns_b, used_offset_b = _save_input_segment(
            input_b, b_seg_path, float(b.get("offset_sec", -1.0)), segment_duration
        )

        print(
            f"[triplets_{trip_idx}] inputs: "
            f"A(track={a_track_id}, wav={_relpath(input_a)}, sr={sr_a}, samples={ns_a}, offset={used_offset_a:.2f}s, onset_count={a_plan['onset_count']}, rms={a_plan['rms']:.4f}, drums_cond={_relpath(input_a_drums)}) "
            f"B(track={b_track_id}, wav={_relpath(input_b)}, sr={sr_b}, samples={ns_b}, offset={used_offset_b:.2f}s)"
        )

        drums, drums_sr = torchaudio.load(str(drums_segment_for_condition))
        if drums.ndim == 2 and drums.shape[0] > 1:
            drums = drums.mean(dim=0, keepdim=True)
        drums = normalize_audio(drums, strategy="loudness", loudness_headroom_db=16, sample_rate=int(drums_sr))
        drums_cond = drums

        manifest_path = run_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        anchor_path = _write_segment_as(triplet_dir, "anchor", a_seg_path)
        _append_manifest(
            manifest_path,
            {"type": "anchor", "output_wav": anchor_path, "prompt": None, "seed": None, "stats": {"source": "input_A_segment"}},
        )

        negative_path = _write_segment_as(triplet_dir, "negative", b_seg_path)
        _append_manifest(
            manifest_path,
            {"type": "negative", "output_wav": negative_path, "prompt": None, "seed": None, "stats": {"source": "input_B_segment"}},
        )

        positives_info: list[dict] = []
        # Generate all positives in a single batched forward pass
        pos_seed = run_seed
        _set_seed(pos_seed, device)
        batch_stats: dict = {"seed": int(pos_seed), "device": device, "batch_size": len(positive_prompts)}
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs_pos = model.generate_music(
            descriptions=positive_prompts,
            chords=None,
            drums_wav=drums_cond,
            drums_sample_rate=int(drums_sr),
            melody_salience_matrix=None,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            progress=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        batch_stats["duration_sec"] = float(time.perf_counter() - t0)
        if device == "cuda":
            batch_stats["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        wavs_pos = wavs_pos.detach().cpu()

        for k, pos_prompt in enumerate(positive_prompts, start=1):
            wav_pos = wavs_pos[k - 1]  # [C, T]
            name = f"positive_{k:02d}"
            audio_write(
                str(triplet_dir / name),
                wav_pos,
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
                add_suffix=True,
            )
            out_path = _relpath(triplet_dir / f"{name}.wav")
            rec = {
                "type": "positive",
                "positive_index": k,
                "output_wav": out_path,
                "prompt": pos_prompt,
                "seed": int(pos_seed),
                "stats": batch_stats,
            }
            positives_info.append(rec)
            _append_manifest(manifest_path, rec)

        run_meta = {
            "model_id": model_id,
            "device": device,
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "segment_duration": segment_duration,
            "cfg_all": cfg_all,
            "cfg_txt": cfg_txt,
            "ode_rtol": ode_rtol,
            "ode_atol": ode_atol,
            "euler": euler,
            "euler_steps": euler_steps,
            "seed": int(run_seed),
            "positive_seed_mode": positive_seed_mode,
            "inputs": {
                "dataset": dataset_name,
                "audio_root": _relpath(audio_root),
                "drums_cache_root": _relpath(drums_cache_root),
                "source_type": source_type,
                "stem": "drums" if source_type == "drums_stem" else "fullmix",
                "A_track_id": a_track_id,
                "B_track_id": b_track_id,
                "A_wav": _relpath(input_a),
                "B_wav": _relpath(input_b),
                "A_drums_condition_wav": _relpath(input_a_drums),
                "A_offset_sec": float(used_offset_a),
                "B_offset_sec": float(used_offset_b),
                "A_segment_saved": _relpath(a_seg_path),
                "B_segment_saved": _relpath(b_seg_path),
                "A_drums_segment_saved": _relpath(drums_segment_for_condition),
                "anchor_quality_gate": {
                    "min_anchor_onset_count": int(min_anchor_onset_count),
                    "min_anchor_rms": float(min_anchor_rms),
                    "A_onset_count": int(a_plan["onset_count"]),
                    "A_onset_rate": float(a_plan["onset_rate"]),
                    "A_rms": float(a_plan["rms"]),
                },
                "drums_conditioning": {
                    "source": "input_A_segment" if drums_segment_for_condition == a_seg_path else "separated_drums_from_mix",
                    "drums_sample_rate": int(drums_sr),
                    "drums_segment_ns": int(ns_a_drums),
                    "drums_segment_offset_sec": float(used_offset_a_drums),
                    "normalize": {"strategy": "loudness", "loudness_headroom_db": 16},
                },
            },
            "prompts": {"anchor_prompt": _anchor_prompt, "positive_prompts": positive_prompts},
            "note": (
                "Rhythm triplets: Anchor/Negative are input segments from the selected dataset. "
                "If the input is full-mix, JASCO is conditioned on a Demucs-separated drums stem from input A "
                "(melody disabled)."
            ),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        triplet_meta = {
            "triplet_index": 1,
            "triplet_dir": _relpath(triplet_dir),
            "inputs": {
                "dataset": dataset_name,
                "audio_root": _relpath(audio_root),
                "drums_cache_root": _relpath(drums_cache_root),
                "source_type": source_type,
                "stem": "drums" if source_type == "drums_stem" else "fullmix",
                "A_track_id": a_track_id,
                "B_track_id": b_track_id,
                "A_wav": _relpath(input_a),
                "B_wav": _relpath(input_b),
                "A_drums_condition_wav": _relpath(input_a_drums),
                "A_offset_sec": float(used_offset_a),
                "B_offset_sec": float(used_offset_b),
                "A_segment_saved": _relpath(a_seg_path),
                "B_segment_saved": _relpath(b_seg_path),
                "A_drums_segment_saved": _relpath(drums_segment_for_condition),
            },
            "generation": {
                "model_id": model_id,
                "device": device,
                "cfg_all": cfg_all,
                "cfg_txt": cfg_txt,
                "ode_rtol": ode_rtol,
                "ode_atol": ode_atol,
                "euler": euler,
                "euler_steps": euler_steps,
                "segment_duration": segment_duration,
                "seed": int(run_seed),
                "positive_seed_mode": positive_seed_mode,
                "conditioning": {"drums": True, "melody": False, "chords": False},
            },
            "prompts": {"anchor_prompt": _anchor_prompt, "positive_prompts": positive_prompts},
            "outputs": {
                "anchor_wav": anchor_path,
                "negative_wav": negative_path,
                "positives": [x["output_wav"] for x in positives_info],
            },
        }
        (triplet_dir / "triplet_meta.json").write_text(
            json.dumps(triplet_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[triplets_{trip_idx}] OK -> {run_dir}")

if __name__ == "__main__":
    main()
