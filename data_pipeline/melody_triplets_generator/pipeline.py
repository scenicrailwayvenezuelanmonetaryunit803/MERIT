#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
import torchaudio

import numpy as np
import torch

import jasco_ravel_80s as gen

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triplets_input_index.index_builder import build_melody_index

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

def _resolve_index_audio_root(index_obj: dict) -> Path:
    config = index_obj.get("config", {}) if isinstance(index_obj, dict) else {}
    raw_root = config.get("data_root") or config.get("audio_root") or config.get("moisesdb_root")
    if not raw_root:
        return MOISESDB_ROOT
    path = Path(str(raw_root))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path

def _resolve_index_salience_root(index_obj: dict, default_root: Path) -> Path:
    config = index_obj.get("config", {}) if isinstance(index_obj, dict) else {}
    raw_root = config.get("salience_root")
    if not raw_root:
        return default_root
    path = Path(str(raw_root))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path

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

def _median_smooth_int(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = int(np.median(padded[i : i + win]))
    return out

def _build_cqt_argmax_onehot_salience_from_segment_wav(
    wav_path: Path,
    segment_duration: float,
    frame_rate: float,
    midi_min: int = 43,
    midi_max: int = 95,
    sr: int = 22050,
    hop: int = 256,
    voiced_db: float = -80.0,
    smooth_win: int = 5,
    use_harmonic: bool = True,
) -> torch.Tensor:
    try:
        import librosa
    except Exception as e:
        raise SystemExit(f"librosa is required for pipeline salience extraction: {e}")

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    if y.size == 0:
        raise SystemExit(f"Empty audio: {wav_path}")

    target_len = int(round(segment_duration * sr))
    if y.shape[0] < target_len:
        y = librosa.util.fix_length(y, size=target_len)
    else:
        y = y[:target_len]

    if use_harmonic:
        y = librosa.effects.harmonic(y)

    n_bins = (midi_max - midi_min + 1)
    fmin = librosa.midi_to_hz(midi_min)

    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=12,
    )
    mag = np.abs(cqt).astype(np.float32, copy=False)
    F, T_src = mag.shape
    if F != n_bins:
        raise SystemExit(f"Unexpected CQT bins: got {F}, expected {n_bins}")

    eps = 1e-8
    frame_max = mag.max(axis=0) + eps
    global_max = float(frame_max.max() + eps)
    frame_db = 20.0 * np.log10(frame_max / global_max)
    voiced = frame_db >= float(voiced_db)

    expected_T = int(round(segment_duration * frame_rate))
    x = torch.from_numpy(mag).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=expected_T, mode="linear", align_corners=False)
    mag_rs = x.squeeze(0).numpy()

    idx = np.argmax(mag_rs, axis=0).astype(np.int32, copy=False)
    if int(smooth_win) > 1:
        idx = _median_smooth_int(idx, int(smooth_win))

    out = np.zeros_like(mag_rs, dtype=np.float32)
    t = np.arange(mag_rs.shape[1])
    out[idx, t] = 1.0

    if not np.all(voiced):
        voiced_rs = torch.from_numpy(voiced.astype(np.float32)).view(1, 1, -1)
        voiced_rs = torch.nn.functional.interpolate(voiced_rs, size=expected_T, mode="linear", align_corners=False)
        voiced_rs = (voiced_rs.view(-1).numpy() >= 0.5)
        out[:, ~voiced_rs] = 0.0

    return torch.from_numpy(out)

def _cqt_peakiness_stats_from_segment_wav(
    wav_path: Path,
    segment_duration: float,
    midi_min: int = 43,
    midi_max: int = 95,
    sr: int = 22050,
    hop: int = 256,
    voiced_db: float = -80.0,
    use_harmonic: bool = True,
) -> dict:
    try:
        import librosa
    except Exception as e:
        raise SystemExit(f"librosa is required for pipeline peakiness extraction: {e}")

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    if y.size == 0:
        return {"ratio_med": 0.0, "ratio_mean": 0.0, "voiced_ratio": 0.0}

    target_len = int(round(segment_duration * sr))
    if y.shape[0] < target_len:
        y = librosa.util.fix_length(y, size=target_len)
    else:
        y = y[:target_len]

    if use_harmonic:
        y = librosa.effects.harmonic(y)

    n_bins = (midi_max - midi_min + 1)
    fmin = librosa.midi_to_hz(midi_min)
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=12,
    )
    mag = np.abs(cqt).astype(np.float32, copy=False)
    if mag.size == 0:
        return {"ratio_med": 0.0, "ratio_mean": 0.0, "voiced_ratio": 0.0}

    eps = 1e-8

    top2 = np.partition(mag, -2, axis=0)[-2:]
    first = np.maximum(top2[0], top2[1])
    second = np.minimum(top2[0], top2[1])
    ratio = (first + eps) / (second + eps)

    frame_max = mag.max(axis=0) + eps
    global_max = float(frame_max.max() + eps)
    frame_db = 20.0 * np.log10(frame_max / global_max)
    voiced = frame_db >= float(voiced_db)

    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    if not np.any(voiced):
        return {"ratio_med": 0.0, "ratio_mean": 0.0, "voiced_ratio": voiced_ratio}

    r = ratio[voiced]
    return {
        "voiced_ratio": voiced_ratio,
        "ratio_med": float(np.median(r)),
        "ratio_mean": float(np.mean(r)),
        "ratio_p10": float(np.percentile(r, 10)),
        "ratio_p90": float(np.percentile(r, 90)),
    }

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

def _scan_moisesdb_tracks(moises_root: Path, stems: list[str]) -> list[dict]:
    base = moises_root / MOISESDB_VERSION_DIRNAME
    if not base.exists():
        raise SystemExit(f"MoisesDB not found: {base} (expected extracted DB at {moises_root})")

    stems = [s.strip() for s in stems if s.strip()]
    tracks: list[dict] = []
    for track_dir in sorted(base.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        files: list[str] = []
        for stem in stems:
            stem_dir = track_dir / stem
            if not stem_dir.is_dir():
                continue
            for wav in sorted(stem_dir.glob("*.wav")):
                rel = (Path(MOISESDB_VERSION_DIRNAME) / track_id / stem / wav.name).as_posix()
                files.append(rel)
        tracks.append({"track_id": track_id, "files": files})
    return tracks

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

def _stem_from_relpath(relpath: str) -> str | None:
    parts = relpath.split("/")
    if len(parts) < 3:
        return None
    return parts[-2]

def _tracks_with_stem(moises_root: Path, tracks: list[dict], stem: str) -> list[dict]:
    out: list[dict] = []
    for t in tracks:
        track_id = t.get("track_id")
        files = t.get("files")
        if not isinstance(track_id, str) or not isinstance(files, list):
            continue
        matched: list[str] = []
        for f in files:
            if not isinstance(f, str):
                continue
            if not f.endswith(".wav"):
                continue
            if _stem_from_relpath(f) != stem:
                continue
            if not (moises_root / f).exists():
                continue
            matched.append(f)
        if matched:
            out.append({"track_id": track_id, "stem": stem, "files": matched})
    return out

def _runtime_diag() -> None:
    try:
        import transformers
    except Exception as e:
        raise SystemExit(f"Missing dependency: transformers ({e}). Please run with the project venv.")

    print(f"python = {sys.executable}")
    print(f"torch = {torch.__version__}")
    print(f"transformers = {getattr(transformers, '__version__', 'unknown')}")

    is_torch_available = None
    for attr in ("is_torch_available",):
        fn = getattr(transformers, attr, None)
        if callable(fn):
            is_torch_available = fn
            break
    if is_torch_available is None:
        try:
            from transformers.utils import is_torch_available as fn
            is_torch_available = fn
        except Exception:
            is_torch_available = None

    if callable(is_torch_available):
        ok = bool(is_torch_available())
        if not ok:
            raise SystemExit(
                "Environment error: transformers reports torch backend is unavailable.\n"
                "Ensure you are running in the correct conda/venv environment with "
                "matching torch and transformers versions. See README.md for setup instructions."
            )

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate N melody triplets (pipeline).")
    ap.add_argument("num_triplets", type=int, help="How many triplets to generate.")
    ap.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="Root output directory for triplets (required).",
    )
    ap.add_argument(
        "--index", type=Path, default=None,
        help="Path to pre-built melody_index.json (overrides TRIPLETS_MELODY_INDEX_JSON env and auto-build).",
    )
    ap.add_argument(
        "--num-positives", type=int, default=3,
        help="Number of positive samples per triplet (default: 3).",
    )
    ap.add_argument(
        "--prompts-file", type=Path, default=None,
        help="Prompts file. Auto-discovers prompts_5000.txt then prompts.txt if omitted.",
    )
    ap.add_argument(
        "--allow-anchor-reuse", action="store_true",
        help="Cycle through anchors when num_triplets > available unique anchors.",
    )
    ap.add_argument(
        "--model", type=str,
        default="facebook/jasco-chords-drums-melody-1B",
        help="JASCO model ID (default: facebook/jasco-chords-drums-melody-1B). "
             "Use melody-400M for a lighter alternative.",
    )
    ap.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES before any CUDA init). "
             "E.g. --gpu 2 runs on cuda:2. Omit to use auto-detected device.",
    )
    ap.add_argument(
        "--start-at", type=int, default=None,
        help="Override the starting triplet number instead of auto-detecting from output-dir. "
             "Useful for splitting work across multiple processes (e.g. --start-at 835 for GPU 1).",
    )
    args = ap.parse_args()

    if args.num_triplets <= 0:
        raise SystemExit("num_triplets must be > 0")

    # Set GPU environment BEFORE any CUDA / torch init so the right device is visible
    if args.gpu is not None:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")

    _runtime_diag()

    model_id = args.model
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

    cfg_all = 1.25
    cfg_txt = 2.5

    ode_rtol = None
    ode_atol = None
    euler = False
    euler_steps = None

    drums_mode = "none"

    base_seed = 1234
    positive_seed_mode = "same"

    print("Loading prefiltered melody index...")
    if args.index is not None:
        if not args.index.exists():
            raise SystemExit(f"--index path not found: {args.index}")
        melody_index = json.loads(args.index.read_text(encoding="utf-8"))
        if str(melody_index.get("generator", "")).strip() != "melody":
            raise SystemExit(f"--index JSON is not a melody index (generator != 'melody'): {args.index}")
        print(f"Using melody index from --index: {args.index}")
    else:
        melody_index = _load_index_from_env("TRIPLETS_MELODY_INDEX_JSON", "melody")
        if melody_index is None:
            melody_index = build_melody_index(force=False, verbose=True)
    config = melody_index.get("config", {}) if isinstance(melody_index, dict) else {}
    audio_root = _resolve_index_audio_root(melody_index)
    if not audio_root.exists():
        raise SystemExit(
            f"Audio root from index does not exist: {audio_root}\n"
            "Check 'moisesdb_root' / 'audio_root' in the index config, or pass a corrected --index."
        )
    salience_root = _resolve_index_salience_root(melody_index, audio_root)
    dataset_name = str(config.get("dataset", "moisesdb")).strip() or "moisesdb"
    salience_tau = float(config.get("salience_tau", 0.3))
    preferred_stems = list(melody_index.get("config", {}).get("preferred_stems") or [])
    min_anchor_nonzero_ratio_after_fill = float(
        config.get("min_anchor_nonzero_ratio_after_fill", min_anchor_nonzero_ratio_after_fill)
    )
    entries_by_stem = {
        stem: list(melody_index.get("entries_by_stem", {}).get(stem) or [])
        for stem in preferred_stems
    }
    stem_counts = ", ".join(f"{stem}={len(entries_by_stem.get(stem) or [])}" for stem in preferred_stems)
    print(
        f"Audio root: {audio_root} "
        f"(dataset={dataset_name}, prefiltered anchors={len(melody_index.get('anchors') or [])}, {stem_counts})"
    )
    if salience_root != audio_root or "salience_root" in config:
        print(f"Salience root: {salience_root} (tau={salience_tau})")

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

    # Load all prompts once; anchor + positives are picked randomly per-triplet for diversity
    if args.prompts_file is not None:
        _prompts_path = args.prompts_file
    elif (SCRIPT_DIR / "prompts_5000.txt").exists():
        _prompts_path = SCRIPT_DIR / "prompts_5000.txt"
    else:
        _prompts_path = SCRIPT_DIR / "prompts.txt"
    all_prompts = gen._load_positive_prompts(str(_prompts_path), [])
    if len(all_prompts) < args.num_positives + 1:
        raise SystemExit(
            f"Need at least num_positives+1={args.num_positives + 1} prompts "
            f"in {_prompts_path} (got {len(all_prompts)})"
        )
    print(f"Loaded {len(all_prompts)} prompts from {_prompts_path}")

    melody_lock = False
    melody_lock_suffix = (
        "Main melody must be clearly audible and strictly follow the provided melody contour; "
        "no counter-melody; no improvisation."
    )
    if melody_lock:
        all_prompts = [gen._apply_prompt_suffix(p, melody_lock_suffix) for p in all_prompts]

    anchor_plan: list[dict] = []
    selected_track_ids: set[str] = set()
    index_anchors = list(melody_index.get("anchors") or [])
    if not index_anchors:
        raise SystemExit(
            "Melody index exists but contains no valid anchors. "
            f"Please rebuild it: python {REPO_ROOT / 'triplets_input_index' / 'build_index.py'} melody --force"
        )

    scanned_tracks = 0
    for rec in index_anchors:
        track_id = str(rec.get("track_id", "")).strip()
        if not track_id:
            continue
        if track_id in used_a_track_ids or track_id in selected_track_ids:
            continue
        scanned_tracks += 1
        stem = str(rec.get("stem", "")).strip()
        b_choices = [x for x in entries_by_stem.get(stem) or [] if str(x.get("track_id", "")) != track_id]
        if not b_choices:
            continue
        anchor_plan.append(rec)
        selected_track_ids.add(track_id)
        print(
            f"[anchor] selected (index): track={track_id} stem={stem} "
            f"ratio_after_fill={float(rec.get('ratio_after_fill', 0.0)):.3f} wav={rec.get('wav_rel')}"
        )

        if len(anchor_plan) >= args.num_triplets:
            break

    if len(anchor_plan) < args.num_triplets:
        if not args.allow_anchor_reuse:
            raise SystemExit(
                f"Not enough unique anchors (have {len(anchor_plan)}, need {args.num_triplets}).\n"
                "Pass --allow-anchor-reuse to cycle through anchors with different random prompts per triplet.\n"
                f"(min_anchor_nonzero_ratio={min_anchor_nonzero_ratio_after_fill:.2f}, "
                f"scanned={scanned_tracks}, selected={len(anchor_plan)})"
            )
        # Build a validated pool of anchors that have >=2 tracks per stem, then cycle
        valid_pool = [
            rec for rec in index_anchors
            if str(rec.get("track_id", "")).strip()
            and [x for x in (entries_by_stem.get(str(rec.get("stem", ""))) or [])
                 if str(x.get("track_id", "")) != str(rec.get("track_id", ""))]
        ]
        if not valid_pool:
            raise SystemExit("No valid anchors with >=2 tracks per stem found in index.")
        while len(anchor_plan) < args.num_triplets:
            anchor_plan.append(valid_pool[len(anchor_plan) % len(valid_pool)])
        print(
            f"[anchor-reuse] Cycled {len(valid_pool)} valid anchors to fill {len(anchor_plan)} slots. "
            "Each reuse will use different random prompts."
        )

    stem_counts: dict[str, int] = {}
    for a in anchor_plan:
        s = str(a.get("stem", ""))
        stem_counts[s] = stem_counts.get(s, 0) + 1
    stem_counts_str = ", ".join(f"{k}={v}" for k, v in stem_counts.items())
    print(f"Selected anchors: {stem_counts_str} (anchors_selected={len(anchor_plan)})")

    from audiocraft.models import JASCO
    from audiocraft.data.audio import audio_write

    chord_map = gen.PROJECT_ROOT / "assets" / "chord_to_index_mapping.pkl"
    if not chord_map.exists():
        raise SystemExit(f"Chord map not found: {chord_map}")

    device = gen.pick_device("auto")
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
    filtered, skipped = gen._filter_supported_generate_kwargs(model, gen_kwargs)
    model.set_generation_params(**filtered)
    print("Applied generation params:", filtered)
    if skipped:
        print("Skipped unsupported generation params:", skipped)

    for plan_idx, trip_idx in enumerate(range(start_idx, end_idx + 1), start=0):
        run_dir = out_base / f"triplets_{trip_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        triplet_dir = run_dir / "triplet"
        triplet_dir.mkdir(parents=True, exist_ok=True)

        run_seed = base_seed + int(trip_idx)
        rng = np.random.default_rng(run_seed)

        # Per-triplet random prompt selection: 1 anchor prompt + num_positives positive prompts
        _n_pick = args.num_positives + 1
        _prompt_idxs = rng.choice(len(all_prompts), size=min(_n_pick, len(all_prompts)), replace=False)
        anchor_prompt = all_prompts[int(_prompt_idxs[0])]
        positive_prompts = [all_prompts[int(i)] for i in _prompt_idxs[1:]]

        a_seg_path = run_dir / "input_A_segment.wav"
        b_seg_path = run_dir / "input_B_segment.wav"

        a_plan = anchor_plan[plan_idx]
        input_stem = str(a_plan.get("stem", ""))
        if not input_stem:
            raise SystemExit(f"[triplets_{trip_idx}] Internal error: unknown stem in anchor_plan: {input_stem!r}")
        candidates = list(entries_by_stem.get(input_stem) or [])

        input_a = audio_root / str(a_plan["wav_rel"])
        a_track_id = str(a_plan["track_id"])
        used_offset_a = float(a_plan["offset_sec"])
        sr_a, ns_a, used_offset_a = gen._save_input_segment(input_a, a_seg_path, used_offset_a, segment_duration)

        b_choices = [t for t in candidates if str(t.get("track_id", "")) != a_track_id]
        if not b_choices:
            raise SystemExit(f"Need >=2 tracks for stem='{input_stem}' to pick B != A, but only found 1.")
        b = b_choices[int(rng.integers(0, len(b_choices)))]
        b_rel = str(b["wav_rel"])
        input_b = audio_root / b_rel
        b_track_id = str(b["track_id"])
        sr_b, ns_b, used_offset_b = gen._save_input_segment(
            input_b, b_seg_path, float(b.get("offset_sec", -1.0)), segment_duration
        )

        print(
            f"[triplets_{trip_idx}] inputs: "
            f"A(track={a_track_id}, stem={input_stem}, wav={_relpath(input_a)}, sr={sr_a}, samples={ns_a}, offset={used_offset_a:.2f}s, salience_ratio_after_fill={a_plan['ratio_after_fill']:.3f}) "
            f"B(track={b_track_id}, stem={input_stem}, wav={_relpath(input_b)}, sr={sr_b}, samples={ns_b}, offset={used_offset_b:.2f}s)"
        )

        a_npz_rel = str(a_plan.get("salience_npz_rel", "")).strip()
        b_npz_rel = str(b.get("salience_npz_rel", "")).strip()
        a_npz = str(salience_root / a_npz_rel) if a_npz_rel else ""
        b_npz = str(salience_root / b_npz_rel) if b_npz_rel else ""

        sal_a, sal_a_source, sal_a_stats = gen._build_salience(
            wav_path=input_a,
            salience_npz=a_npz,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            offset_sec=used_offset_a,
            tau=salience_tau,
            pyin_sr=pyin_sr,
            voiced_prob_thres=voiced_prob_thres,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            smooth_win=smooth_win,
            fill_gaps_frames=fill_gaps_frames,
        )
        sal_b, sal_b_source, sal_b_stats = gen._build_salience(
            wav_path=input_b,
            salience_npz=b_npz,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            offset_sec=used_offset_b,
            tau=salience_tau,
            pyin_sr=pyin_sr,
            voiced_prob_thres=voiced_prob_thres,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            smooth_win=smooth_win,
            fill_gaps_frames=fill_gaps_frames,
        )
        r1_a = float(sal_a_stats["nonzero_frame_ratio_after_fill"])

        if float(r1_a) < float(min_anchor_nonzero_ratio_after_fill):
            raise SystemExit(
                f"[triplets_{trip_idx}] Anchor salience too sparse after fill: {r1_a:.3f} < {min_anchor_nonzero_ratio_after_fill:.2f}. "
                "Please increase dataset quality or lower the threshold."
            )

        sal_a_path = run_dir / "salience_A.npy"
        sal_b_path = run_dir / "salience_B.npy"
        np.save(str(sal_a_path), sal_a.numpy())
        np.save(str(sal_b_path), sal_b.numpy())

        drums_wav = None
        if drums_mode == "silent":
            drums_wav = gen._prepare_silent_drums(model.sample_rate, segment_duration, device)

        sal_a = sal_a.to(torch.device(device))
        sal_b = sal_b.to(torch.device(device))

        manifest_path = run_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        def _append_manifest(obj: dict) -> None:
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        def _write_segment_as(name: str, seg_wav_path: Path) -> str:
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

        anchor_path = _write_segment_as("anchor", a_seg_path)
        _append_manifest({
            "type": "anchor",
            "output_wav": anchor_path,
            "prompt": anchor_prompt,
            "seed": None,
            "stats": {"source": "input_A_segment"},
        })

        negative_path = _write_segment_as("negative", b_seg_path)
        _append_manifest({
            "type": "negative",
            "output_wav": negative_path,
            "prompt": anchor_prompt,
            "seed": None,
            "stats": {"source": "input_B_segment"},
        })

        # Batch-generate all positives in a single JASCO forward pass (same melody, different prompts)
        pos_seed = run_seed if positive_seed_mode == "same" else run_seed + 1
        positives_info: list[dict] = []
        wavs_pos, stats_batch = gen._generate_batch(
            model=model,
            device=device,
            prompts=positive_prompts,
            salience=sal_a,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            progress=False,
            seed=pos_seed,
            drums_wav=drums_wav,
        )
        # wavs_pos: [N, C, T] on CPU
        for k, pos_prompt in enumerate(positive_prompts, start=1):
            wav_pos_k = wavs_pos[k - 1]  # [C, T]
            name = f"positive_{k:02d}"
            audio_write(str(triplet_dir / name), wav_pos_k, model.sample_rate,
                        strategy="loudness", loudness_compressor=True, add_suffix=True)
            out_path = _relpath(triplet_dir / f"{name}.wav")
            rec = {
                "type": "positive",
                "positive_index": k,
                "output_wav": out_path,
                "prompt": pos_prompt,
                "seed": int(pos_seed),
                "stats": stats_batch,
            }
            positives_info.append(rec)
            _append_manifest(rec)

        run_meta = {
            "model_id": model_id,
            "device": device,
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "segment_duration": segment_duration,
            "frame_rate": frame_rate,
            "melody_bins": melody_bins,
            "cfg_all": cfg_all,
            "cfg_txt": cfg_txt,
            "ode_rtol": ode_rtol,
            "ode_atol": ode_atol,
            "euler": euler,
            "euler_steps": euler_steps,
            "drums_mode": drums_mode,
            "seed": int(run_seed),
            "positive_seed_mode": positive_seed_mode,
            "melody_lock": melody_lock,
            "melody_lock_suffix": melody_lock_suffix,
            "inputs": {
                "dataset": dataset_name,
                "audio_root": _relpath(audio_root),
                "salience_root": _relpath(salience_root),
                "salience_tau": salience_tau,
                "stem": input_stem,
                "A_track_id": a_track_id,
                "B_track_id": b_track_id,
                "A_wav": _relpath(input_a),
                "B_wav": _relpath(input_b),
                "offset_a": float(used_offset_a),
                "offset_b": float(used_offset_b),
                "A_segment_saved": _relpath(a_seg_path),
                "B_segment_saved": _relpath(b_seg_path),
                "salience_A_source": sal_a_source,
                "salience_B_source": sal_b_source,
                "salience_A_saved": _relpath(sal_a_path),
                "salience_B_saved": _relpath(sal_b_path),
                "salience_A_stats": sal_a_stats,
                "salience_B_stats": sal_b_stats,
            },
            "prompts": {
                "anchor_prompt": anchor_prompt,
                "positive_prompts": positive_prompts,
            },
            "note": "Pipeline run. Anchor/Negative are the extracted input segments (A/B) resampled to model.sample_rate; only Positives are synthesized with JASCO using melody A.",
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        triplet_meta = {
            "triplet_index": 1,
            "triplet_dir": _relpath(triplet_dir),
            "inputs": {
                "dataset": dataset_name,
                "audio_root": _relpath(audio_root),
                "salience_root": _relpath(salience_root),
                "salience_tau": salience_tau,
                "stem": input_stem,
                "A_track_id": a_track_id,
                "B_track_id": b_track_id,
                "A_wav": _relpath(input_a),
                "B_wav": _relpath(input_b),
                "A_offset_sec": float(used_offset_a),
                "B_offset_sec": float(used_offset_b),
                "A_segment_saved": _relpath(a_seg_path),
                "B_segment_saved": _relpath(b_seg_path),
                "salience_A_source": sal_a_source,
                "salience_B_source": sal_b_source,
                "salience_A_saved": _relpath(sal_a_path),
                "salience_B_saved": _relpath(sal_b_path),
                "salience_A_stats": sal_a_stats,
                "salience_B_stats": sal_b_stats,
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
                "drums_mode": drums_mode,
                "segment_duration": segment_duration,
                "frame_rate": frame_rate,
                "melody_bins": melody_bins,
                "seed": int(run_seed),
                "positive_seed_mode": positive_seed_mode,
                "melody_lock": melody_lock,
                "melody_lock_suffix": melody_lock_suffix,
            },
            "prompts": {
                "anchor_prompt": anchor_prompt,
                "positive_prompts": positive_prompts,
            },
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
