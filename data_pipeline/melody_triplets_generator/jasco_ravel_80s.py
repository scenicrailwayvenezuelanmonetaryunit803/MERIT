#!/usr/bin/env python3
"""
Generate ONE melody-triplet (Anchor/Positive/Negative) with JASCO (melody conditioning).

Triplet definition (melody-triplet):
- Anchor:   melody from input A, style = anchor_prompt
- Positive: melody from input A, style = positive_prompts (multiple, e.g., 5 variants)
- Negative: melody from input B, style = anchor_prompt (same style as Anchor, different melody)

Key points vs typical "quality is bad" cases:
1) Use paper-recommended multi-source CFG defaults:
   cfg_coef_all=1.5, cfg_coef_txt=0.5  (JASCO paper)  # see arXiv 2406.10970
2) pYIN fallback can produce many all-zero frames => weak melody constraint.
   We report nonzero_frame_ratio and optionally fill short gaps (--fill_gaps).
3) Optional silent drums conditioning (--drums_mode silent) to reduce overpowering drums.

"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------------------------------
# JASCO / audiocraft path bootstrap
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

_JASCO_ROOT_ENV = os.environ.get("JASCO_ROOT", "").strip()
# PROJECT_ROOT = root of the cloned JASCO audiocraft repo (contains audiocraft/ and assets/)
PROJECT_ROOT: Path | None = Path(_JASCO_ROOT_ENV) if _JASCO_ROOT_ENV else None
AUDIOCRAFT_REPO_ROOT: Path | None = (PROJECT_ROOT / "audiocraft") if PROJECT_ROOT else None

if PROJECT_ROOT is not None:
    if str(AUDIOCRAFT_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(AUDIOCRAFT_REPO_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(1, str(PROJECT_ROOT))


# -----------------------------
# Melody salience builders
# -----------------------------
def _midi_from_hz(f: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(f / 440.0)


def _build_binary_from_scores(scores: np.ndarray, tau: float) -> np.ndarray:
    """
    For DeepSalience-like scores [BINS, T]:
      - zero below tau
      - per-frame argmax -> 1-hot, or all-zero frame if nothing passes tau
    """
    scores = scores.astype(np.float32, copy=False)
    scores[scores < tau] = 0.0
    idx = np.argmax(scores, axis=0)
    mx = np.max(scores, axis=0)
    out = np.zeros_like(scores, dtype=np.float32)
    t = np.arange(scores.shape[1])
    out[idx, t] = 1.0
    out[:, mx == 0.0] = 0.0
    return out


def build_salience_from_deepsalience_npz(
    npz_path: Path,
    segment_duration: float,
    target_frame_rate: float,
    melody_bins: int = 53,
    offset_sec: float = 0.0,
    tau: float = 0.3,
    midi_min: int = 43,  # G2
    midi_max: int = 95,  # B6  (43..95 inclusive = 53 bins)
) -> torch.Tensor:
    """
    Load DeepSalience-style npz:
      expects keys:
        - 'salience' [F, T]
        - 'freqs'    [F]
        - 'times'    [T] (optional but recommended)
    Steps:
      1) slice times to [offset, offset+duration)
      2) map frequency bins -> MIDI, aggregate into [53, T]
      3) resample time axis to target_frame_rate (e.g., 50Hz)
      4) threshold + argmax -> binary [53, target_T]
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Salience npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "salience" not in data or "freqs" not in data:
        raise ValueError(
            f"npz missing required keys. Need at least 'salience' and 'freqs'. Got: {list(data.keys())}"
        )

    sal = np.asarray(data["salience"], dtype=np.float32)  # [F, T]
    freqs = np.asarray(data["freqs"], dtype=np.float32)   # [F]
    times = np.asarray(data["times"], dtype=np.float32) if "times" in data else None

    # Slice time window
    if times is not None and times.ndim == 1 and times.shape[0] == sal.shape[1]:
        t0 = offset_sec
        t1 = offset_sec + segment_duration
        keep_t = (times >= t0) & (times < t1)
        if not np.any(keep_t):
            raise ValueError(
                f"No frames found in npz for time window [{t0}, {t1}). "
                "Check your npz corresponds to the same audio excerpt."
            )
        sal = sal[:, keep_t]
    else:
        # Fallback: assume starts at 0 and approximate fps (~86 if hop=256 @ 22050)
        src_fps = 22050.0 / 256.0
        want = int(round(segment_duration * src_fps))
        sal = sal[:, :want]

    # Aggregate frequency bins -> MIDI bins [53, T]
    midi = _midi_from_hz(freqs)
    midi_rounded = np.rint(midi).astype(np.int32)

    if (midi_max - midi_min + 1) != melody_bins:
        raise ValueError("midi_min/max must match melody_bins=53 (inclusive range).")

    agg = np.zeros((melody_bins, sal.shape[1]), dtype=np.float32)
    for f in range(freqs.shape[0]):
        m = midi_rounded[f]
        if midi_min <= m <= midi_max:
            agg[m - midi_min] += sal[f]

    # Resample time axis to target_frame_rate
    target_T = int(round(segment_duration * target_frame_rate))
    x = torch.from_numpy(agg).unsqueeze(0)  # [1, C, T]
    x = F.interpolate(x, size=target_T, mode="linear", align_corners=False)
    agg_rs = x.squeeze(0).numpy()

    binary = _build_binary_from_scores(agg_rs, tau=tau)
    return torch.from_numpy(binary)  # CPU tensor [53, target_T]


def build_salience_from_wav_pyin(
    wav_path: Path,
    segment_duration: float,
    frame_rate: float,
    melody_bins: int = 53,
    melody_sr: int = 22050,
    offset_sec: float = 0.0,
    voiced_prob_thres: float = 0.2,
    f0_min_hz: float = 65.0,    # allow lower notes; we still clamp to G2..B6 bins below
    f0_max_hz: float = 2000.0,
    midi_min: int = 43,         # G2
    midi_max: int = 95,         # B6
    smooth_win: int = 5,
) -> torch.Tensor:
    """
    pYIN fallback (best for monophonic-ish inputs).
    We:
      - load mono @ melody_sr
      - cut [offset, offset+duration]
      - take harmonic component
      - run pYIN
      - map to 53 MIDI bins (G2..B6) by clamping
    """
    try:
        import librosa
    except Exception as e:
        raise RuntimeError(
            "librosa is required for pYIN fallback but could not be imported. "
            "Install librosa or provide --salience_*_npz.\n"
            f"Import error: {e}"
        )

    y, sr = librosa.load(str(wav_path), sr=melody_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Loaded empty audio: {wav_path}")

    start = int(round(offset_sec * sr))
    end = start + int(round(segment_duration * sr))
    if start >= y.shape[0]:
        raise ValueError(f"offset_sec={offset_sec} is beyond audio length for {wav_path}.")
    y = y[start:end]
    if y.shape[0] < int(round(segment_duration * sr)):
        y = librosa.util.fix_length(y, size=int(round(segment_duration * sr)))

    # Harmonic part reduces percussive interference
    y_harm = librosa.effects.harmonic(y)

    hop_length = max(1, int(round(sr / frame_rate)))
    expected_frames = int(round(segment_duration * frame_rate))

    f0, _, voiced_prob = librosa.pyin(
        y_harm,
        fmin=f0_min_hz,
        fmax=f0_max_hz,
        sr=sr,
        hop_length=hop_length,
    )

    if f0 is None:
        f0 = np.full((expected_frames,), np.nan, dtype=np.float32)
        voiced_prob = np.zeros((expected_frames,), dtype=np.float32)
    else:
        f0 = np.asarray(f0, dtype=np.float32)
        voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    # pad/trim to expected_frames
    if f0.shape[0] < expected_frames:
        pad = expected_frames - f0.shape[0]
        f0 = np.pad(f0, (0, pad), constant_values=np.nan)
        voiced_prob = np.pad(voiced_prob, (0, pad), constant_values=0.0)
    else:
        f0 = f0[:expected_frames]
        voiced_prob = voiced_prob[:expected_frames]

    valid = (voiced_prob >= voiced_prob_thres) & np.isfinite(f0)
    f0 = np.where(valid, f0, np.nan)

    midi = _midi_from_hz(f0)
    midi = np.where(np.isfinite(midi), midi, np.nan)

    # median smoothing (only on valid frames)
    if smooth_win > 1:
        finite = np.isfinite(midi)
        midi_tmp = np.where(finite, midi, 0.0)
        pad = smooth_win // 2
        padded = np.pad(midi_tmp, (pad, pad), mode="edge")
        sm = np.empty_like(midi_tmp)
        for i in range(midi_tmp.shape[0]):
            sm[i] = np.median(padded[i : i + smooth_win])
        midi = np.where(finite, sm, np.nan)

    if (midi_max - midi_min + 1) != melody_bins:
        raise ValueError("midi_min/max must match melody_bins=53 (inclusive range).")

    # clamp to conditioning range
    midi = np.clip(midi, midi_min, midi_max)
    idx = np.rint(np.nan_to_num(midi, nan=midi_min)).astype(np.int32) - midi_min
    idx = np.clip(idx, 0, melody_bins - 1)

    sal = np.zeros((melody_bins, expected_frames), dtype=np.float32)
    for t in range(expected_frames):
        if np.isfinite(midi[t]):
            sal[idx[t], t] = 1.0

    return torch.from_numpy(sal)  # CPU tensor [53, T]


# -----------------------------
# Utilities
# -----------------------------
def _apply_prompt_suffix(prompt: str, suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return prompt
    p = prompt.strip()
    if p.endswith(suffix):
        return p
    return (p + " " + suffix).strip()


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * np.power(2.0, (midi - 69.0) / 12.0)


def _synth_preview_from_onehot_salience(
    salience: torch.Tensor,  # [BINS, T]
    frame_rate: float,
    sr: int = 22050,
    midi_min: int = 43,
    amp: float = 0.2,
) -> np.ndarray:
    """
    Synthesize a simple sine "beep" preview of the extracted melody contour.
    Useful to quickly audit whether your salience matches the intended melody.
    """
    x = salience.detach().cpu().numpy().astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"salience must be [BINS, T], got {x.shape}")

    _, T = x.shape
    hop = max(1, int(round(sr / float(frame_rate))))
    out = np.zeros((T * hop,), dtype=np.float32)

    idx = np.argmax(x, axis=0)  # [T]
    mx = np.max(x, axis=0)  # [T]
    voiced = mx > 0

    midi = (midi_min + idx).astype(np.float32)
    freqs = _midi_to_hz(midi)

    phase = 0.0
    t = np.arange(hop, dtype=np.float32) / float(sr)
    for i in range(T):
        if not voiced[i]:
            phase = 0.0
            continue
        f = float(freqs[i])
        w = 2.0 * math.tau * f
        seg = amp * np.sin(w * t + phase)
        phase = (phase + w * hop / float(sr)) % (math.tau)
        out[i * hop : (i + 1) * hop] = seg.astype(np.float32, copy=False)
    return out


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def nonzero_frame_ratio(sal: torch.Tensor) -> float:
    with torch.no_grad():
        nz = (sal.sum(dim=0) > 0).float().mean().item()
    return float(nz)


def fill_short_gaps(sal: torch.Tensor, max_gap_frames: int) -> torch.Tensor:
    """
    Fill runs of all-zero frames up to max_gap_frames by copying previous nonzero frame.
    This helps when pYIN intermittently drops voicing for a couple of frames.
    """
    if max_gap_frames <= 0:
        return sal

    x = sal.detach().cpu().numpy().astype(np.float32, copy=False)  # [B, T]
    B, T = x.shape
    nz = (x.sum(axis=0) > 0)

    if nz.all():
        return sal

    t = 0
    while t < T:
        if nz[t]:
            t += 1
            continue
        start = t
        while t < T and not nz[t]:
            t += 1
        end = t
        gap_len = end - start
        if gap_len <= max_gap_frames:
            if start > 0 and nz[start - 1]:
                x[:, start:end] = x[:, start - 1 : start]
                nz[start:end] = True
            elif end < T and nz[end]:
                x[:, start:end] = x[:, end : end + 1]
                nz[start:end] = True

    return torch.from_numpy(x)  # CPU tensor


def _save_input_segment(
    input_wav: Path,
    out_path: Path,
    offset_sec: float,
    segment_duration: float,
    quiet: bool = False,
) -> Tuple[int, int, float]:
    import torchaudio

    y, sr = torchaudio.load(str(input_wav))  # [C, T]
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

    # If the requested segment is effectively silent, search the file for a non-silent window.
    # This prevents producing "10s of silence" inputs that make melody conditioning useless.
    SILENCE_RMS_THRES = 1e-4
    SEARCH_HOP_SEC = 1.0
    seg_rms = _rms_mono(seg)
    if auto_pick or (seg_rms < SILENCE_RMS_THRES and max_start > 0):
        mono_full = y.float().mean(dim=0)  # [T]
        sq = mono_full.pow(2)
        csum = torch.cat([torch.zeros(1, dtype=sq.dtype), torch.cumsum(sq, dim=0)], dim=0)  # [T+1]

        hop = max(1, int(round(SEARCH_HOP_SEC * sr)))
        candidates = list(range(0, max_start + 1, hop))
        if candidates[-1] != max_start:
            candidates.append(max_start)

        # window mean-square via prefix sums
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
        # Audio too short to search different windows (or max_start == 0).
        raise ValueError(
            f"Input segment at offset={offset_sec:.2f}s looks silent and no alternative window exists: "
            f"{input_wav} (rms={seg_rms:.6f} < {SILENCE_RMS_THRES})."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), seg, sr)
    return sr, seg.shape[1], used_offset_sec


def _load_positive_prompts(positive_prompts_file: str, positive_prompt_inline: Sequence[str]) -> List[str]:
    if positive_prompt_inline:
        return [p.strip() for p in positive_prompt_inline if p.strip()]

    if positive_prompts_file:
        p = Path(positive_prompts_file)
        if not p.exists():
            raise FileNotFoundError(f"positive_prompts_file not found: {p}")
        lines = p.read_text(encoding="utf-8").splitlines()
        out: List[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        if not out:
            raise ValueError(f"positive_prompts_file is empty or only comments: {p}")
        return out

    raise ValueError(
        "No positive prompts provided. Please pass --positive_prompts_file or --positive_prompt. "
        "Recommended: use the repo's prompts.txt."
    )


def _build_salience(
    wav_path: Path,
    salience_npz: str,
    segment_duration: float,
    frame_rate: float,
    melody_bins: int,
    offset_sec: float,
    tau: float,
    pyin_sr: int,
    voiced_prob_thres: float,
    f0_min_hz: float,
    f0_max_hz: float,
    smooth_win: int,
    fill_gaps_frames: int,
) -> Tuple[torch.Tensor, str, Dict[str, float]]:
    """
    Returns:
      salience (CPU tensor), source_str, stats dict (ratios etc)
    """
    if salience_npz:
        try:
            sal = build_salience_from_deepsalience_npz(
                npz_path=Path(salience_npz),
                segment_duration=segment_duration,
                target_frame_rate=frame_rate,
                melody_bins=melody_bins,
                offset_sec=offset_sec,
                tau=tau,
            )
            source = "deepsalience_npz"
        except Exception as e:
            print(
                f"[WARN] DeepSalience npz failed at offset={offset_sec:.2f}s for {wav_path} "
                f"(npz={salience_npz}): {e}. Falling back to pYIN."
            )
            salience_npz = ""
            sal = build_salience_from_wav_pyin(
                wav_path=wav_path,
                segment_duration=segment_duration,
                frame_rate=frame_rate,
                melody_bins=melody_bins,
                melody_sr=pyin_sr,
                offset_sec=offset_sec,
                voiced_prob_thres=voiced_prob_thres,
                f0_min_hz=f0_min_hz,
                f0_max_hz=f0_max_hz,
                smooth_win=smooth_win,
            )
            source = "pyin_fallback"
    else:
        sal = build_salience_from_wav_pyin(
            wav_path=wav_path,
            segment_duration=segment_duration,
            frame_rate=frame_rate,
            melody_bins=melody_bins,
            melody_sr=pyin_sr,
            offset_sec=offset_sec,
            voiced_prob_thres=voiced_prob_thres,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            smooth_win=smooth_win,
        )
        source = "pyin_fallback"

    r0 = nonzero_frame_ratio(sal)
    if fill_gaps_frames > 0:
        sal2 = fill_short_gaps(sal, fill_gaps_frames)
        r1 = nonzero_frame_ratio(sal2)
        sal = sal2
    else:
        r1 = r0

    stats = {
        "nonzero_frame_ratio_before_fill": float(r0),
        "nonzero_frame_ratio_after_fill": float(r1),
    }
    return sal, source, stats


def _set_seed(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _filter_supported_generate_kwargs(model, kwargs: Dict) -> Tuple[Dict, List[str]]:
    """
    model.lm.generate(...) signature differs across audiocraft versions.
    Filter unknown kwargs to avoid crashes (e.g., ode_solver is NOT supported).
    """
    sig = inspect.signature(model.lm.generate)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    filtered = {}
    skipped = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if k in allowed:
            filtered[k] = v
        else:
            skipped.append(k)
    return filtered, skipped


def _prepare_silent_drums(model_sample_rate: int, segment_duration: float, device: str) -> torch.Tensor:
    T = int(round(model_sample_rate * segment_duration))
    drums = torch.zeros((1, 1, T), dtype=torch.float32)
    return drums.to(device)


def _generate_one(
    model,
    device: str,
    prompt: str,
    salience: torch.Tensor,
    segment_duration: float,
    frame_rate: float,
    melody_bins: int,
    progress: bool,
    seed: int,
    drums_wav: torch.Tensor | None,
) -> Tuple[torch.Tensor, dict]:
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
        melody_salience_matrix=salience,
        segment_duration=segment_duration,
        frame_rate=frame_rate,
        melody_bins=melody_bins,
        progress=progress,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    stats["duration_sec"] = float(t1 - t0)
    if device == "cuda":
        stats["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return wav, stats


def _generate_batch(
    model,
    device: str,
    prompts: List[str],
    salience: torch.Tensor,
    segment_duration: float,
    frame_rate: float,
    melody_bins: int,
    progress: bool,
    seed: int,
    drums_wav: torch.Tensor | None,
) -> Tuple[torch.Tensor, dict]:
    """Batch-generate multiple samples that share the same melody salience but use different text prompts.

    JASCO's _prepare_melody_conditions applies the same salience tensor to every item in the batch,
    so we can pass descriptions=[p1, p2, ..., pN] with a single [53, T] salience and get back [N, C, T].
    This is significantly faster than N sequential _generate_one calls.

    Returns:
        wavs: CPU tensor of shape [N, C, T] — one wav per prompt.
        stats: timing / memory stats dict.
    """
    _set_seed(seed, device)
    stats = {"seed": int(seed), "device": device, "batch_size": len(prompts)}

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    wavs = model.generate_music(
        descriptions=prompts,
        chords=None,
        drums_wav=drums_wav,
        melody_salience_matrix=salience,
        segment_duration=segment_duration,
        frame_rate=frame_rate,
        melody_bins=melody_bins,
        progress=progress,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    stats["duration_sec"] = float(t1 - t0)
    stats["duration_per_sample_sec"] = float(t1 - t0) / max(1, len(prompts))
    if device == "cuda":
        stats["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return wavs.detach().cpu(), stats


# -----------------------------
# Main
# -----------------------------
def main():
    global PROJECT_ROOT, AUDIOCRAFT_REPO_ROOT
    if PROJECT_ROOT is None:
        raise SystemExit(
            "Error: JASCO_ROOT environment variable is not set.\n"
            "Set it to the root of the cloned JASCO audiocraft repository:\n"
            "  export JASCO_ROOT=/path/to/jasco-audiocraft\n"
            "See README.md for installation instructions."
        )

    from audiocraft.models import JASCO
    from audiocraft.data.audio import audio_write


    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--input_a", type=str, required=True, help="Input audio A (melody A).")
    ap.add_argument("--input_b", type=str, required=True, help="Input audio B (melody B).")

    ap.add_argument("--offset_a", type=float, default=0.0, help="Start offset (sec) for input A segment.")
    ap.add_argument("--offset_b", type=float, default=0.0, help="Start offset (sec) for input B segment.")

    ap.add_argument("--salience_a_npz", type=str, default="", help="Optional DeepSalience npz for input A.")
    ap.add_argument("--salience_b_npz", type=str, default="", help="Optional DeepSalience npz for input B.")

    # Output
    ap.add_argument("--out_dir", type=str, default=str(SCRIPT_DIR / "melody_triplets_outputs"))
    ap.add_argument("--run_name", type=str, default="", help="Default: run_YYYYMMDD_HHMMSS")
    ap.add_argument("--triplet_index", type=int, default=1)

    # Segment & salience
    ap.add_argument("--segment_duration", type=float, default=10.0)
    ap.add_argument("--frame_rate", type=float, default=50.0)
    ap.add_argument("--melody_bins", type=int, default=53)

    # DeepSalience threshold
    ap.add_argument("--tau", type=float, default=0.3)

    # pYIN params (fallback)
    ap.add_argument("--pyin_sr", type=int, default=22050)
    ap.add_argument("--voiced_prob_thres", type=float, default=0.2)
    ap.add_argument("--f0_min_hz", type=float, default=65.0)
    ap.add_argument("--f0_max_hz", type=float, default=2000.0)
    ap.add_argument("--smooth_win", type=int, default=5)

    # Make salience more continuous
    ap.add_argument("--fill_gaps", type=int, default=0, help="Fill all-zero gaps up to N frames (0 disables).")
    ap.add_argument("--warn_ratio_below", type=float, default=0.80, help="Warn if nonzero_frame_ratio < this.")

    # Model & generation params
    ap.add_argument("--model_id", type=str, default="facebook/jasco-chords-drums-melody-1B")

    # Paper-recommended defaults: cfg_all=1.5, cfg_txt=0.5
    ap.add_argument("--cfg_all", type=float, default=1.5)
    ap.add_argument("--cfg_txt", type=float, default=0.5)
    ap.add_argument("--cfg_all_positive", type=float, default=None,
                    help="Optional override for positives only (defaults to --cfg_all).")
    ap.add_argument("--cfg_txt_positive", type=float, default=None,
                    help="Optional override for positives only (defaults to --cfg_txt).")

    # ODE / solver controls (supported kwargs depend on audiocraft version; we auto-filter)
    ap.add_argument("--ode_rtol", type=float, default=None)
    ap.add_argument("--ode_atol", type=float, default=None)
    ap.add_argument("--euler", action="store_true", help="Use Euler solver (usually faster, often lower quality).")
    ap.add_argument("--euler_steps", type=int, default=None)

    # Drums control (optional)
    ap.add_argument("--drums_mode", type=str, default="none", choices=["none", "silent"],
                    help="none: no drums conditioning; silent: condition with all-zero drums track to reduce drums.")

    # Prompts
    ap.add_argument(
        "--anchor_prompt",
        type=str,
        default="An 80s driving pop song electronic drums and synth pads in the background",
    )
    ap.add_argument(
        "--melody_lock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a short suffix to every prompt to encourage strict melody preservation and clarity.",
    )
    ap.add_argument(
        "--melody_lock_suffix",
        type=str,
        default="Main melody must be clearly audible and strictly follow the provided melody contour; no counter-melody; no improvisation.",
    )
    ap.add_argument("--positive_prompts_file", type=str, default=str(SCRIPT_DIR / "prompts.txt"))
    ap.add_argument("--positive_prompt", action="append", default=[], help="Can be repeated. Overrides defaults if provided.")
    ap.add_argument(
        "--num_positive",
        type=int,
        default=0,
        help="How many positive prompts to use. 0 means use all prompts from prompts file.",
    )

    # Seeds
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--positive_seed_mode", type=str, default="same", choices=["same", "increment"])

    # Debug helpers
    ap.add_argument("--save_salience_preview", action="store_true",
                    help="Write sine-beep previews of salience A/B into the run folder.")
    ap.add_argument("--salience_preview_sr", type=int, default=22050)
    ap.add_argument("--salience_preview_amp", type=float, default=0.2)

    # Runtime
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    input_a = Path(args.input_a)
    input_b = Path(args.input_b)
    if not input_a.exists():
        raise SystemExit(f"Input A not found: {input_a}")
    if not input_b.exists():
        raise SystemExit(f"Input B not found: {input_b}")

    chord_map = PROJECT_ROOT / "assets" / "chord_to_index_mapping.pkl"
    if not chord_map.exists():
        raise SystemExit(f"Chord map not found: {chord_map}")

    device = pick_device(args.device)
    print("device =", device)
    if device == "cuda":
        print("cuda device:", torch.cuda.get_device_name(0))

    # Prepare run folder
    out_dir = Path(args.out_dir)
    run_name = args.run_name.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    triplet_dir = run_dir / f"triplet_{args.triplet_index:04d}"
    triplet_dir.mkdir(parents=True, exist_ok=True)

    # Save input segments used
    a_seg_path = run_dir / "input_A_segment.wav"
    b_seg_path = run_dir / "input_B_segment.wav"
    sr_a, ns_a, used_offset_a = _save_input_segment(input_a, a_seg_path, args.offset_a, args.segment_duration)
    sr_b, ns_b, used_offset_b = _save_input_segment(input_b, b_seg_path, args.offset_b, args.segment_duration)
    print(f"Wrote input A segment: {a_seg_path} (sr={sr_a}, samples={ns_a}, offset_used={used_offset_a:.2f}s)")
    print(f"Wrote input B segment: {b_seg_path} (sr={sr_b}, samples={ns_b}, offset_used={used_offset_b:.2f}s)")

    # Build salience A/B
    print("Building salience A...")
    t0 = time.perf_counter()
    sal_a, sal_a_source, sal_a_stats = _build_salience(
        wav_path=input_a,
        salience_npz=args.salience_a_npz,
        segment_duration=args.segment_duration,
        frame_rate=args.frame_rate,
        melody_bins=args.melody_bins,
        offset_sec=used_offset_a,
        tau=args.tau,
        pyin_sr=args.pyin_sr,
        voiced_prob_thres=args.voiced_prob_thres,
        f0_min_hz=args.f0_min_hz,
        f0_max_hz=args.f0_max_hz,
        smooth_win=args.smooth_win,
        fill_gaps_frames=args.fill_gaps,
    )
    t1 = time.perf_counter()
    print(f"Salience A source: {sal_a_source}, shape={tuple(sal_a.shape)}, time={t1 - t0:.2f}s")
    print(f"Salience A nonzero_frame_ratio={sal_a_stats['nonzero_frame_ratio_after_fill']:.3f}"
          f" (<= {args.warn_ratio_below} => melody constraint likely weak)")

    print("Building salience B...")
    t0 = time.perf_counter()
    sal_b, sal_b_source, sal_b_stats = _build_salience(
        wav_path=input_b,
        salience_npz=args.salience_b_npz,
        segment_duration=args.segment_duration,
        frame_rate=args.frame_rate,
        melody_bins=args.melody_bins,
        offset_sec=used_offset_b,
        tau=args.tau,
        pyin_sr=args.pyin_sr,
        voiced_prob_thres=args.voiced_prob_thres,
        f0_min_hz=args.f0_min_hz,
        f0_max_hz=args.f0_max_hz,
        smooth_win=args.smooth_win,
        fill_gaps_frames=args.fill_gaps,
    )
    t1 = time.perf_counter()
    print(f"Salience B source: {sal_b_source}, shape={tuple(sal_b.shape)}, time={t1 - t0:.2f}s")
    print(f"Salience B nonzero_frame_ratio={sal_b_stats['nonzero_frame_ratio_after_fill']:.3f}"
          f" (<= {args.warn_ratio_below} => melody constraint likely weak)")

    # Save salience arrays
    sal_a_path = run_dir / "salience_A.npy"
    sal_b_path = run_dir / "salience_B.npy"
    np.save(str(sal_a_path), sal_a.numpy())
    np.save(str(sal_b_path), sal_b.numpy())
    print("Saved salience A:", sal_a_path)
    print("Saved salience B:", sal_b_path)

    if args.save_salience_preview:
        import torchaudio

        a_prev = _synth_preview_from_onehot_salience(
            salience=sal_a,
            frame_rate=args.frame_rate,
            sr=args.salience_preview_sr,
            midi_min=43,
            amp=float(args.salience_preview_amp),
        )
        b_prev = _synth_preview_from_onehot_salience(
            salience=sal_b,
            frame_rate=args.frame_rate,
            sr=args.salience_preview_sr,
            midi_min=43,
            amp=float(args.salience_preview_amp),
        )
        a_prev_path = run_dir / "salience_preview_A.wav"
        b_prev_path = run_dir / "salience_preview_B.wav"
        torchaudio.save(str(a_prev_path), torch.from_numpy(a_prev).unsqueeze(0), args.salience_preview_sr)
        torchaudio.save(str(b_prev_path), torch.from_numpy(b_prev).unsqueeze(0), args.salience_preview_sr)
        print("Wrote salience preview A:", a_prev_path)
        print("Wrote salience preview B:", b_prev_path)

    # Prompts
    anchor_prompt = args.anchor_prompt.strip()
    positive_prompts = _load_positive_prompts(args.positive_prompts_file, args.positive_prompt)
    if args.num_positive > 0:
        positive_prompts = positive_prompts[: args.num_positive]
    if not positive_prompts:
        raise ValueError("No positive prompts loaded.")

    if args.melody_lock:
        anchor_prompt = _apply_prompt_suffix(anchor_prompt, args.melody_lock_suffix)
        positive_prompts = [_apply_prompt_suffix(p, args.melody_lock_suffix) for p in positive_prompts]

    print("Anchor prompt:\n  ", anchor_prompt)
    print(f"Positive prompts ({len(positive_prompts)}):")
    for i, p in enumerate(positive_prompts, 1):
        print(f"  {i:02d}. {p}")

    if args.dry_run:
        print("Dry-run mode: exiting.")
        return

    # Load model
    print("Loading JASCO model...")
    model = JASCO.get_pretrained(
        args.model_id,
        device=device,
        chords_mapping_path=str(chord_map),
    )

    # Apply generation params (auto-filter)
    gen_kwargs = {
        "cfg_coef_all": float(args.cfg_all),
        "cfg_coef_txt": float(args.cfg_txt),
        "ode_rtol": args.ode_rtol,
        "ode_atol": args.ode_atol,
        "euler": bool(args.euler),
        "euler_steps": args.euler_steps,
    }
    filtered, skipped = _filter_supported_generate_kwargs(model, gen_kwargs)
    model.set_generation_params(**filtered)
    print("Applied generation params:", filtered)
    if skipped:
        print("Skipped unsupported generation params:", skipped)

    pos_cfg_all = float(args.cfg_all_positive) if args.cfg_all_positive is not None else float(args.cfg_all)
    pos_cfg_txt = float(args.cfg_txt_positive) if args.cfg_txt_positive is not None else float(args.cfg_txt)
    if (pos_cfg_all, pos_cfg_txt) != (float(args.cfg_all), float(args.cfg_txt)):
        print(f"Positive CFG override: cfg_all_positive={pos_cfg_all}, cfg_txt_positive={pos_cfg_txt}")

    # Move salience to device once
    sal_a = sal_a.to(torch.device(device))
    sal_b = sal_b.to(torch.device(device))

    # Optional silent drums conditioning
    drums_wav = None
    if args.drums_mode == "silent":
        drums_wav = _prepare_silent_drums(model.sample_rate, args.segment_duration, device)
        print("Using silent drums conditioning (all zeros).")

    # Write run meta
    run_meta = {
        "model_id": args.model_id,
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "segment_duration": args.segment_duration,
        "frame_rate": args.frame_rate,
        "melody_bins": args.melody_bins,
        "cfg_all": args.cfg_all,
        "cfg_txt": args.cfg_txt,
        "cfg_all_positive": args.cfg_all_positive,
        "cfg_txt_positive": args.cfg_txt_positive,
        "ode_rtol": args.ode_rtol,
        "ode_atol": args.ode_atol,
        "euler": args.euler,
        "euler_steps": args.euler_steps,
        "drums_mode": args.drums_mode,
        "seed": args.seed,
        "positive_seed_mode": args.positive_seed_mode,
        "melody_lock": bool(args.melody_lock),
        "melody_lock_suffix": str(args.melody_lock_suffix),
        "save_salience_preview": bool(args.save_salience_preview),
        "salience_preview_sr": args.salience_preview_sr,
        "salience_preview_amp": args.salience_preview_amp,
        "inputs": {
            "A_wav": str(input_a),
            "B_wav": str(input_b),
            "offset_a_requested": float(args.offset_a),
            "offset_b_requested": float(args.offset_b),
            "offset_a": float(used_offset_a),
            "offset_b": float(used_offset_b),
            "A_segment_saved": str(a_seg_path),
            "B_segment_saved": str(b_seg_path),
            "salience_A_source": sal_a_source,
            "salience_B_source": sal_b_source,
            "salience_A_npz": args.salience_a_npz,
            "salience_B_npz": args.salience_b_npz,
            "salience_A_saved": str(sal_a_path),
            "salience_B_saved": str(sal_b_path),
            "salience_A_stats": sal_a_stats,
            "salience_B_stats": sal_b_stats,
            "fill_gaps_frames": args.fill_gaps,
        },
        "prompts": {
            "anchor_prompt": anchor_prompt,
            "positive_prompts": positive_prompts,
        },
        "note": "Anchor/Negative share anchor_prompt; Anchor/Positives use melody A; Negative uses melody B.",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = run_dir / "manifest.jsonl"

    # -------- Anchor --------
    print("Generating Anchor...")
    wav_anchor, stats_anchor = _generate_one(
        model=model,
        device=device,
        prompt=anchor_prompt,
        salience=sal_a,
        segment_duration=args.segment_duration,
        frame_rate=args.frame_rate,
        melody_bins=args.melody_bins,
        progress=args.progress,
        seed=args.seed,
        drums_wav=drums_wav,
    )
    wav_anchor = wav_anchor.detach().cpu().squeeze(0)
    audio_write(str(triplet_dir / "anchor"), wav_anchor, model.sample_rate,
                strategy="loudness", loudness_compressor=True, add_suffix=True)
    anchor_path = str(triplet_dir / "anchor.wav")

    # -------- Negative (melody B, same style prompt) --------
    print("Generating Negative...")
    wav_neg, stats_neg = _generate_one(
        model=model,
        device=device,
        prompt=anchor_prompt,
        salience=sal_b,
        segment_duration=args.segment_duration,
        frame_rate=args.frame_rate,
        melody_bins=args.melody_bins,
        progress=args.progress,
        seed=args.seed,
        drums_wav=drums_wav,
    )
    wav_neg = wav_neg.detach().cpu().squeeze(0)
    audio_write(str(triplet_dir / "negative"), wav_neg, model.sample_rate,
                strategy="loudness", loudness_compressor=True, add_suffix=True)
    negative_path = str(triplet_dir / "negative.wav")

    # -------- Positives --------
    if (pos_cfg_all, pos_cfg_txt) != (float(args.cfg_all), float(args.cfg_txt)):
        pos_gen_kwargs = dict(gen_kwargs)
        pos_gen_kwargs["cfg_coef_all"] = float(pos_cfg_all)
        pos_gen_kwargs["cfg_coef_txt"] = float(pos_cfg_txt)
        filtered_pos, skipped_pos = _filter_supported_generate_kwargs(model, pos_gen_kwargs)
        model.set_generation_params(**filtered_pos)
        print("Applied POSITIVE generation params:", filtered_pos)
        if skipped_pos:
            print("Skipped unsupported POSITIVE generation params:", skipped_pos)

    print(f"Generating {len(positive_prompts)} Positives...")
    positives_info: List[dict] = []
    for k, pos_prompt in enumerate(positive_prompts, start=1):
        if args.positive_seed_mode == "increment":
            pos_seed = args.seed + k
        else:
            pos_seed = args.seed

        wav_pos, stats_pos = _generate_one(
            model=model,
            device=device,
            prompt=pos_prompt,
            salience=sal_a,
            segment_duration=args.segment_duration,
            frame_rate=args.frame_rate,
            melody_bins=args.melody_bins,
            progress=args.progress,
            seed=pos_seed,
            drums_wav=drums_wav,
        )
        wav_pos = wav_pos.detach().cpu().squeeze(0)
        name = f"positive_{k:02d}"
        audio_write(str(triplet_dir / name), wav_pos, model.sample_rate,
                    strategy="loudness", loudness_compressor=True, add_suffix=True)
        out_path = str(triplet_dir / f"{name}.wav")

        rec = {
            "type": "positive",
            "positive_index": k,
            "output_wav": out_path,
            "prompt": pos_prompt,
            "seed": int(pos_seed),
            "stats": stats_pos,
        }
        positives_info.append(rec)
        print(f"  [OK] {Path(out_path).name} | seed={pos_seed} | sec={stats_pos.get('duration_sec', -1):.2f}")

    # Manifest lines
    def _append_manifest(obj: dict) -> None:
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    _append_manifest({
        "type": "anchor",
        "output_wav": anchor_path,
        "prompt": anchor_prompt,
        "seed": int(args.seed),
        "stats": stats_anchor,
    })
    _append_manifest({
        "type": "negative",
        "output_wav": negative_path,
        "prompt": anchor_prompt,
        "seed": int(args.seed),
        "stats": stats_neg,
    })
    for rec in positives_info:
        _append_manifest(rec)

    # Triplet meta
    triplet_meta = {
        "triplet_index": args.triplet_index,
        "triplet_dir": str(triplet_dir),
        "inputs": {
            "A_wav": str(input_a),
            "B_wav": str(input_b),
            "A_offset_sec": float(used_offset_a),
            "B_offset_sec": float(used_offset_b),
            "A_segment_saved": str(a_seg_path),
            "B_segment_saved": str(b_seg_path),
            "salience_A_source": sal_a_source,
            "salience_B_source": sal_b_source,
            "salience_A_saved": str(sal_a_path),
            "salience_B_saved": str(sal_b_path),
            "salience_A_stats": sal_a_stats,
            "salience_B_stats": sal_b_stats,
        },
        "generation": {
            "model_id": args.model_id,
            "device": device,
            "cfg_all": args.cfg_all,
            "cfg_txt": args.cfg_txt,
            "cfg_all_positive": args.cfg_all_positive,
            "cfg_txt_positive": args.cfg_txt_positive,
            "ode_rtol": args.ode_rtol,
            "ode_atol": args.ode_atol,
            "euler": args.euler,
            "euler_steps": args.euler_steps,
            "drums_mode": args.drums_mode,
            "segment_duration": args.segment_duration,
            "frame_rate": args.frame_rate,
            "melody_bins": args.melody_bins,
            "seed": args.seed,
            "positive_seed_mode": args.positive_seed_mode,
            "melody_lock": bool(args.melody_lock),
            "melody_lock_suffix": str(args.melody_lock_suffix),
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
    (triplet_dir / "triplet_meta.json").write_text(json.dumps(triplet_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print("Run folder:", run_dir)
    print("Triplet folder:", triplet_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
