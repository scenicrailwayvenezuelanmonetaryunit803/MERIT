#!/usr/bin/env python3
"""
Generate DeepSalience-compatible npz from a WAV file (JASCO-friendly CQT-based pseudo-salience).

Output npz format (expected by build_salience_from_deepsalience_npz):
  - salience: [F, T] float32  (frequency x time)
  - freqs:    [F] float32      (bin center frequencies in Hz)
  - times:    [T] float32      (frame times in seconds)

Why this version (vs naive CQT magnitude)?
- Your previous npz often got "all-zero after tau" in JASCO pipeline.
- This script makes the salience scale robust:
  1) restrict pitch range to JASCO bins (G2..B6 by default),
  2) harmonic-only preprocessing to reduce drums,
  3) voiced gating + per-frame normalization so tau won't wipe everything,
  4) optional onehot mode to output a cleaner melody contour.

NOTE:
This is NOT the official DeepSalience multi-F0 model. It's a practical fallback.
For best fidelity on complex polyphonic mixtures, use the original DeepSalience repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


EPS = 1e-8


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, EPS))


def _synth_preview_from_salience(
    salience: np.ndarray,  # [F, T]
    freqs: np.ndarray,     # [F]
    hop: int,
    sr: int,
    voiced_mask: Optional[np.ndarray] = None,  # [T] bool
    amp: float = 0.2,
) -> np.ndarray:
    """
    Make a simple sine preview of the extracted pitch contour (argmax per frame).
    """
    F, T = salience.shape
    idx = np.argmax(salience, axis=0)  # [T]
    frame_max = np.max(salience, axis=0)  # [T]

    if voiced_mask is None:
        voiced_mask = frame_max > 0

    out = np.zeros(T * hop, dtype=np.float32)
    phase = 0.0
    t = (np.arange(hop, dtype=np.float32) / sr)

    for i in range(T):
        if not voiced_mask[i]:
            phase = 0.0
            continue
        f = float(freqs[idx[i]])
        if f <= 0:
            phase = 0.0
            continue
        w = 2.0 * np.pi * f
        seg = amp * np.sin(w * t + phase)
        phase = (phase + w * hop / sr) % (2.0 * np.pi)
        out[i * hop : (i + 1) * hop] = seg.astype(np.float32, copy=False)

    return out


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """
    Try torchaudio first (common in your project), fallback to soundfile/scipy if needed.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    try:
        import torch
        import torchaudio
        wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        torchaudio.save(str(path), wav, sr)
        return
    except Exception:
        pass

    try:
        import soundfile as sf
        sf.write(str(path), audio, sr)
        return
    except Exception:
        pass

    try:
        from scipy.io.wavfile import write as wavwrite
        wav_i16 = (audio * 32767.0).astype(np.int16)
        wavwrite(str(path), sr, wav_i16)
        return
    except Exception as e:
        print(f"[WARN] Could not write preview wav: {path} ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="WAV -> DeepSalience-style npz (JASCO-friendly CQT-based).")
    ap.add_argument("wav", type=str, help="Input WAV path.")
    ap.add_argument("-o", "--output", type=str, default="", help="Output npz path (default: <stem>_jasco_salience.npz).")

    ap.add_argument("--offset", type=float, default=0.0, help="Start time in seconds (same meaning as your JASCO script).")
    ap.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (recommend match JASCO segment_duration).")

    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate (match JASCO: 22050).")
    ap.add_argument("--hop", type=int, default=256, help="Hop length (match JASCO: 256).")

    # Pitch range aligned to JASCO bins (G2..B6)
    ap.add_argument("--midi_min", type=int, default=43, help="Default 43 (G2).")
    ap.add_argument("--midi_max", type=int, default=95, help="Default 95 (B6).")
    ap.add_argument("--bins_per_semitone", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Frequency resolution. 1=semitone bins (53 bins total). 2=half-semitone (105 bins), etc.")

    # Preprocess
    ap.add_argument("--use_harmonic", action="store_true", help="Use harmonic component (recommended).")
    ap.add_argument("--freq_weight", type=float, default=0.0,
                    help="Optional: multiply magnitudes by (freq^(-freq_weight)) to prefer lower freqs. Try 0.5 if octave errors happen.")

    # Voiced gating (avoid normalizing noise-only frames)
    ap.add_argument("--voiced_db", type=float, default=-40.0,
                    help="Frame is voiced if frame_max >= voiced_db below global max. Example: -40.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Exponent after per-frame normalization. >1 makes peaks sharper (try 1.5~2.0).")
    ap.add_argument(
        "--confidence_scale",
        type=float,
        default=0.0,
        help="Extra confidence scaling based on peakiness (second-max). 0 disables.",
    )

    # Output mode
    ap.add_argument(
        "--mode",
        type=str,
        default="onehot",
        choices=["onehot", "scores"],
        help="scores: DeepSalience-like map (use --tau 0.3 in generator). onehot: already argmaxed (binary-ish).",
    )
    ap.add_argument("--neighbor", type=float, default=0.2,
                    help="For onehot mode: also set +/-1 bin to this value (smoother after resample).")

    # Simple octave correction (helpful for piano/guitar harmonics)
    ap.add_argument("--octave_correction", action="store_true", help="Try to correct octave errors by checking 1-2 octaves down.")
    ap.add_argument("--octave_ratio", type=float, default=0.6,
                    help="Accept lower-octave bin if its energy >= peak * octave_ratio.")

    # Debug / preview
    ap.add_argument("--preview_wav", type=str, default="", help="Optional: write a beep preview wav of extracted contour.")
    ap.add_argument("--debug", action="store_true", help="Print extra stats to help choose tau/params.")

    args = ap.parse_args()

    try:
        import librosa
    except ImportError as e:
        raise SystemExit("librosa is required. pip install librosa") from e

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise SystemExit(f"WAV not found: {wav_path}")

    y, sr = librosa.load(str(wav_path), sr=args.sr, mono=True)
    if y.size == 0:
        raise SystemExit("Empty audio.")

    # Slice segment
    t_start = int(round(args.offset * sr))
    n_samples = int(round(args.duration * sr))
    if t_start >= y.shape[0]:
        raise SystemExit(f"offset {args.offset}s is beyond audio length.")

    y = y[t_start : t_start + n_samples]
    if y.shape[0] < n_samples:
        y = librosa.util.fix_length(y, size=n_samples)

    if args.use_harmonic:
        y = librosa.effects.harmonic(y)

    # CQT parameters aligned to MIDI range
    bins_per_octave = 12 * args.bins_per_semitone
    fmin = librosa.midi_to_hz(args.midi_min)
    n_bins = (args.midi_max - args.midi_min) * args.bins_per_semitone + 1

    # CQT magnitude: [F, T]
    cqt = librosa.cqt(
        y,
        sr=args.sr,
        hop_length=args.hop,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    mag = np.abs(cqt).astype(np.float32, copy=False)
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave).astype(np.float32)

    # Optional low-freq preference (helps reduce octave errors sometimes)
    if args.freq_weight > 0:
        w = (freqs + EPS) ** (-float(args.freq_weight))
        mag = mag * w[:, None].astype(np.float32)

    F, T = mag.shape
    frame_max = mag.max(axis=0) + EPS
    global_max = float(frame_max.max() + EPS)
    frame_db = _db(frame_max / global_max)  # <=0
    voiced = frame_db >= float(args.voiced_db)

    # Per-frame normalize (only for voiced frames)
    sal = np.zeros_like(mag, dtype=np.float32)
    sal[:, voiced] = (mag[:, voiced] / frame_max[voiced]).astype(np.float32, copy=False)

    # Sharpen peaks if desired
    if args.gamma != 1.0:
        sal[:, voiced] = np.power(sal[:, voiced], float(args.gamma)).astype(np.float32, copy=False)

    active = voiced.copy()
    if float(args.confidence_scale) > 0 and np.any(voiced):
        sal_voiced = sal[:, voiced]  # [F, T_v], max is 1.0 per frame
        second = np.partition(sal_voiced, -2, axis=0)[-2]  # [T_v]
        ratio = 1.0 / (second + EPS)
        conf = np.clip(float(args.confidence_scale) * (ratio - 1.0), 0.0, 1.0).astype(np.float32, copy=False)
        sal[:, voiced] *= conf[None, :]
        active = (sal.max(axis=0) > 0)

    # Output mode
    if args.mode == "onehot":
        out = np.zeros_like(sal, dtype=np.float32)
        peak = np.argmax(sal, axis=0)  # [T]

        if args.octave_correction:
            # Try 1-2 octaves down (in bins)
            step_oct = bins_per_octave
            for t in range(T):
                if not voiced[t]:
                    continue
                p = int(peak[t])
                p_val = float(sal[p, t])
                best = p

                for k in (1, 2):
                    q = p - k * step_oct
                    if q < 0:
                        continue
                    q_val = float(sal[q, t])
                    if q_val >= p_val * float(args.octave_ratio):
                        best = q
                        p_val = q_val
                peak[t] = best

        # One-hot (+ neighbors)
        for t in range(T):
            if not active[t]:
                continue
            p = int(peak[t])
            out[p, t] = 1.0
            nb = float(args.neighbor)
            if nb > 0:
                if p - 1 >= 0:
                    out[p - 1, t] = max(out[p - 1, t], nb)
                if p + 1 < F:
                    out[p + 1, t] = max(out[p + 1, t], nb)

        salience = out
    else:
        salience = sal

    # times: keep "absolute" timeline (so your JASCO script can slice using same offset)
    times = (float(args.offset) + (np.arange(T, dtype=np.float32) * float(args.hop)) / float(args.sr)).astype(np.float32)

    # Stats
    voiced_ratio = float(voiced.mean())
    active_ratio = float(active.mean())
    if args.debug:
        smax = float(salience.max())
        p95 = float(np.percentile(salience, 95))
        p99 = float(np.percentile(salience, 99))
        print("=== DEBUG ===")
        print(f"mag: shape={mag.shape}, global_max={global_max:.6f}")
        print(f"frame_db: min={frame_db.min():.2f}dB, max={frame_db.max():.2f}dB, voiced_db={args.voiced_db}dB")
        print(f"voiced_ratio={voiced_ratio:.3f}  active_ratio={active_ratio:.3f}  (demo salience often ~0.6-0.75)")
        print(f"salience stats: max={smax:.3f}, p95={p95:.3f}, p99={p99:.3f}, mode={args.mode}")
        print("Suggested tau for your JASCO script: start with --tau 0.3 (legacy default).")
        print("==============")

    # Write npz
    out_path = Path(args.output.strip() or (str(wav_path.with_suffix("")) + "_jasco_salience.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, salience=salience.astype(np.float32), freqs=freqs, times=times)
    print(
        f"Wrote {out_path} (salience {salience.shape}, freqs {freqs.shape}, times {times.shape}, "
        f"voiced_ratio={voiced_ratio:.3f}, active_ratio={active_ratio:.3f})"
    )

    # Optional preview wav
    if args.preview_wav.strip():
        prev = _synth_preview_from_salience(salience, freqs, hop=args.hop, sr=args.sr, voiced_mask=active)
        prev_path = Path(args.preview_wav.strip())
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(prev_path, prev, args.sr)
        print(f"Wrote preview wav: {prev_path}")


if __name__ == "__main__":
    main()
