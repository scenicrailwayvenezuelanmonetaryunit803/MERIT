#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triplets_input_index.index_builder import _drums_rhythm_stats, _save_input_segment

def _pick_device(mode: str) -> str:
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_metadata(metadata_json: Path | None) -> dict[str, dict]:
    if metadata_json is None or not metadata_json.exists():
        return {}
    try:
        rows = json.loads(metadata_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read metadata JSON {metadata_json}: {e}")
    out: dict[str, dict] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("id") or row.get("track_id") or "").strip()
        if key:
            out[key] = row
    return out

def _separate_drums(
    input_audio: Path,
    output_wav: Path,
    demucs_model,
    device: str,
    force: bool = False,
) -> tuple[int, int]:
    if output_wav.exists() and not force:
        info = torchaudio.info(str(output_wav))
        return int(info.sample_rate), int(info.num_frames)

    wav, sample_rate = torchaudio.load(str(input_audio))
    wav = convert_audio(wav, sample_rate, demucs_model.samplerate, demucs_model.audio_channels)
    with torch.no_grad():
        stems = apply_model(demucs_model, wav.to(device).unsqueeze(0), device=device).squeeze(0)
    drum_stem = stems[demucs_model.sources.index("drums")]
    drum_stem = convert_audio(drum_stem.cpu(), demucs_model.samplerate, sample_rate, 1)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_wav), drum_stem, sample_rate)
    return int(sample_rate), int(drum_stem.shape[-1])

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a rhythm_index.json from a flat audio folder by separating drums with Demucs."
    )
    ap.add_argument("audio_dir", type=Path, help="Folder containing full-mix input audio files.")
    ap.add_argument("output_json", type=Path, help="Where to write rhythm_index.json.")
    ap.add_argument(
        "--dataset-name",
        type=str,
        default="audio_folder_mix",
        help="Dataset label to store in the index config.",
    )
    ap.add_argument(
        "--drums-cache-dir",
        type=Path,
        default=None,
        help="Where to store separated drum stems. Defaults to <audio_dir>_demucs_drums.",
    )
    ap.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional metadata JSON aligned by track id / filename stem.",
    )
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--force-separate", action="store_true", help="Re-run Demucs even if cached stems exist.")
    ap.add_argument("--segment-duration", type=float, default=10.0)
    ap.add_argument("--onset-sr", type=int, default=22050)
    ap.add_argument("--onset-hop", type=int, default=512)
    ap.add_argument("--min-anchor-onset-count", type=int, default=20)
    ap.add_argument("--min-anchor-rms", type=float, default=0.005)
    args = ap.parse_args()

    audio_dir = args.audio_dir.resolve()
    output_json = args.output_json.resolve()
    drums_cache_dir = (
        args.drums_cache_dir.resolve()
        if args.drums_cache_dir is not None
        else (audio_dir.parent / f"{audio_dir.name}_demucs_drums").resolve()
    )
    metadata_by_id = _load_metadata(args.metadata_json.resolve() if args.metadata_json is not None else None)
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio folder not found: {audio_dir}")

    suffixes = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes])
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 audio files in {audio_dir} to build a rhythm triplet index.")

    device = _pick_device(args.device)
    print(f"[demucs] device={device}")
    demucs_model = pretrained.get_model("htdemucs").to(device)
    demucs_model.eval()

    tmp_dir = audio_dir / ".rhythm_index_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_seg_path = tmp_dir / "drums_segment.wav"

    entries: list[dict] = []
    checked_files = 0
    try:
        for audio_path in files:
            track_id = audio_path.stem
            drums_path = drums_cache_dir / f"{track_id}_drums.wav"
            try:
                full_sr, full_ns = _separate_drums(
                    audio_path, drums_path, demucs_model=demucs_model, device=device, force=args.force_separate
                )
                sr, ns, off = _save_input_segment(drums_path, tmp_seg_path, -1.0, args.segment_duration, quiet=True)
                st = _drums_rhythm_stats(
                    tmp_seg_path,
                    args.segment_duration,
                    sr=int(args.onset_sr),
                    hop=int(args.onset_hop),
                )
                checked_files += 1
            except Exception as e:
                print(f"[skip] {audio_path.name}: {e}")
                continue

            meta = metadata_by_id.get(track_id, {})
            rec = {
                "track_id": track_id,
                "stem": "fullmix",
                "wav_rel": audio_path.name,
                "drums_wav_rel": drums_path.name,
                "offset_sec": float(off),
                "sr": int(full_sr),
                "ns": int(full_ns),
                "drums_sr": int(sr),
                "drums_ns": int(ns),
                "rms": float(st["rms"]),
                "onset_count": int(st["onset_count"]),
                "onset_rate": float(st["onset_rate"]),
            }
            if meta:
                for key_in, key_out in [
                    ("name", "title"),
                    ("artist_name", "artist_name"),
                    ("audiodownload", "audiodownload"),
                ]:
                    if meta.get(key_in):
                        rec[key_out] = meta[key_in]
                mi = meta.get("musicinfo") or {}
                tags = mi.get("tags") or {}
                if mi.get("speed"):
                    rec["speed"] = mi["speed"]
                if tags.get("genres"):
                    rec["genres"] = tags["genres"]
                if tags.get("instruments"):
                    rec["instruments"] = tags["instruments"]
                if tags.get("vartags"):
                    rec["vartags"] = tags["vartags"]

            entries.append(rec)
            print(
                f"[ok] {audio_path.name}: drums={drums_path.name} offset={off:.2f}s "
                f"onsets={int(st['onset_count'])} rms={float(st['rms']):.4f}"
            )
    finally:
        if tmp_seg_path.exists():
            tmp_seg_path.unlink()
        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

    if len(entries) < 2:
        raise SystemExit("Could not build a usable rhythm index: fewer than 2 files were processed successfully.")

    anchors = [
        x
        for x in entries
        if int(x["onset_count"]) >= int(args.min_anchor_onset_count) and float(x["rms"]) >= float(args.min_anchor_rms)
    ]
    if not anchors:
        raise SystemExit(
            "Could not build a usable rhythm index: no entry passed the minimum rhythm threshold. "
            "Try lowering --min-anchor-onset-count / --min-anchor-rms."
        )

    entries.sort(key=lambda x: (str(x["track_id"]), str(x["wav_rel"])))
    anchors.sort(key=lambda x: (-int(x["onset_count"]), -float(x["rms"]), str(x["track_id"])))

    config = {
        "dataset": str(args.dataset_name),
        "audio_root": str(audio_dir),
        "drums_cache_root": str(drums_cache_dir),
        "source_type": "fullmix",
        "condition_builder": "demucs_htdemucs",
        "segment_duration": float(args.segment_duration),
        "onset_sr": int(args.onset_sr),
        "onset_hop": int(args.onset_hop),
        "min_anchor_onset_count": int(args.min_anchor_onset_count),
        "min_anchor_rms": float(args.min_anchor_rms),
        "song_ids": [p.stem for p in files],
    }
    if args.metadata_json is not None:
        config["metadata_json"] = str(args.metadata_json.resolve())

    data = {
        "schema_version": 1,
        "generator": "rhythm",
        "built_at": time.time(),
        "config": config,
        "stats": {
            "tracks_total": len(files),
            "checked_files": int(checked_files),
            "entries_available": len(entries),
            "anchors_available": len(anchors),
        },
        "anchors": anchors,
        "entries": entries,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] rhythm index -> {output_json}")

if __name__ == "__main__":
    main()
