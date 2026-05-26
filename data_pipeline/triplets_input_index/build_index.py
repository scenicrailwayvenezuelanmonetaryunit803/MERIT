#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triplets_input_index.index_builder import (
    MELODY_INDEX_PATH,
    NUM_WORKERS,
    RHYTHM_INDEX_PATH,
    TIMBRE_INDEX_PATH,
    build_melody_index,
    build_rhythm_index,
    build_timbre_index,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build prefiltered MoisesDB indexes for triplets generators.")
    ap.add_argument("task", choices=["melody", "rhythm", "timbre", "all"])
    ap.add_argument("--force", action="store_true", help="Rebuild index even if a matching JSON already exists.")
    ap.add_argument("--max-songs", type=int, default=None, help="Only scan the first N songs (sorted) for a small test index.")
    ap.add_argument(
        "--song-ids",
        nargs="*",
        default=None,
        help="Optional explicit song ids to include in the index.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If set, writes melody/rhythm/timbre_index.json under this folder.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel worker threads per builder (default: {NUM_WORKERS}, env: TRIMUS_NUM_WORKERS).",
    )
    args = ap.parse_args()

    def _path_for(name: str) -> Path | None:
        if args.output_dir is None:
            return None
        return args.output_dir / f"{name}_index.json"

    # ---- task=all: run all three builders simultaneously as subprocesses ----
    if args.task == "all":
        env = os.environ.copy()
        env["TRIMUS_NUM_WORKERS"] = str(args.workers)
        base_cmd = [
            sys.executable, str(Path(__file__).resolve()),
        ]
        if args.force:
            extra = ["--force"]
        else:
            extra = []
        if args.max_songs is not None:
            extra += ["--max-songs", str(args.max_songs)]
        if args.song_ids:
            extra += ["--song-ids"] + list(args.song_ids)
        out_dir_args = ["--output-dir", str(args.output_dir)] if args.output_dir else []
        procs: list = []
        for task_name in ["timbre", "rhythm", "melody"]:
            cmd = base_cmd + [task_name] + extra + out_dir_args
            print(f"[all] launching: {' '.join(cmd)}")
            procs.append((task_name, subprocess.Popen(cmd, env=env)))
        failed = []
        for task_name, proc in procs:
            code = proc.wait()
            if code != 0:
                failed.append(task_name)
                print(f"[all] {task_name} FAILED (exit {code})", file=sys.stderr)
            else:
                print(f"[all] {task_name} OK")
        if failed:
            sys.exit(f"[all] failed: {failed}")
        return

    # ---- single task ----
    if args.task == "melody":
        path = _path_for("melody")
        build_melody_index(
            force=args.force,
            verbose=True,
            index_path=path,
            song_ids=args.song_ids,
            max_songs=args.max_songs,
            num_workers=args.workers,
        )
        print(f"[ok] melody index -> {path or MELODY_INDEX_PATH}")
    if args.task == "rhythm":
        path = _path_for("rhythm")
        build_rhythm_index(
            force=args.force,
            verbose=True,
            index_path=path,
            song_ids=args.song_ids,
            max_songs=args.max_songs,
            num_workers=args.workers,
        )
        print(f"[ok] rhythm index -> {path or RHYTHM_INDEX_PATH}")
    if args.task == "timbre":
        path = _path_for("timbre")
        build_timbre_index(
            force=args.force,
            verbose=True,
            index_path=path,
            song_ids=args.song_ids,
            max_songs=args.max_songs,
            num_workers=args.workers,
        )
        print(f"[ok] timbre index -> {path or TIMBRE_INDEX_PATH}")


if __name__ == "__main__":
    main()
