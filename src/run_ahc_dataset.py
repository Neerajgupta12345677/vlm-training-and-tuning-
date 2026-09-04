"""Batch-runs the pipeline over the organisers' train/ or test/ folder tree
and produces one predictions.csv in their exact submission schema.

Expected layout (from "AHC Visual Intelligence Hackathon - Training and
Public Test Data"):
    train/<class_name>/videos/*.mp4 (+ videos.csv, ground_truth.csv)
    test/videos/*.mp4 (+ videos.csv, ground_truth.csv)

Runs pipeline.py as a SUBPROCESS per video rather than importing run_one()
directly - run_one() takes a full argparse Namespace with ~30 fields, and
reconstructing that by hand risks a missing field crashing deep inside on a
field this script's author forgot existed. Shelling out reuses pipeline.py's
own tested argument parser instead, the same pattern already proven in
demo.py tonight.

    :: score ourselves against the public test set - the one with real numbers
    python src\\run_ahc_dataset.py --data_dir C:\\dvad\\data\\ahc --split test --out C:\\dvad\\outputs\\predictions.csv
    python src\\score_submission.py --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv --pred C:\\dvad\\outputs\\predictions.csv

    :: pull real (video, class, description) distillation targets from train/ -
    :: this replaces our synthetic n=15 Groq-labelled set with real in-distribution ones
    python src\\run_ahc_dataset.py --data_dir C:\\dvad\\data\\ahc --extract-labels-only --out C:\\dvad\\data\\ahc_distill_labels.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from common import OUTPUTS_DIR
from submission import build_rows, read_jsonl, write_csv

PY = sys.executable
SRC = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _read_id_map(videos_csv: Path) -> dict[str, str]:
    """video_id -> filename, from a videos.csv whose exact column names are
    not fully documented. Tries common patterns; callers must fall back to
    filename-stem-as-id if this returns empty, since guessing wrong here
    must never crash the run.
    """
    if not videos_csv.exists():
        return {}
    try:
        with videos_csv.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read {videos_csv}: {e}")
        return {}
    if not rows:
        return {}
    cols = {c.lower(): c for c in rows[0].keys()}
    id_col = next((cols[c] for c in ("video_id", "id") if c in cols), None)
    file_col = next((cols[c] for c in ("file_name", "filename", "file", "video_path", "path")
                     if c in cols), None)
    if not id_col or not file_col:
        print(f"[warn] {videos_csv} columns {list(rows[0].keys())} did not match any "
              f"expected id/filename pattern - falling back to filename-stem-as-id")
        return {}
    return {r[id_col]: r[file_col] for r in rows}


def find_videos(split_dir: Path) -> list[tuple[str, Path]]:
    """Return [(video_id, path), ...] for a test/ dir or one train/<class>/ dir."""
    videos_dir = split_dir / "videos"
    if not videos_dir.exists():
        return []
    id_map = _read_id_map(split_dir / "videos.csv")
    file_to_id = {Path(v).name: k for k, v in id_map.items()}
    out = []
    for f in sorted(videos_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        video_id = file_to_id.get(f.name, f.stem)  # fall back to filename stem
        out.append((video_id, f))
    return out


def calibrate_zones_for(video: Path, args) -> Path | None:
    """Per-video auto zone calibration, matching demo.py's pattern.

    Without this, EVERY video gets zone_kind="unknown" (lane_like, so
    stopped_vehicle still fires) but wrong_way_driving can NEVER fire -
    that rule needs a calibrated zone.flow_deg, and there is no fallback for
    it the way there is for lane_like. Across CCTV/dashcam/drone footage with
    arbitrary camera framing, one fixed zones file cannot apply to all
    videos, so this runs fresh per video and fails open (no zones) rather
    than aborting the video's run if calibration finds nothing to work with.
    """
    if args.no_zones:
        return None
    zones_path = video.with_name(video.stem + "_zones.json")
    if zones_path.exists():
        return zones_path
    cmd = [PY, str(SRC / "calibrate_zones.py"), "--auto", "--source", str(video),
          "--frames", "300", "--stride", "2", "--out", str(zones_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    return zones_path if proc.returncode == 0 and zones_path.exists() else None


def run_pipeline_on(video: Path, events_out: Path, args) -> bool:
    """One subprocess call to pipeline.py. Returns True on success."""
    zones = calibrate_zones_for(video, args)
    cmd = [
        PY, str(SRC / "pipeline.py"),
        "--source", str(video),
        "--decision", args.decision,
        "--stride", str(args.stride),
        "--stop-seconds", str(args.stop_seconds),
        "--cooldown", str(args.cooldown),
        "--loiter-seconds", str(args.loiter_seconds),
        "--crowd-count", str(args.crowd_count),
        "--wrong-way-tolerance", str(args.wrong_way_tolerance),
        "--max-calls-per-track", str(args.max_calls_per_track),
        "--duplicate-window", str(args.duplicate_window),
        "--out", str(events_out),
        "--summary-out", str(events_out.with_suffix(".summary.json")),
    ]
    if zones:
        cmd += ["--zones", str(zones)]
    if args.aerial:
        cmd.append("--aerial")
    if args.night:
        cmd.append("--night")
    if args.weights:
        cmd += ["--weights", args.weights]
    if args.backend != "mock":
        cmd += ["--backend", args.backend, "--model", args.model]
    if args.scene_sweep:
        cmd += ["--scene-sweep", str(args.scene_sweep)]
    if args.watch_for:
        cmd += ["--watch-for", args.watch_for]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=args.per_video_timeout)
    if proc.returncode != 0:
        tail = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip()][-6:]
        print(f"    [!] pipeline failed (exit {proc.returncode}):")
        for line in tail:
            print(f"        {line[:120]}")
        return False
    return True


def extract_distillation_labels(data_dir: Path, out_path: Path) -> None:
    """Pull (video, class_name, timestamps, description_summary) straight out
    of train/<class>/ground_truth.csv - these are REAL organiser-provided
    labels for distillation/fine-tuning, per the dataset doc's stated intent.
    Strictly better than our synthetic n=15 Groq-labelled set: real footage,
    real events, in-distribution with the actual test set.
    """
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"{train_dir} not found - has the training pack finished downloading?")

    rows_out = []
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        gt_csv = class_dir / "ground_truth.csv"
        if not gt_csv.exists():
            continue
        id_map = _read_id_map(class_dir / "videos.csv")
        with gt_csv.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                video_id = r.get("video_id", "")
                filename = id_map.get(video_id)
                rows_out.append({
                    "video_id": video_id,
                    "video_file": filename,  # may be None if videos.csv column names didn't match
                    "class_folder": class_dir.name,
                    "class_name": r.get("class_name", class_dir.name),
                    "is_anomaly": r.get("is_anomaly"),
                    "level": r.get("level"),
                    "start_time_sec": r.get("start_time_sec") or None,
                    "end_time_sec": r.get("end_time_sec") or None,
                    "description_summary": r.get("description_summary") or "",
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with_desc = sum(1 for r in rows_out if r["description_summary"])
    classes = sorted({r["class_name"] for r in rows_out})
    print(f"[ok] {len(rows_out)} label row(s) from {len(classes)} class folder(s) -> {out_path}")
    print(f"[ok] {with_desc} row(s) carry a non-empty description_summary "
          f"({100 * with_desc / max(len(rows_out), 1):.0f}%)")
    print(f"[ok] classes found: {', '.join(classes)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch-run the pipeline over the AHC dataset tree.")
    p.add_argument("--data_dir", required=True, help=r"e.g. C:\dvad\data\ahc")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--extract-labels-only", action="store_true",
                   help="Skip running the pipeline; just pull description_summary labels "
                        "from train/<class>/ground_truth.csv for distillation.")
    p.add_argument("--limit", type=int, default=0, help="Cap the number of videos (0 = all).")
    p.add_argument("--decision", default="rules", choices=["rules", "hybrid", "vlm"])
    p.add_argument("--aerial", action="store_true", default=True)
    p.add_argument("--no-aerial", dest="aerial", action="store_false")
    p.add_argument("--night", action="store_true")
    p.add_argument("--weights", default=None)
    p.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    p.add_argument("--model", default="moondream")
    p.add_argument("--scene-sweep", type=float, default=0.0)
    p.add_argument("--watch-for", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--per-video-timeout", type=int, default=900)
    p.add_argument("--no-zones", action="store_true",
                   help="Skip per-video auto zone calibration. Costs wrong_way_driving "
                        "entirely (needs zone.flow_deg, no fallback) but is faster.")
    # Defaults tuned DOWN from pipeline.py's own (--stop-seconds 20) - the
    # dataset doc describes "short event clips" among the training data, and a
    # 20s dwell requirement can exceed a whole short clip's length. Verified:
    # a 21.5s test clip with the anomaly starting at 2.4s produced ZERO events
    # at the pipeline default, only because 20s of continuous dwell never fit
    # before the clip ended.
    p.add_argument("--stop-seconds", type=float, default=8.0)
    p.add_argument("--cooldown", type=float, default=6.0)
    p.add_argument("--loiter-seconds", type=float, default=10.0)
    p.add_argument("--crowd-count", type=int, default=8)
    p.add_argument("--wrong-way-tolerance", type=float, default=100.0)
    p.add_argument("--max-calls-per-track", type=int, default=30)
    # Measured finding: with the LIVE default (duplicate_window_s=20), a truck
    # stopped for a whole 21.5s test clip logged exactly ONE alert, so our
    # reported end_time_sec was "when we first alerted" (10.08s) rather than
    # anywhere near the condition's real persistence - understating duration on
    # every long-running anomaly. duplicate_window_s exists specifically to
    # prevent alert-fatigue spam in the LIVE/operator-facing path; that
    # tradeoff does not apply to an offline batch run being scored on interval
    # accuracy, so it is shortened here rather than changed globally.
    p.add_argument("--duplicate-window", type=float, default=4.0)
    p.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--out", default=str(OUTPUTS_DIR / "predictions.csv"))
    p.add_argument("--events-dir", default=str(OUTPUTS_DIR / "ahc_events"))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.extract_labels_only:
        extract_distillation_labels(data_dir, Path(args.out))
        return

    split_dir = data_dir / args.split
    if not split_dir.exists():
        raise SystemExit(f"{split_dir} not found. Is the dataset downloaded and extracted yet? "
                         f"Expected {data_dir} to contain a '{args.split}' folder.")

    if args.split == "test":
        videos = find_videos(split_dir)
    else:
        videos = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            videos += find_videos(class_dir)

    if not videos:
        raise SystemExit(f"No videos found under {split_dir}. Check the folder structure matches "
                         f"the documented layout (…/videos/*.mp4).")
    if args.limit:
        videos = videos[: args.limit]

    events_dir = Path(args.events_dir)
    events_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    print(f"[plan] {len(videos)} video(s) from {split_dir}")
    print(f"[plan] decision={args.decision}  aerial={args.aerial}  night={args.night}  "
          f"weights={args.weights or 'stock'}")

    total_rows = 0
    failures = 0
    t_start = time.perf_counter()
    for i, (video_id, video_path) in enumerate(videos, 1):
        t0 = time.perf_counter()
        events_path = events_dir / f"{video_id}.jsonl"
        ok = run_pipeline_on(video_path, events_path, args)
        if not ok or not events_path.exists():
            failures += 1
            # A failed run must still produce a submission row - a missing
            # video_id in predictions.csv is worse than a wrong guess of "normal".
            rows = build_rows([], video_id, level=args.level)
        else:
            rows = build_rows(read_jsonl(events_path), video_id, level=args.level)
        write_csv(rows, out_path, append=(i > 1))
        total_rows += len(rows)
        dt = time.perf_counter() - t0
        labels = ", ".join(sorted({r["class_name"] for r in rows}))
        print(f"  [{i}/{len(videos)}] {video_id:<24} {dt:6.1f}s  -> {labels}")

    elapsed = time.perf_counter() - t_start
    print(f"\n[done] {len(videos)} video(s), {total_rows} row(s), {failures} failure(s), "
          f"{elapsed:.0f}s total -> {out_path}")
    if args.split == "test":
        gt = split_dir / "ground_truth.csv"
        if gt.exists():
            print(f"\nScore it now:\n  python src\\score_submission.py --gt {gt} --pred {out_path}")


if __name__ == "__main__":
    main()
