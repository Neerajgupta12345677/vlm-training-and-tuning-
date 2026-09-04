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

from common import OUTPUTS_DIR, probe_video
from submission import build_rows, read_jsonl, write_csv

PY = sys.executable
SRC = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

# Classes the motion rules may contribute under --label-source hybrid, provided
# the appearance classifier was not trained to emit them. Intersected with the
# live checkpoint's class list at runtime rather than assumed, so retraining
# with a wider head automatically narrows what the rules are allowed to add.
RULE_ONLY_CLASSES = {"stalled_or_broken_down_vehicle"}


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
    """Return [(video_id, path), ...] for a test/ dir or one train/<class>/ dir.

    videos.csv is the authority for ids. Files missing from disk are still
    returned (path may not exist) so the submission gets a `normal` fallback
    row instead of a silent gap — the public test pack we received is missing
    T030.mp4 even though videos.csv and ground_truth.csv list it.
    """
    videos_dir = split_dir / "videos"
    id_map = _read_id_map(split_dir / "videos.csv")
    file_to_id = {Path(v).name: k for k, v in id_map.items()}
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    if videos_dir.exists():
        for f in sorted(videos_dir.iterdir()):
            if f.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            video_id = file_to_id.get(f.name, f.stem)  # fall back to filename stem
            out.append((video_id, f))
            seen.add(video_id)
    for vid, rel in id_map.items():
        if vid in seen:
            continue
        path = split_dir / rel
        if not path.exists():
            path = videos_dir / Path(rel).name
        out.append((vid, path))
    return out


def level_aware_thresholds(duration_s: float) -> dict[str, float]:
    """Dwell windows scaled to how much footage there actually is.

    A fixed 12s congestion window on a 5.8s clip is arithmetically identical to
    having no congestion rule at all, and that is what the public test measured:
    T009 was called `normal` while the detector was returning 45.8 objects per
    frame including 204 cars. Every Level-1 clip in the set is 5.7-5.8s, so the
    stopped (8s), loitering (10s) and congestion (12s) rules could never fire on
    any of them regardless of detection quality.
    """
    if duration_s < 8.0:
        return {
            "stop_seconds": max(1.5, duration_s * 0.35),
            "loiter_seconds": max(2.0, duration_s * 0.45),
            "congestion_seconds": max(2.0, duration_s * 0.40),
            "congestion_cooldown": duration_s,
            "duplicate_window": min(4.0, duration_s * 0.5),
            "cooldown": max(1.0, duration_s * 0.3),
        }
    if duration_s < 30.0:
        return {
            "stop_seconds": 5.0,
            "loiter_seconds": 8.0,
            "congestion_seconds": 8.0,
            "congestion_cooldown": 30.0,
            "duplicate_window": 4.0,
            "cooldown": 6.0,
        }
    return {
        "stop_seconds": 8.0,
        "loiter_seconds": 10.0,
        "congestion_seconds": 12.0,
        "congestion_cooldown": 60.0,
        "duplicate_window": 4.0,
        "cooldown": 6.0,
    }


def adaptive_stride(fps: float, requested: int) -> int:
    """Never skip frames on footage that barely has any.

    T021-T024 are 896x448 at 1.9 fps - 30-38 frames total. At --stride 2 the
    pipeline saw 19 samples, which is not enough for ByteTrack to hold an
    identity across the clip, so no dwell-based rule could accumulate state.
    """
    if fps < 5.0:
        return 1
    if fps < 15.0:
        return min(requested, 2)
    return requested


def load_appearance(args):
    """The Stage 1.5 classifier, or None. Never fatal: a missing model must
    degrade to the motion-only pipeline, not abort a scoring run."""
    if not args.appearance:
        return None
    try:
        from appearance_classifier import DEFAULT_WEIGHTS, AppearanceClassifier

        weights = args.appearance_weights or DEFAULT_WEIGHTS
        clf = AppearanceClassifier(weights, threshold=args.appearance_threshold)
        print(f"[stage1.5] appearance classifier: {clf.classes} @ {args.appearance_threshold}")
        return clf
    except Exception as e:  # noqa: BLE001
        print(f"[stage1.5] disabled ({e})")
        return None


def appearance_rows(clf, video: Path, video_id: str, duration_s: float, level: int) -> list[dict]:
    """Submission rows from the appearance classifier, or []."""
    verdict = clf.classify_video(video)
    if verdict is None:
        return []
    desc = (f"{verdict['class_name'].replace('_', ' ')} visible in frame "
            f"(appearance classifier, confidence {verdict['confidence']:.2f}).")
    # Short clips are the organisers' Level 1 shape: one row, no timestamps.
    # Localising a 6-second clip adds nothing and risks a worse temporal IoU
    # than leaving the interval empty, which is what their own GT does.
    if duration_s < 8.0:
        return [{"video_id": video_id, "level": level, "is_anomaly": True,
                 "class_name": verdict["class_name"],
                 "start_time_sec": "", "end_time_sec": "", "description_summary": desc}]
    windows = clf.classify_windows(video)
    if not windows:
        return [{"video_id": video_id, "level": level, "is_anomaly": True,
                 "class_name": verdict["class_name"],
                 "start_time_sec": "", "end_time_sec": "", "description_summary": desc}]
    return [{"video_id": video_id, "level": level, "is_anomaly": True,
             "class_name": w["class_name"],
             "start_time_sec": w["start_time_sec"], "end_time_sec": w["end_time_sec"],
             "description_summary": f"{w['class_name'].replace('_', ' ')} visible "
                                    f"(confidence {w['confidence']:.2f})."}
            for w in windows]


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

    th = {
        "stop_seconds": args.stop_seconds,
        "loiter_seconds": args.loiter_seconds,
        "congestion_seconds": args.congestion_seconds,
        "congestion_cooldown": args.congestion_cooldown,
        "duplicate_window": args.duplicate_window,
        "cooldown": args.cooldown,
    }
    stride = args.stride
    if args.adaptive:
        try:
            meta = probe_video(video)
            duration = meta.frame_count / meta.fps if meta.fps else 0.0
            if duration > 0:
                th = level_aware_thresholds(duration)
            stride = adaptive_stride(meta.fps, args.stride)
        except Exception as e:  # noqa: BLE001 - a probe failure must not kill the video
            print(f"    [warn] probe failed ({e}); using fixed thresholds")

    cmd = [
        PY, str(SRC / "pipeline.py"),
        "--source", str(video),
        "--decision", args.decision,
        "--stride", str(stride),
        "--stop-seconds", f"{th['stop_seconds']:.2f}",
        "--cooldown", f"{th['cooldown']:.2f}",
        "--loiter-seconds", f"{th['loiter_seconds']:.2f}",
        "--congestion-seconds", f"{th['congestion_seconds']:.2f}",
        "--congestion-cooldown", f"{th['congestion_cooldown']:.2f}",
        "--crowd-count", str(args.crowd_count),
        "--wrong-way-tolerance", str(args.wrong_way_tolerance),
        "--wrong-way-min-consistency", str(args.wrong_way_min_consistency),
        "--wrong-way-min-samples", str(args.wrong_way_min_samples),
        "--max-calls-per-track", str(args.max_calls_per_track),
        "--duplicate-window", f"{th['duplicate_window']:.2f}",
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
    p.add_argument("--wrong-way-tolerance", type=float, default=135.0)
    p.add_argument("--wrong-way-min-consistency", type=float, default=0.55)
    p.add_argument("--wrong-way-min-samples", type=int, default=40)
    p.add_argument("--congestion-seconds", type=float, default=12.0)
    p.add_argument("--congestion-cooldown", type=float, default=60.0)
    p.add_argument("--adaptive", action="store_true", default=True,
                   help="Scale dwell thresholds and stride to each clip's duration/fps.")
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    p.add_argument("--videos", default=None,
                   help="Comma-separated video_ids to run (e.g. T018,T009). Default: all.")
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
    p.add_argument("--appearance", action="store_true", default=True,
                   help="Run the Stage 1.5 appearance classifier (fire/smoke/flood/debris) "
                        "before the motion pipeline. Silently skipped if unavailable.")
    p.add_argument("--no-appearance", dest="appearance", action="store_false")
    p.add_argument("--appearance-weights", default=None)
    # 0.72 was tuned against a 7-class head. An 11-class head spreads the
    # probability mass over more outputs, so the same absolute cut rejects
    # almost everything. Pick this with src\tune_appearance.py, which optimises
    # the real scorer over dumped probabilities rather than guessing.
    p.add_argument("--appearance-threshold", type=float, default=0.30)
    p.add_argument("--label-source", default="hybrid",
                   choices=["hybrid", "appearance", "rules"],
                   help="Who owns the class label. 'appearance': the classifier alone "
                        "(fast - no motion pipeline). 'rules': motion pipeline alone. "
                        "'hybrid' (default): the classifier labels the clip, and the motion "
                        "rules may only add classes the classifier does not model.")
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
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",") if v.strip()}
        videos = [(vid, p) for vid, p in videos if vid in wanted]
        missing = wanted - {vid for vid, _ in videos}
        if missing:
            print(f"[warn] --videos ids not found: {sorted(missing)}")
        if not videos:
            raise SystemExit(f"None of --videos {sorted(wanted)} matched.")
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

    appearance = load_appearance(args)

    total_rows = 0
    failures = 0
    t_start = time.perf_counter()
    for i, (video_id, video_path) in enumerate(videos, 1):
        t0 = time.perf_counter()
        events_path = events_dir / f"{video_id}.jsonl"
        if not video_path.exists():
            print(f"  [{i}/{len(videos)}] {video_id:<24} MISSING FILE -> normal")
            rows = build_rows([], video_id, level=args.level)
            write_csv(rows, out_path, append=(i > 1))
            total_rows += len(rows)
            continue

        # Stage 1.5 first: it is ~0.3s per clip and answers the classes the
        # motion rules structurally cannot. On a short clip a confident
        # appearance verdict IS the whole story, so the motion pipeline is
        # skipped - it costs 20-40s and can only add a contradictory label.
        app_rows: list[dict] = []
        duration_s = 0.0
        if appearance is not None:
            try:
                meta = probe_video(video_path)
                duration_s = meta.frame_count / meta.fps if meta.fps else 0.0
            except Exception:  # noqa: BLE001
                duration_s = 0.0
            try:
                app_rows = appearance_rows(appearance, video_path, video_id,
                                           duration_s, args.level)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] appearance classifier failed: {e}")

        # The classifier alone, when asked for it: no motion pipeline at all.
        # This is also the honest economics story - MobileNetV3-Small is ~0.3s
        # per clip against 20-40s for detect+track over the same video.
        if args.label_source == "appearance" or (app_rows and duration_s and duration_s < 8.0):
            rows = app_rows or build_rows([], video_id, level=args.level)
            write_csv(rows, out_path, append=(i > 1))
            total_rows += len(rows)
            dt = time.perf_counter() - t0
            labels = ", ".join(sorted({r["class_name"] for r in rows}))
            print(f"  [{i}/{len(videos)}] {video_id:<24} {dt:6.1f}s  -> {labels} [appearance]")
            continue

        ok = run_pipeline_on(video_path, events_path, args)
        summary_path = events_path.with_suffix(".summary.json")
        if not ok or not (events_path.exists() or summary_path.exists()):
            # A genuine failure: the pipeline exited non-zero, or produced no
            # output at all. A clean run over a video with nothing to report
            # writes only the summary and NO .jsonl - that is a correct
            # `normal`, not a crash, and counting it inflated the reported
            # failure count to 22/34 on a run where nothing actually failed.
            failures += 1
            # A failed run must still produce a submission row - a missing
            # video_id in predictions.csv is worse than a wrong guess of "normal".
            rows = build_rows([], video_id, level=args.level)
        else:
            events = read_jsonl(events_path) if events_path.exists() else []
            rows = build_rows(events, video_id, level=args.level)
        if args.label_source == "hybrid" and appearance is not None:
            # The classifier owns the label; the rules may only ADD a class it
            # was not trained to emit. Measured justification: on the classes
            # both can produce, the rules were worse - collision precision 0.33,
            # congestion structurally unable to fire (see src\diag_speeds.py),
            # and wrong-way either spraying false positives or gated silent.
            # stalled_or_broken_down_vehicle is the exception in both
            # directions: 4 training videos is too few to model, and it is the
            # rules' single reliable true positive on the public test set.
            addable = {c for c in RULE_ONLY_CLASSES if c not in appearance.labels}
            rule_rows = [r for r in rows if r["class_name"] in addable]
            rows = (app_rows + rule_rows) or build_rows([], video_id, level=args.level)
        elif app_rows:
            # Drop a bare `normal` placeholder from the motion side - the
            # appearance verdict is a positive finding and a `normal` row
            # alongside it would contradict it in the same video.
            rows = [r for r in rows if r["class_name"] != "normal"]
            rows = app_rows + rows
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
