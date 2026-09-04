"""Measure the per-frame vehicle speed distribution for a clip.

Written because the congestion threshold was set twice by intuition and missed
twice. `congestion_crawl_speed` is a cut on `TrackState.norm_speed`, and nothing
in the repo reported what that quantity actually looks like on a clip the
organisers labelled `traffic_congestion`. This prints the distribution so the
cut can be read off the data instead of guessed.

    python src\diag_speeds.py --data_dir C:\dvad\data\ahc --videos T008,T009,T003

For each frame it reports how many vehicles are live and the 10th/25th/50th/
75th percentile of their norm_speed, then a summary of what share of vehicles
would be counted as "jammed" at several candidate thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import iter_frames_threaded, probe_video
from context_state import ContextStateTracker, TriggerConfig, ZoneMap
from detect_track import Stage1Tracker

CANDIDATE_CUTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def analyse(video: Path, args) -> dict:
    meta = probe_video(video)
    stride = 1 if meta.fps < 5.0 else min(args.stride, 2) if meta.fps < 15.0 else args.stride
    stage1 = Stage1Tracker(
        weights=args.weights, device=args.device, conf=args.conf,
        imgsz=args.imgsz, fps=meta.fps, stride=stride,
        compensate_ego_motion=False,
    )
    # Thresholds are irrelevant here; we only want the tracker's speed estimates,
    # so triggering is pushed out of reach to keep the run cheap and quiet.
    stage2 = ContextStateTracker(
        zones=ZoneMap.load(None, default_kind="driving_lane"),
        config=TriggerConfig(stop_seconds=1e9, loiter_seconds=1e9,
                             congestion_seconds=1e9, enable_collision=False),
    )
    per_frame: list[tuple[float, int, list[float]]] = []
    for frame_idx, frame in iter_frames_threaded(video, max_frames=args.max_frames, stride=stride):
        det = stage1.process(frame_idx, frame)
        stage2.update(det)
        live = [st for st in stage2.tracks.values()
                if st.is_vehicle and st.last_seen_s >= det.timestamp_s - 1.0]
        speeds = [st.norm_speed for st in live]
        per_frame.append((det.timestamp_s, len(live), speeds))

    print(f"\n{'=' * 74}")
    print(f"{video.name}  {meta.width}x{meta.height} @ {meta.fps:.1f}fps  "
          f"{meta.frame_count} frames ({meta.frame_count / max(meta.fps, 1e-6):.1f}s)  stride={stride}")
    print(f"{'=' * 74}")
    print(f"{'t(s)':>7} {'n_veh':>6} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7}")
    for t, n, speeds in per_frame:
        if not speeds:
            print(f"{t:7.2f} {n:6d} {'-':>7} {'-':>7} {'-':>7} {'-':>7}")
            continue
        p10, p25, p50, p75 = np.percentile(speeds, [10, 25, 50, 75])
        print(f"{t:7.2f} {n:6d} {p10:7.3f} {p25:7.3f} {p50:7.3f} {p75:7.3f}")

    # What share of vehicles each candidate cut would call "jammed", and for how
    # long that share stays above congestion_share. That second number is the
    # one that decides whether the rule can fire at all.
    print("\n  cut    mean_share  frames_share>=0.6  longest_run_s")
    dt = stride / max(meta.fps, 1e-6)
    summary = {}
    for cut in CANDIDATE_CUTS:
        shares = [(sum(1 for s in sp if s < cut) / len(sp)) if sp else 0.0
                  for _, _, sp in per_frame]
        above = [s >= 0.6 for s in shares]
        longest = run = 0
        for a in above:
            run = run + 1 if a else 0
            longest = max(longest, run)
        summary[cut] = (float(np.mean(shares)) if shares else 0.0, sum(above), longest * dt)
        print(f"  {cut:<6.2f} {summary[cut][0]:10.3f} {summary[cut][1]:18d} "
              f"{summary[cut][2]:14.2f}")
    veh_counts = [n for _, n, _ in per_frame]
    print(f"\n  vehicles/frame: mean {np.mean(veh_counts):.1f}  "
          f"median {np.median(veh_counts):.0f}  max {max(veh_counts)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=Path(r"C:\dvad\data\ahc"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", default="", help="comma-separated video ids, e.g. T008,T009")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    args = ap.parse_args()

    vdir = args.data_dir / args.split / "videos"
    wanted = [v.strip() for v in args.videos.split(",") if v.strip()]
    vids = [vdir / f"{v}.mp4" for v in wanted] if wanted else sorted(vdir.glob("*.mp4"))
    for v in vids:
        if not v.exists():
            print(f"[skip] {v} missing")
            continue
        analyse(v, args)


if __name__ == "__main__":
    main()
