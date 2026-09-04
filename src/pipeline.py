"""End-to-end three-stage pipeline: detect+track -> context state -> VLM verdict.

    python src\\pipeline.py --source C:\\dvad\\data\\vehicles.mp4 --backend mock
    python src\\pipeline.py --data_dir C:\\dvad\\data --backend ollama --model moondream

Saturday's real dataset is a one-flag swap: point --data_dir at the new folder.

Each triggered event's frame is written to <outputs>/events/. That directory is
the input to src\\distill_label.py, so the fine-tuning set is drawn from the
real Stage-2 trigger distribution rather than from random frames.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    append_jsonl,
    ensure_dirs,
    frame_sequence,
    iter_frames,
    iter_frames_threaded,
    probe_video,
)
from common import AnomalyVerdict
from context_state import ContextStateTracker, TriggerConfig, ZoneMap
from detect_track import Stage1Tracker
from vlm_reason import VLMReasoner, check_ollama, combine, highlight_target, set_watch_for

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def find_videos(data_dir: Path) -> list[Path]:
    """Find video files, plus directories that are frame sequences.

    Public anomaly benchmarks ship as numbered frame folders, so a --data_dir
    holding `clip01/000001.jpg ...` is treated exactly like a set of videos.
    """
    videos = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    seq_dirs = sorted(
        d for d in [data_dir, *data_dir.rglob("*")]
        if d.is_dir() and frame_sequence(d) is not None
    )
    return videos + seq_dirs


def _tid(ev) -> str:
    """Track id for display. Scene-level events (crowd, congestion, sweep) have
    none, and f"{None:>3}" is a TypeError - which crashed the harvest path and
    the Stage-3 queue-full path, both of which are documented commands."""
    return "  -" if ev.track_id is None else str(ev.track_id)


def _log_verdict(ev, verdict, ms: float) -> None:
    flag = "ANOMALY" if verdict.anomalous else "benign "
    tid = f"{_tid(ev):>3}"
    print(f"  [{ev.timestamp_s:6.1f}s] {flag} track {tid} {ev.class_name:<10} "
          f"{ev.kind:<18} sev={verdict.severity:.2f} ({ms:6.0f}ms) {verdict.reason}")


def _alert(ev, verdict, hold: float) -> dict:
    return {
        "until_s": ev.timestamp_s + hold,
        "severity": verdict.severity,
        "reason": verdict.reason,
        "kind": ev.kind,
    }


def run_one(video: Path, args, truncate_events: bool = True) -> dict:
    meta = probe_video(video)
    print(f"\n{'=' * 70}")
    print(f"VIDEO  {video.name}  ({meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.frame_count} frames)")
    print(f"{'=' * 70}")

    stage1 = Stage1Tracker(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        fps=meta.fps,
        stride=args.stride,
        compensate_ego_motion=args.ego_motion,
    )
    stage2 = ContextStateTracker(
        zones=ZoneMap.load(args.zones, default_kind=args.default_zone),
        config=TriggerConfig(
            stop_seconds=args.stop_seconds,
            cooldown_seconds=args.cooldown,
            max_calls_per_track=args.max_calls_per_track,
            loiter_seconds=args.loiter_seconds,
            crowd_count=args.crowd_count,
            wrong_way_tolerance_deg=args.wrong_way_tolerance,
            wrong_way_min_flow_consistency=args.wrong_way_min_consistency,
            wrong_way_min_flow_samples=args.wrong_way_min_samples,
            congestion_seconds=args.congestion_seconds,
            congestion_cooldown_s=args.congestion_cooldown,
            enable_slow_vehicle=args.enable_slow_vehicle,
            # Pointless without a model to ask, so ignore it under --decision rules.
            scene_sweep_seconds=(args.scene_sweep
                                 if (args.decision != "rules" and not args.no_vlm) else 0.0),
            **({"duplicate_window_s": args.duplicate_window}
               if args.duplicate_window is not None else {}),
        ),
    )
    needs_model = not args.no_vlm and args.decision != "rules"
    stage3 = VLMReasoner(
        backend=args.backend, model=args.model, max_side=args.max_side, timeout_s=args.timeout
    ) if needs_model else None
    print(f"stage1: {stage1.model_name} on {args.device} | stage2: stop>{args.stop_seconds}s "
          f"| stage3: {'disabled' if args.no_vlm else f'{args.backend}/{args.model}'} "
          f"| decision: {'harvest' if args.no_vlm else args.decision}")

    events_path = Path(args.out) if args.out else OUTPUTS_DIR / f"events_{video.stem}.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    # With a shared --out across several videos, only the first run may truncate,
    # or each video would wipe the previous one's events.
    if truncate_events and events_path.exists():
        events_path.unlink()
    frames_dir = Path(args.frames_dir) if args.frames_dir else OUTPUTS_DIR / "events"
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out_size = (meta.width, meta.height)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        # Encoding 4K H.264 costs more than the whole pipeline (measured: 26.6 ->
        # 9.7 fps). The annotated video is a demo artifact, not the product, so
        # write it at a sane width and keep the throughput numbers honest.
        if args.save_width and meta.width > args.save_width:
            scale = args.save_width / meta.width
            out_size = (args.save_width, int(round(meta.height * scale / 2) * 2))
        writer = cv2.VideoWriter(
            str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps / max(args.stride, 1), out_size
        )

    stage1_ms: list[float] = []
    vlm_ms: list[float] = []
    verdicts: list[dict] = []
    active_alerts: dict[int, dict] = {}
    vlm_calls = 0
    dropped = 0
    pending: list = []
    vlm_pool = ThreadPoolExecutor(max_workers=args.vlm_workers) if needs_model else None
    t_start = time.perf_counter()
    t_warm = None

    def finish(fut, ev, row) -> None:
        """Land a completed Stage 3 result: fill the row, log, raise any alert."""
        try:
            if args.decision == "hybrid":
                obs, ms, _raw = fut.result()
                verdict = combine(ev.kind, ev.rule_anomalous, ev.rule_severity, obs)
                row["observation"] = obs.model_dump()
                attempts, failed = 1, False
            else:
                res = fut.result()
                verdict, ms = res.verdict, res.latency_ms
                attempts, failed = res.attempts, res.parse_failed_once
        except Exception as exc:  # noqa: BLE001 - Stage 3 must never kill a run
            verdict = AnomalyVerdict(
                anomalous=ev.rule_anomalous,
                severity=ev.rule_severity,
                reason=f"Stage 3 unavailable ({type(exc).__name__}); fell back to tracker rule.",
            )
            ms, attempts, failed = 0.0, 0, True

        vlm_ms.append(ms)
        row["verdict"] = verdict.model_dump()
        row["vlm"] = {
            "backend": stage3.backend,
            "model": stage3.model,
            "mode": args.decision,
            "latency_ms": round(ms, 1),
            "attempts": attempts,
            "parse_failed_once": failed,
        }
        _log_verdict(ev, verdict, ms)
        if verdict.anomalous:
            active_alerts[ev.track_id] = _alert(ev, verdict, args.alert_hold)
        append_jsonl(events_path, row)
        verdicts.append(row)

    reader = iter_frames_threaded if args.threaded else iter_frames
    for n_done, (frame_idx, frame) in enumerate(
        reader(video, max_frames=args.max_frames, stride=args.stride)
    ):
        if n_done == args.warmup:
            t_warm = time.perf_counter()
        t0 = time.perf_counter()
        det = stage1.process(frame_idx, frame)
        stage1_ms.append((time.perf_counter() - t0) * 1000.0)

        events = stage2.update(det)

        # Land any Stage 3 work that finished while we were decoding frames.
        if pending:
            still = []
            for item in pending:
                if item[0].done():
                    finish(*item)
                else:
                    still.append(item)
            pending = still

        # Negative sampling for the distillation set (harvest runs only).
        if args.sample_normal and n_done % args.sample_normal == 0:
            neg = stage2.sample_normal(det)
            if neg is not None:
                events = events + [neg]

        for ev in events:
            # The frame and the event row are always recorded. Stage 3's verdict
            # is optional: with --no-vlm this becomes a harvesting run that
            # produces exactly the candidate frames distill_label.py labels.
            tid_tag = ev.track_id if ev.track_id is not None else "scene"
            frame_name = f"{video.stem}_f{ev.frame_idx:06d}_t{tid_tag}.jpg"
            cv2.imwrite(str(frames_dir / frame_name), frame)
            row = {**ev.to_dict(), "video": video.name, "frame_file": frame_name, "verdict": None, "vlm": None}

            budget_left = not (args.max_vlm_calls and vlm_calls >= args.max_vlm_calls)
            if args.decision == "rules":
                # No model at all: Stage 2's arithmetic decides. Fastest, and the
                # most reliable path for the dwell-based classes.
                verdict = AnomalyVerdict(
                    anomalous=ev.rule_anomalous,
                    severity=ev.rule_severity,
                    reason=f"Rule '{ev.kind}' on measured tracker state.",
                )
                row["verdict"] = verdict.model_dump()
                row["vlm"] = {"backend": "rules", "model": "none", "latency_ms": 0.0}
                _log_verdict(ev, verdict, 0.0)
                if verdict.anomalous:
                    active_alerts[ev.track_id] = _alert(ev, verdict, args.alert_hold)

            elif stage3 is not None and budget_left:
                # Stage 3 is dispatched to a worker, never awaited inline. A 45s
                # VLM call in the frame loop dropped throughput from 15.8 to 2.5
                # fps; off-thread, Stage 1 keeps running at full rate and the
                # verdict lands a few seconds later - which is irrelevant for an
                # anomaly defined by a 20-second dwell.
                if len(pending) >= args.vlm_queue:
                    dropped += 1
                    print(f"  [{ev.timestamp_s:6.1f}s] SKIP    track {_tid(ev):>3} "
                          f"(Stage 3 queue full: {len(pending)})")
                else:
                    # A scene sweep is about the whole frame, so it must NOT be
                    # highlighted - a magenta box would tell the model to look at
                    # one spot when the point is to look everywhere.
                    bbox = None if ev.kind == "scene_sweep" else ev.bbox
                    image_b64 = stage3.prepare_image(frame, bbox)
                    if args.decision == "hybrid":
                        fut = vlm_pool.submit(stage3.observe_b64, image_b64)
                    else:  # "vlm" - the model decides everything (measured weakest)
                        fut = vlm_pool.submit(stage3.judge_b64, image_b64, ev.context)
                    pending.append((fut, ev, row))
                    vlm_calls += 1
                    continue  # row is written when the future completes
            else:
                print(f"  [{ev.timestamp_s:6.1f}s] EVENT   track {_tid(ev):>3} {ev.class_name:<10} "
                      f"{ev.kind:<18} (harvested, no verdict)")

            append_jsonl(events_path, row)
            verdicts.append(row)

        if writer is not None:
            annotated = _annotate(frame, det, stage2, active_alerts, det.timestamp_s)
            if annotated.shape[1] != out_size[0] or annotated.shape[0] != out_size[1]:
                annotated = cv2.resize(annotated, out_size, interpolation=cv2.INTER_AREA)
            writer.write(annotated)

    elapsed = time.perf_counter() - t_start  # measured BEFORE draining the queue,
    # so throughput reflects the streaming loop rather than tail latency.

    if pending:
        print(f"[stage3] waiting on {len(pending)} queued verdict(s) ...")
        for item in pending:
            finish(*item)
        pending = []
    if vlm_pool is not None:
        vlm_pool.shutdown(wait=True)
    if writer is not None:
        writer.release()

    n = len(stage1_ms)
    s1_all = np.array(stage1_ms) if n else np.array([0.0])
    # Exclude warmup: frame 0 pays ~1s of CUDA/model init and would otherwise
    # dominate the mean and understate the throughput we can actually sustain.
    s1 = s1_all[args.warmup :] if n > args.warmup else s1_all
    warm_elapsed = (time.perf_counter() - t_warm) if t_warm else elapsed
    warm_fps = len(s1) / warm_elapsed if warm_elapsed > 0 else 0.0

    # One processed frame covers `stride` source frames, so sustaining a live
    # feed only requires source_fps/stride.
    required_fps = meta.fps / max(args.stride, 1)
    feeds = warm_fps / required_fps if required_fps else 0.0

    summary = {
        "video": video.name,
        "resolution": f"{meta.width}x{meta.height}",
        "frames_processed": n,
        "stride": args.stride,
        "threaded_decode": bool(args.threaded),
        "wall_clock_s": round(elapsed, 2),
        "warm_fps": round(warm_fps, 1),
        "source_fps": round(meta.fps, 1),
        "required_fps": round(required_fps, 1),
        "realtime": bool(warm_fps >= required_fps),
        "cold_frame_ms": round(float(s1_all[0]), 1),
        "stage1_ms_mean": round(float(s1.mean()), 1),
        "stage1_ms_p95": round(float(np.percentile(s1, 95)), 1),
        "events_triggered": stage2.events_fired,
        "trigger_rate_pct": stage2.stats()["trigger_rate_pct"],
        "decision_mode": "harvest" if args.no_vlm else args.decision,
        "vlm_calls": vlm_calls,
        "vlm_async": bool(vlm_pool is not None),
        "vlm_skipped_queue_full": dropped,
        "vlm_ms_mean": round(float(np.mean(vlm_ms)), 1) if vlm_ms else None,
        "vlm_ms_p95": round(float(np.percentile(vlm_ms, 95)), 1) if vlm_ms else None,
        "anomalies": sum(1 for v in verdicts if v["verdict"] and v["verdict"]["anomalous"]),
        "feeds_per_gpu_estimate": round(feeds, 2),
        "events_file": str(events_path),
    }

    print(f"\n--- {video.name} summary ---")
    for k, v in summary.items():
        print(f"  {k:<24} {v}")
    return summary


def _annotate(frame, det, stage2, active_alerts: dict, now_s: float) -> np.ndarray:
    """Draw tracks, and highlight any track under an active anomaly alert.

    All sizes are relative to frame height. Fixed pixel widths drawn on a 4K
    frame become sub-pixel once the output is downscaled for the demo video,
    which silently erased every box and label.
    """
    out = frame.copy()
    s = out.shape[0] / 720.0  # 720p is the reference size these numbers were tuned at
    thin = max(1, round(2 * s))
    thick_alert = max(2, round(4 * s))
    font = max(0.4, 0.5 * s)
    font_thick = max(1, round(1.2 * s))

    for i, tid in enumerate(det.track_ids):
        tid = int(tid)
        x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
        st = stage2.tracks.get(tid)
        alert = active_alerts.get(tid)
        if alert and now_s <= alert["until_s"]:
            colour, thickness = (0, 0, 255), thick_alert
            label = f"! {alert['kind']} sev={alert['severity']:.2f}"
        else:
            colour, thickness = (0, 220, 0), thin
            dwell = f" {st.stationary_s:.0f}s" if st and st.stationary_s > 1 else ""
            label = f"#{tid}{dwell}"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(out, label, (x1, max(int(14 * s), y1 - int(6 * s))),
                    cv2.FONT_HERSHEY_SIMPLEX, font, colour, font_thick, cv2.LINE_AA)

    live = [a for a in active_alerts.values() if now_s <= a["until_s"]]
    if live:
        bar = int(34 * s)
        cv2.rectangle(out, (0, 0), (out.shape[1], bar), (0, 0, 140), -1)
        top = max(live, key=lambda a: a["severity"])
        cv2.putText(out, f"ALERT: {top['reason'][:90]}", (int(10 * s), int(24 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, (255, 255, 255), font_thick, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Three-stage drone anomaly pipeline, end to end.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="A single video file.")
    src.add_argument("--data_dir", help="A folder of videos (Saturday's dataset drops in here).")
    p.add_argument("--limit-videos", type=int, default=1, help="How many videos from --data_dir (0 = all).")
    # Stage 1
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--aerial", action="store_true",
                   help="Nadir/high-altitude preset: imgsz 1280, conf 0.10. Measured on "
                        "VisDrone, stock settings (640/0.25) recall only 0.152 of aerial "
                        "objects and 0.008 of small ones; this lifts overall recall to "
                        "0.413 and still runs at 41 fps on 1080p.")
    p.add_argument("--no-ego-motion", dest="ego_motion", action="store_false", default=True,
                   help="Disable camera-motion compensation. Only sensible for a genuinely "
                        "fixed camera - on drone footage this makes every dwell-based rule "
                        "silently stop firing. Kept so the A/B stays measurable.")
    p.add_argument("--night", action="store_true",
                   help="Low-light preset: drops --conf to 0.20. Measured on simulated "
                        "night footage, that recovers detection counts to ABOVE the "
                        "daylight baseline. Deliberately does no image enhancement - "
                        "CLAHE cost 23.7->4.0 fps on 4K and detected FEWER objects.")
    p.add_argument("--imgsz", type=int, default=640)
    # Stage 2
    p.add_argument("--zones", default=None, help="Zone calibration JSON from calibrate_zones.py.")
    p.add_argument("--default-zone", default="unknown", help="Zone kind when no polygon matches.")
    p.add_argument("--stop-seconds", type=float, default=20.0)
    p.add_argument("--cooldown", type=float, default=30.0)
    p.add_argument("--max-calls-per-track", type=int, default=3)
    p.add_argument("--loiter-seconds", type=float, default=25.0,
                   help="A person stationary this long (outside a sidewalk) is loitering.")
    p.add_argument("--crowd-count", type=int, default=8,
                   help="Live person tracks in view above this count triggers crowd_density.")
    p.add_argument("--watch-for", default=None,
                   help="Comma-separated event types to look for that have NO hand-coded "
                        "rule, e.g. \"fallen tree, livestock on road, crowd surge\". Goes "
                        "into the VLM prompt and is accepted for escalation, so a new event "
                        "type needs no code, no rule and no retraining. This is the "
                        "open-vocabulary path the brief asks for.")
    p.add_argument("--scene-sweep", type=float, default=0.0,
                   help="Ask the VLM about the whole frame every N seconds, independent of "
                        "any tracked object. This is the ONLY path to static conditions the "
                        "tracker cannot see - flooding, debris, spills, fire. Needs a VLM "
                        "backend; 12 is a reasonable value. 0 = off.")
    p.add_argument("--enable-slow-vehicle", action="store_true",
                   help="Flag vehicles crawling relative to surrounding traffic. OFF by "
                        "default: its thresholds were tuned against one oblique clip and "
                        "it costs 1 false positive on the aerial ground-truth run. Turn on "
                        "once you have footage you can validate it against.")
    p.add_argument("--wrong-way-tolerance", type=float, default=135.0,
                   help="Degrees of heading deviation from a lane's calibrated flow before "
                        "a vehicle counts as wrong-way. Only applied when the lane's flow "
                        "calibration passes the consistency floors below.")
    p.add_argument("--wrong-way-min-consistency", type=float, default=0.55,
                   help="Circular resultant (0-1) a lane's calibrated flow must reach before "
                        "wrong-way may fire in it. 0 disables the gate.")
    p.add_argument("--wrong-way-min-samples", type=int, default=40,
                   help="Motion observations a lane's flow must be averaged from before "
                        "wrong-way may fire in it.")
    p.add_argument("--congestion-seconds", type=float, default=12.0,
                   help="Seconds of sustained stationary traffic before congestion fires. "
                        "Must be well under the clip length or the rule can never trigger.")
    p.add_argument("--congestion-cooldown", type=float, default=60.0)
    p.add_argument("--duplicate-window", type=float, default=None,
                   help="Seconds during which an overlapping box will not re-alert. "
                        "Set 0 on HARVEST runs: suppressing repeat alerts is right for an "
                        "operator, but the same vehicle at 5s/9s/13s dwell makes three "
                        "genuinely different training samples.")
    # Stage 3
    p.add_argument("--backend", default="mock", choices=["ollama", "mock"])
    p.add_argument("--model", default="moondream")
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--decision", default="hybrid", choices=["hybrid", "rules", "vlm"],
                   help="hybrid (default): Stage 2 rules decide, the VLM observes and may "
                        "escalate on a visible hazard. rules: no model. vlm: the model "
                        "decides everything (measured least reliable on small models).")
    p.add_argument("--no-vlm", action="store_true", help="Stages 1-2 only, harvest frames.")
    p.add_argument("--max-vlm-calls", type=int, default=0, help="Cap total VLM calls (0 = unlimited).")
    p.add_argument("--vlm-workers", type=int, default=1,
                   help="Stage 3 worker threads. 1 is right for a single 4GB GPU - "
                        "concurrent calls just contend for the same VRAM.")
    p.add_argument("--vlm-queue", type=int, default=4,
                   help="Max Stage 3 calls in flight; further events are skipped rather "
                        "than queued unboundedly.")
    p.add_argument("--sample-normal", type=int, default=0,
                   help="Every Nth frame, also emit a normally-behaving track as a labelled "
                        "negative. Use for building a balanced distillation set (0 = off).")
    # Output
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame. Dwell-based anomalies do not need full frame rate.")
    p.add_argument("--warmup", type=int, default=5, help="Frames excluded from timing stats.")
    p.add_argument("--threaded", action="store_true", default=True,
                   help="Overlap video decode with inference (default on).")
    p.add_argument("--no-threaded", dest="threaded", action="store_false")
    p.add_argument("--save", default=None, help="Annotated .mp4 output path.")
    p.add_argument("--save-width", type=int, default=1280,
                   help="Downscale the annotated video to this width (0 = keep source res).")
    p.add_argument("--out", default=None, help="Events .jsonl output path.")
    p.add_argument("--frames-dir", default=None, help="Where event frames are saved.")
    p.add_argument("--alert-hold", type=float, default=3.0, help="Seconds an alert stays drawn.")
    p.add_argument("--summary-out", default=None, help="Write the run summary JSON here.")
    args = p.parse_args()

    ensure_dirs()

    if args.watch_for:
        terms = set_watch_for(args.watch_for.split(","))
        print(f"[open-vocab] also watching for: {', '.join(terms)}")
        if args.decision == "rules" or args.no_vlm:
            print("[open-vocab] NOTE: needs a VLM to answer - use --decision hybrid, "
                  "and --scene-sweep to look for these without a triggering object.")

    if args.aerial:
        if args.imgsz == 640:
            args.imgsz = 1280
        if args.conf == 0.3:
            args.conf = 0.10
        print(f"[aerial] imgsz={args.imgsz}, conf={args.conf} "
              "(small aerial objects vanish at 640/0.25)")

    if args.night and args.conf == 0.3:  # only if the user did not set --conf themselves
        args.conf = 0.20
        print("[night] confidence threshold lowered to 0.20 for low-light footage")

    if args.backend == "ollama" and not args.no_vlm and args.decision != "rules":
        ok, models = check_ollama()
        if not ok:
            print("[warn] Ollama unreachable - falling back to the mock backend so the run still completes.")
            args.backend = "mock"
        elif not any(args.model in m for m in models):
            print(f"[warn] model {args.model!r} not installed (have: {models}) - falling back to mock.")
            args.backend = "mock"

    if args.source:
        videos = [Path(args.source)]
    else:
        data_dir = Path(args.data_dir)
        videos = find_videos(data_dir)
        if not videos:
            raise SystemExit(f"No videos found under {data_dir}")
        if args.limit_videos:
            videos = videos[: args.limit_videos]
        print(f"[data] {len(videos)} video(s) selected from {data_dir}")

    summaries = [run_one(v, args, truncate_events=(i == 0)) for i, v in enumerate(videos)]

    if len(summaries) > 1:
        print("\n=== all videos ===")
        print(f"  total events    : {sum(s['events_triggered'] for s in summaries)}")
        print(f"  total anomalies : {sum(s['anomalies'] for s in summaries)}")
    out = Path(args.summary_out) if args.summary_out else OUTPUTS_DIR / "run_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nsummary written: {out}")


if __name__ == "__main__":
    main()
