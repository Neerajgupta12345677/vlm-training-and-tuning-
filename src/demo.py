"""One-command demo driver. Point it at footage; it does the rest.

    python src\\demo.py --source <their_clip.mp4>
    python src\\demo.py --source <folder_of_frames>
    python src\\demo.py --source <clip> --night          # low light
    python src\\demo.py --source <clip> --quick          # first 300 frames only

Exists so nobody is assembling command-line flags in front of judges. It
calibrates zones, runs the pipeline, writes an annotated video, and prints the
headline numbers in a readable block. Every step degrades gracefully: if zone
calibration fails the run still proceeds (Stage 2 treats unmapped areas as
lane-like), and any step that fails says so plainly instead of aborting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import DATA_DIR, OUTPUTS_DIR

PY = sys.executable
SRC = Path(__file__).resolve().parent


def run(args_list: list[str], label: str) -> tuple[bool, str]:
    """Run a step, returning (ok, combined output). Never raises."""
    print(f"\n{'=' * 66}\n  {label}\n{'=' * 66}")
    t0 = time.perf_counter()
    proc = subprocess.run([PY, *args_list], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  [!] step failed after {dt:.1f}s - continuing anyway")
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        for line in tail:
            print(f"      {line[:110]}")
        return False, out
    print(f"  done in {dt:.1f}s")
    return True, out


def main() -> None:
    p = argparse.ArgumentParser(description="One-command demo driver.")
    p.add_argument("--source", required=True, help="Video file or folder of frames.")
    p.add_argument("--data_dir", default=str(DATA_DIR), help="Kept for interface parity.")
    p.add_argument("--aerial", action="store_true", default=True,
                   help="Aerial preset (default ON - drone footage needs it).")
    p.add_argument("--no-aerial", dest="aerial", action="store_false")
    p.add_argument("--night", action="store_true", help="Low-light preset.")
    p.add_argument("--stop-seconds", type=float, default=20.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="Cap at 300 frames for a fast pass.")
    p.add_argument("--skip-zones", action="store_true", help="Skip zone calibration.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip the annotated video. Encoding it costs real throughput, so "
                        "use this when you want to quote the true fps / feeds-per-GPU.")
    p.add_argument("--ground-truth", default=None, help="Optional *_ground_truth.json to score against.")
    p.add_argument("--outdir", default=str(OUTPUTS_DIR))
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem or src.name

    events = outdir / f"demo_{stem}.jsonl"
    summary = outdir / f"demo_{stem}_summary.json"
    annotated = outdir / f"demo_{stem}_annotated.mp4"
    zones = src.with_name(f"{stem}_zones.json") if src.is_file() else src.parent / f"{stem}_zones.json"

    t_start = time.perf_counter()
    print(f"\nDEMO: {src}")

    # 1. Zones. Optional by design - the pipeline works without them, it just
    #    loses the parking/shoulder distinction.
    if not args.skip_zones and not zones.exists():
        ok, _ = run([str(SRC / "calibrate_zones.py"), "--auto", "--source", str(src),
                     "--frames", "300", "--stride", "2"],
                    "1/3  Calibrating zones from observed motion")
        if not ok:
            print("      (no zones - unmapped areas are treated as lane-like, rules still fire)")
    elif zones.exists():
        print(f"\n1/3  Using existing zones: {zones.name}")
    else:
        print("\n1/3  Zone calibration skipped")

    # 2. Pipeline.
    cmd = [str(SRC / "pipeline.py"), "--source", str(src),
           "--decision", "rules",
           "--stop-seconds", str(args.stop_seconds),
           "--stride", str(args.stride),
           "--out", str(events), "--summary-out", str(summary)]
    if not args.no_video:
        cmd += ["--save", str(annotated), "--save-width", "1280"]
    if zones.exists():
        cmd += ["--zones", str(zones)]
    if args.aerial:
        cmd.append("--aerial")
    if args.night:
        cmd.append("--night")
    if args.quick:
        cmd += ["--max-frames", "300"]

    ok, out = run(cmd, "2/3  Running the pipeline")
    alerts = [l.strip() for l in out.splitlines() if "ANOMALY" in l]

    # 3. Optional scoring.
    gt = Path(args.ground_truth) if args.ground_truth else None
    eval_out = ""
    if gt and gt.exists():
        _, eval_out = run([str(SRC / "eval.py"), "--ground-truth", str(gt),
                           "--predictions", str(events), "--run-summary", str(summary)],
                          "3/3  Scoring against ground truth")
    else:
        print("\n3/3  No ground truth given - skipping scoring")

    # ---- headline block ----
    s = {}
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            s = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        except Exception:  # noqa: BLE001
            pass

    print("\n" + "=" * 66)
    print("  RESULTS")
    print("=" * 66)
    if s:
        rt = "YES" if s.get("realtime") else "no"
        print(f"  resolution        {s.get('resolution', '?')}   stride {s.get('stride', '?')}")
        print(f"  throughput        {s.get('warm_fps', '?')} fps   (need {s.get('required_fps', '?')})   real-time: {rt}")
        print(f"  per frame         {s.get('stage1_ms_mean', '?')} ms mean / {s.get('stage1_ms_p95', '?')} ms p95")
        print(f"  FEEDS PER GPU     ~{s.get('feeds_per_gpu_estimate', '?')}")
        if not args.no_video:
            # Do not let the demo undersell the system: encoding the annotated
            # video costs more than the whole pipeline on 4K footage.
            print("                    ^ INCLUDES annotated-video encoding, which costs")
            print("                      more than the pipeline itself. Re-run with")
            print("                      --no-video for the true throughput figure.")
        print(f"  frames -> VLM     {s.get('trigger_rate_pct', '?')}%   <- the cost argument")
        print(f"  events            {s.get('events_triggered', 0)}   anomalies: {s.get('anomalies', 0)}")
    else:
        print("  (no summary written - the pipeline step likely failed)")

    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts[:8]:
            print(f"    {a[:104]}")
        if len(alerts) > 8:
            print(f"    ... and {len(alerts) - 8} more")

    for line in eval_out.splitlines():
        if any(k in line for k in ("detected ", "best IoU", "FALSE POSITIVES", "detection_latency")):
            print(f"  {line.strip()}")

    print(f"\n  annotated video   {annotated}")
    print(f"  events jsonl      {events}")
    print(f"\n  total elapsed     {time.perf_counter() - t_start:.1f}s")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
