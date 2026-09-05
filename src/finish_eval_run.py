"""Wait for the Kaggle eval kernel, pull it, and build the submission from it.

One command so the result is usable the moment it lands instead of needing
four manual steps under time pressure. Each step prints what it did and stops
on failure rather than producing a half-built sheet.

    python src\\finish_eval_run.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
SRC = Path(__file__).parent


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"[fail] {label} exited {proc.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="dvad-eval-frames-run")
    ap.add_argument("--out", default=r"C:\dvad\models\kaggle_evalframes")
    ap.add_argument("--pred", default=r"C:\dvad\outputs\eval_pred_timed_v3.csv")
    ap.add_argument("--manifest", default=r"C:\dvad\outputs\manifest_eval.json")
    ap.add_argument("--videos", default=r"C:\dvad\data\eval_ahc\all\test\videos")
    ap.add_argument("--summaries", default=r"C:\dvad\outputs\ahc_events")
    ap.add_argument("--runtime", default="submissions/eval_submission_final.json")
    ap.add_argument("--final", default="submissions/eval_submission_v6.json")
    ap.add_argument("--mode", choices=("fill", "replace"), default="fill")
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = ap.parse_args()

    kaggle = Path(PY).with_name("kaggle.exe")
    deadline = time.time() + args.timeout_min * 60
    while True:
        proc = subprocess.run([str(kaggle), "kernels", "status", f"guptaneeraj123/{args.slug}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = (proc.stdout or "") + (proc.stderr or "")
        stamp = time.strftime("%H:%M:%S")
        if "COMPLETE" in txt:
            print(f"[{stamp}] COMPLETE", flush=True)
            break
        if "ERROR" in txt or "CANCEL" in txt:
            raise SystemExit(f"[{stamp}] kernel failed: {txt.strip()}")
        if time.time() > deadline:
            raise SystemExit(f"[{stamp}] timed out after {args.timeout_min} min")
        print(f"[{stamp}] running...", flush=True)
        time.sleep(args.poll)

    run([PY, str(SRC / "push_notebook.py"), "--pull", "--slug", args.slug, "--out", args.out],
        "pull outputs")

    frames = Path(args.out) / "eval_frames.jsonl"
    if not frames.exists():
        hits = list(Path(args.out).rglob("eval_frames.jsonl"))
        if not hits:
            raise SystemExit(f"[fail] no eval_frames.jsonl under {args.out}")
        frames = hits[0]
    print(f"[frames] {frames}")

    vlm_csv = r"C:\dvad\outputs\eval_pred_vlm.csv"
    run([PY, str(SRC / "vlm_windows.py"), "--frames", str(frames), "--pred", args.pred,
         "--out", vlm_csv, "--mode", args.mode], "rebuild events from the VLM timeline")

    raw_json = r"C:\dvad\outputs\eval_submission_v6_raw.json"
    run([PY, str(SRC / "export_arena.py"), "--manifest", args.manifest, "--pred", vlm_csv,
         "--summaries", args.summaries, "--videos", args.videos, "--out", raw_json],
        "export submission")

    run([PY, str(SRC / "merge_runtime.py"), "--events", raw_json,
         "--runtime", args.runtime, "--out", args.final], "carry measured runtime_metadata")

    print(f"\n[ready] {args.final}")


if __name__ == "__main__":
    main()
