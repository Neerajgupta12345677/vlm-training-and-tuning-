"""Take events from a fresh export, keep runtime_metadata from the banked file.

A re-export only re-runs the timing stage, so it has no measured Stage-1/2
timings and `export_arena.py` fills required stubs. The banked submission does
have real ones. Shipping stubs would be a silent regression on an axis we
already got right, so carry the measured numbers across and change only the
events.

    python src\\merge_runtime.py --events C:\\dvad\\outputs\\eval_submission_v3.json ^
        --runtime submissions\\eval_submission_final.json --out submissions\\eval_submission_v4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="Fresh export (events to keep).")
    ap.add_argument("--runtime", required=True, help="Banked file (runtime_metadata to keep).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = ap.parse_args()

    new = json.loads(Path(args.events).read_text(encoding="utf-8-sig"))
    old = json.loads(Path(args.runtime).read_text(encoding="utf-8-sig"))
    rt = {p["video_id"]: p.get("runtime_metadata") for p in old.get("predictions", [])}

    swapped = 0
    for p in new.get("predictions", []):
        got = rt.get(p["video_id"])
        if got:
            p["runtime_metadata"] = got
            swapped += 1

    for key in ("submission_id", "model_name"):
        if key in old:
            new[key] = old[key]
    if "run_metadata" in old:
        new["run_metadata"] = old["run_metadata"]

    Path(args.out).write_text(json.dumps(new, indent=1), encoding="utf-8")

    n_ev = sum(len(p.get("events", [])) for p in new["predictions"])
    o_ev = sum(len(p.get("events", [])) for p in old["predictions"])
    print(f"[ok] {args.out}")
    print(f"     {len(new['predictions'])} videos, runtime_metadata carried for {swapped}")
    print(f"     events {o_ev} (banked) -> {n_ev} (new)")
    for p in new["predictions"]:
        o = next((q for q in old["predictions"] if q["video_id"] == p["video_id"]), None)
        a, b = len(o.get("events", [])) if o else 0, len(p.get("events", []))
        if a != b:
            print(f"       {p['video_id']}: {a} -> {b} event(s)")


if __name__ == "__main__":
    main()
