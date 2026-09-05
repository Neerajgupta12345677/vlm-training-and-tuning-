"""How far are our Level-2/3 windows from the IoU>=0.5 gate?

D3 scored 0 matched of 8 while the CLASSES were often right, so the loss is
timing, not recognition. This prints every GT interval next to our best
same-class prediction and the IoU, so it is visible whether a window is nearly
there (widen/shift it) or nowhere near (a different problem).

    python src\\diag_iou_gap.py --sub submissions\\submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_arena import IOU_GATE, _intervals, _iou, load_gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=r"C:\dvad\data\ahc\test\ground_truth.csv")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = ap.parse_args()

    gt_by_vid, level = load_gt(Path(args.gt))
    sub = json.loads(Path(args.sub).read_text(encoding="utf-8-sig"))
    events = {p["video_id"]: p.get("events", []) for p in sub.get("predictions", [])}

    near = 0
    for vid in sorted(v for v in level if level[v] in (2, 3)):
        gt = _intervals(gt_by_vid[vid])
        if not gt:
            continue
        pred = [e for e in events.get(vid, [])
                if e.get("start_time_sec") is not None and e.get("end_time_sec") is not None]
        print(f"\n--- {vid}  (L{level[vid]})  gt={len(gt)} pred={len(pred)}")
        for gc, gs, ge in gt:
            same = [p for p in pred if p.get("class_name") == gc]
            if not same:
                classes = sorted({p.get("class_name") for p in pred}) or ["<none>"]
                print(f"  GT {gc:<32} [{gs:7.1f},{ge:7.1f}] {ge-gs:6.1f}s  "
                      f"-> NO same-class prediction (we said {classes})")
                continue
            best = max(same, key=lambda p: _iou((float(p["start_time_sec"]), float(p["end_time_sec"])), (gs, ge)))
            bs, be = float(best["start_time_sec"]), float(best["end_time_sec"])
            iou = _iou((bs, be), (gs, ge))
            verdict = "MATCH" if iou >= IOU_GATE else ("near" if iou >= 0.25 else "miss")
            if verdict == "near":
                near += 1
            print(f"  GT {gc:<32} [{gs:7.1f},{ge:7.1f}] {ge-gs:6.1f}s  "
                  f"ours [{bs:7.1f},{be:7.1f}] {be-bs:6.1f}s  IoU={iou:.3f}  {verdict}")
    print(f"\n{near} GT event(s) sit in 0.25<=IoU<0.5 - those are the cheapest to convert.")


if __name__ == "__main__":
    main()
