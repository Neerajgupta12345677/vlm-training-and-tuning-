"""Score a submission.json the way the arena does, per level, locally.

`score_submission.py` reports macro-F1 over a CSV. That is NOT the arena metric
and it disagrees with the leaderboard in both directions: it rewards multi-label
guessing (the arena takes ONE label per Level-1 video) and it grades temporal
overlap at IoU>=0.3 (the arena gates at 0.5 and counts every unmatched fragment
against you). Tuning against macro-F1 is how you end up 3rd with 0.61 locally.

This reads the actual submission.json plus the level column in ground_truth.csv
and reports each level the way submission.pdf describes it.

WHAT IS EXACT vs WHAT IS ESTIMATED - read this before trusting a number:
  - EXACT: the counts. Matched events, false alarms, per-video verdicts, which
    video failed and why. These reconstruct the live leaderboard breakdown of
    the 49.2 run precisely (D1 8 FA, D2 3 matched of 18 with 7 FA, D3 0 of 8
    with 6 FA), so the mechanics below are the arena's mechanics.
  - ESTIMATED: the Level-2/3 marks. submission.pdf says a scored video is "a
    weighted mix of did you alert, matched events, and how well your timings
    line up" with "timing weighs more at Level 3", but never publishes the
    weights. L2_WEIGHTS/L3_WEIGHTS below are a reading of that sentence, not
    the truth. The D1 formula IS published (half anomaly accuracy, half class
    accuracy) yet still lands 19.8/25 where the arena said 17.5 - so there is
    a detail in their implementation we do not have.

Use the counts to decide what to fix. Treat the marks as directional only.

    python src\\score_arena.py --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv ^
        --sub submissions\\submission.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

IOU_GATE = 0.5          # the arena's temporal match threshold, published
MARKS = {1: 25.0, 2: 35.0, 3: 40.0}

# Estimated, not published. See the module docstring.
L2_WEIGHTS = {"alert": 0.2, "match": 0.5, "timing": 0.3}
L3_WEIGHTS = {"alert": 0.2, "match": 0.4, "timing": 0.4}


def _truthy(s) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path: Path) -> tuple[dict[str, list[dict]], dict[str, int]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    by_vid: dict[str, list[dict]] = defaultdict(list)
    level: dict[str, int] = {}
    for r in rows:
        by_vid[r["video_id"]].append(r)
        level[r["video_id"]] = int(r["level"])
    return by_vid, level


def _intervals(rows: list[dict]) -> list[tuple[str, float, float]]:
    """(class, start, end) for GT rows that actually carry a window."""
    out = []
    for r in rows:
        s, e = (r.get("start_time_sec") or "").strip(), (r.get("end_time_sec") or "").strip()
        if not s or not e:
            continue
        try:
            out.append((r["class_name"], float(s), float(e)))
        except ValueError:
            continue
    return out


def match_events(gt: list[tuple[str, float, float]],
                 pred: list[dict]) -> tuple[list[float], int]:
    """Greedy one-to-one match at IoU>=0.5 with the class agreeing.

    Returns the IoU of each matched GT event and the count of unmatched
    predictions. "Several partial fragments for one real event don't help. At
    most one can match, and the rest count against you" - so a prediction can
    be consumed by only one GT event, and leftovers are false alarms.
    """
    cands = []
    for pi, p in enumerate(pred):
        s, e = p.get("start_time_sec"), p.get("end_time_sec")
        if s is None or e is None:
            continue
        for gi, (gc, gs, ge) in enumerate(gt):
            if p.get("class_name") != gc:
                continue
            iou = _iou((float(s), float(e)), (gs, ge))
            if iou >= IOU_GATE:
                cands.append((iou, gi, pi))
    cands.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[float] = []
    for iou, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append(iou)
    return matched, len(pred) - len(used_p)


def score_level1(vids: list[str], gt_by_vid: dict, events: dict) -> dict:
    anom_ok = cls_ok = 0
    wrong = []
    for v in vids:
        g = gt_by_vid[v]
        is_anom = any(_truthy(r["is_anomaly"]) for r in g)
        gt_cls = {r["class_name"] for r in g if r["class_name"] != "normal"} if is_anom else set()
        pred_cls = {e["class_name"] for e in events.get(v, [])}
        a = bool(gt_cls) == bool(pred_cls)
        c = bool(gt_cls & pred_cls) if gt_cls else not pred_cls
        anom_ok += a
        cls_ok += c
        if not (a and c):
            wrong.append({"video_id": v, "truth": sorted(gt_cls) or ["normal"],
                          "pred": sorted(pred_cls) or ["<empty>"],
                          "anomaly_ok": a, "class_ok": c})
    n = len(vids) or 1
    frac = 0.5 * anom_ok / n + 0.5 * cls_ok / n
    return {"videos": len(vids), "anomaly_ok": anom_ok, "class_ok": cls_ok,
            "false_alarms": len(wrong), "fraction": frac,
            "marks": frac * MARKS[1], "wrong": wrong}


def score_timed_level(lv: int, vids: list[str], gt_by_vid: dict, events: dict) -> dict:
    w = L2_WEIGHTS if lv == 2 else L3_WEIGHTS
    per_video, total_gt, total_matched, total_fa = [], 0, 0, 0
    detail = []
    for v in vids:
        gt = _intervals(gt_by_vid[v])
        pred = events.get(v, [])
        if not gt:
            # Ground truth normal: predict nothing = 1, predict anything = 0.
            s = 0.0 if pred else 1.0
            total_fa += len(pred)
            per_video.append(s)
            detail.append({"video_id": v, "gt_events": 0, "pred": len(pred),
                           "matched": 0, "false_alarms": len(pred), "score": s})
            continue
        matched, fa = match_events(gt, pred)
        total_gt += len(gt)
        total_matched += len(matched)
        total_fa += fa
        alert = 1.0 if pred else 0.0
        match_frac = len(matched) / len(gt)
        timing = (sum(matched) / len(matched)) if matched else 0.0
        s = w["alert"] * alert + w["match"] * match_frac + w["timing"] * timing
        per_video.append(s)
        detail.append({"video_id": v, "gt_events": len(gt), "pred": len(pred),
                       "matched": len(matched), "false_alarms": fa,
                       "mean_iou_matched": round(timing, 3), "score": round(s, 3)})
    frac = sum(per_video) / len(per_video) if per_video else 0.0
    return {"videos": len(vids), "gt_events": total_gt, "matched": total_matched,
            "false_alarms": total_fa, "fraction": frac,
            "marks": frac * MARKS[lv], "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=r"C:\dvad\data\ahc\test\ground_truth.csv")
    ap.add_argument("--sub", required=True, help="submission.json to score")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gt_by_vid, level = load_gt(Path(args.gt))
    sub = json.loads(Path(args.sub).read_text(encoding="utf-8-sig"))
    events = {p["video_id"]: p.get("events", []) for p in sub.get("predictions", [])}

    missing = sorted(set(level) - set(events))
    if missing:
        print(f"[warn] {len(missing)} GT video(s) absent from the submission - the arena "
              f"scores those as normal: {', '.join(missing[:8])}")

    by_level = defaultdict(list)
    for v, lv in level.items():
        by_level[lv].append(v)

    total = 0.0
    result = {}
    d1 = score_level1(sorted(by_level[1]), gt_by_vid, events)
    result["level1"] = d1
    total += d1["marks"]
    print(f"=== Level 1  ({d1['videos']} videos, {MARKS[1]:.0f} marks) ===")
    print(f"  anomaly-vs-normal correct : {d1['anomaly_ok']}/{d1['videos']}")
    print(f"  class correct             : {d1['class_ok']}/{d1['videos']}")
    print(f"  wrong videos              : {d1['false_alarms']}")
    print(f"  marks (published formula) : {d1['marks']:.2f} / {MARKS[1]:.0f}")
    for wv in d1["wrong"]:
        print(f"    {wv['video_id']}: truth={wv['truth']} pred={wv['pred']}")

    for lv in (2, 3):
        vids = sorted(by_level.get(lv, []))
        if not vids:
            continue
        r = score_timed_level(lv, vids, gt_by_vid, events)
        result[f"level{lv}"] = r
        total += r["marks"]
        print(f"\n=== Level {lv}  ({r['videos']} videos, {MARKS[lv]:.0f} marks) ===")
        print(f"  GT events                 : {r['gt_events']}")
        print(f"  matched (class + IoU>=0.5): {r['matched']}")
        print(f"  false alarms              : {r['false_alarms']}")
        print(f"  marks (ESTIMATED weights) : {r['marks']:.2f} / {MARKS[lv]:.0f}")
        for d in r["detail"]:
            flag = "" if d["matched"] == d["gt_events"] and not d["false_alarms"] else "  <-- losing marks"
            print(f"    {d['video_id']}: gt={d['gt_events']} pred={d['pred']} "
                  f"matched={d['matched']} fa={d['false_alarms']}{flag}")

    result["total"] = total
    print(f"\n=== TOTAL (directional) : {total:.1f} / 100 ===")
    print("  Counts above are exact arena mechanics. L2/L3 marks use estimated")
    print("  weights - trust the matched/false-alarm counts, not the decimal.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[ok] wrote {args.json_out}")


if __name__ == "__main__":
    main()
