"""Build a PARTIAL arena upload, and score the sheet it would produce.

submission.pdf: "A file only updates the videos it mentions. Your other
answers stay exactly as they were." So a risky change to one video does not
have to be uploaded alongside a good change to another - and since a later
upload replaces the score outright with no best-of, scoping the file is the
only way to take a win without also taking a loss.

This takes the CURRENTLY BANKED submission (what the arena already has), a
CANDIDATE submission (what we just built), and a list of video ids to patch.
It writes the patch file, and scores the merged sheet so the effect of the
upload is known before it costs a run.

    python src\\make_patch.py --banked submissions\\submission.json ^
        --candidate C:\\dvad\\outputs\\submission_v4.json --ids T026,T027,T028 ^
        --out submissions\\patch_l2.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from score_arena import load_gt, score_level1, score_timed_level


def _by_id(doc: dict) -> dict[str, dict]:
    return {p["video_id"]: p for p in doc.get("predictions", [])}


def _score(events: dict, gt_by_vid: dict, level: dict) -> tuple[float, dict]:
    by_level: dict[int, list[str]] = {}
    for v, lv in level.items():
        by_level.setdefault(lv, []).append(v)
    parts, total = {}, 0.0
    d1 = score_level1(sorted(by_level.get(1, [])), gt_by_vid, events)
    parts[1] = d1
    total += d1["marks"]
    for lv in (2, 3):
        if not by_level.get(lv):
            continue
        r = score_timed_level(lv, sorted(by_level[lv]), gt_by_vid, events)
        parts[lv] = r
        total += r["marks"]
    return total, parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banked", required=True, help="What the arena already has.")
    ap.add_argument("--candidate", required=True, help="Newly built submission.")
    ap.add_argument("--ids", required=True, help="Comma-separated video ids to patch.")
    ap.add_argument("--gt", default=r"C:\dvad\data\ahc\test\ground_truth.csv")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    banked = json.loads(Path(args.banked).read_text(encoding="utf-8-sig"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8-sig"))
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    b_by, c_by = _by_id(banked), _by_id(cand)
    missing = [i for i in ids if i not in c_by]
    if missing:
        raise SystemExit(f"candidate has no entry for: {', '.join(missing)}")

    gt_by_vid, level = load_gt(Path(args.gt))
    ev_before = {v: p.get("events", []) for v, p in b_by.items()}
    ev_after = dict(ev_before)
    for i in ids:
        ev_after[i] = c_by[i].get("events", [])

    before, pb = _score(ev_before, gt_by_vid, level)
    after, pa = _score(ev_after, gt_by_vid, level)

    print(f"patching {len(ids)} video(s): {', '.join(ids)}\n")
    print(f"{'video':<8}{'matched':>18}{'false alarms':>16}")
    for i in ids:
        lv = level.get(i)
        if lv == 1:
            print(f"  {i:<6}  Level 1 - class/anomaly only")
            continue
        db = next((d for d in pb[lv]["detail"] if d["video_id"] == i), None)
        da = next((d for d in pa[lv]["detail"] if d["video_id"] == i), None)
        if db and da:
            print(f"{i:<8}{db['matched']:>8} -> {da['matched']:<7}"
                  f"{db['false_alarms']:>8} -> {da['false_alarms']:<7}"
                  f"{'  WORSE' if (da['matched'] < db['matched'] or da['false_alarms'] > db['false_alarms']) else '  ok'}")

    print(f"\nsheet total: {before:.1f} -> {after:.1f}  ({after-before:+.1f})")
    for lv in (1, 2, 3):
        if lv in pb and lv in pa:
            print(f"  L{lv}: {pb[lv]['marks']:.2f} -> {pa[lv]['marks']:.2f}")
    if after < before:
        print("\n[STOP] this patch lowers the sheet. There is no best-of; do not upload.")

    out_doc = copy.deepcopy(banked)
    out_doc["predictions"] = [copy.deepcopy(c_by[i]) for i in ids]
    Path(args.out).write_text(json.dumps(out_doc, indent=1), encoding="utf-8")
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"\n[ok] {args.out}  ({size_kb:.1f} KB, {len(ids)} video(s))")


if __name__ == "__main__":
    main()
