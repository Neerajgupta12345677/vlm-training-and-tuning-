"""Score a Kaggle VLM eval dump (eval_results.jsonl) the way the arena scores L1.

The eval notebook emits one class per video. Level 1 is where that is directly
comparable to the arena's `found N/20` and false-alarm columns, so this reports
exactly those two numbers plus the per-video disagreements against whatever is
currently banked - which is what decides whether a new adapter is worth
merging.

    python src\\score_vlm_eval.py --eval C:\\dvad\\models\\kaggle_evalb4\\eval_results.jsonl ^
        --banked submissions\\submission.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TRUE = {"true", "1", "yes"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="eval_results.jsonl from the Kaggle run")
    ap.add_argument("--banked", default="submissions/submission.json")
    ap.add_argument("--gt", default=r"C:\dvad\data\ahc\test\ground_truth.csv")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = ap.parse_args()

    truth: dict[str, set[str]] = {}
    level: dict[str, int] = {}
    for r in csv.DictReader(Path(args.gt).open(encoding="utf-8-sig")):
        level[r["video_id"]] = int(r["level"])
        if str(r["is_anomaly"]).strip().lower() in TRUE:
            truth.setdefault(r["video_id"], set()).add(r["class_name"])

    rows = [json.loads(l) for l in Path(args.eval).open(encoding="utf-8") if l.strip()]
    vlm = {r["video_id"]: {r["class_name"]} - {"normal"} for r in rows}

    banked_doc = json.loads(Path(args.banked).read_text(encoding="utf-8-sig"))
    banked = {p["video_id"]: {e["class_name"] for e in p.get("events", [])}
              for p in banked_doc.get("predictions", [])}

    l1 = sorted(v for v in level if level[v] == 1)

    def tally(get) -> tuple[int, int]:
        found = fa = 0
        for v in l1:
            t = truth.get(v, set())
            pc = get(v)
            if not t:
                fa += len(pc)
            else:
                if pc & t:
                    found += 1
                fa += len(pc - t)
        return found, fa

    n_anom = sum(1 for v in l1 if truth.get(v))
    for name, get in (("new VLM eval", lambda v: vlm.get(v, set())),
                      ("banked sheet", lambda v: banked.get(v, set()))):
        f, fa = tally(get)
        print(f"{name:<14} Level 1: found {f}/{n_anom}   false alarms {fa}")

    print("\ndisagreements on Level 1:")
    n = 0
    for v in l1:
        b, w = banked.get(v, set()), vlm.get(v, set())
        if b == w:
            continue
        n += 1
        t = sorted(truth.get(v, set())) or ["normal"]
        verdict = ""
        if t != ["normal"]:
            if w & set(t) and not (b & set(t)):
                verdict = "   <-- NEW is right"
            elif b & set(t) and not (w & set(t)):
                verdict = "   <-- banked is right"
        elif w and not b:
            verdict = "   <-- NEW adds a false alarm"
        elif b and not w:
            verdict = "   <-- NEW silences a false alarm"
        print(f"  {v}  truth={t}  banked={sorted(b) or ['empty']}  "
              f"new={sorted(w) or ['empty']}{verdict}")
    if not n:
        print("  none")


if __name__ == "__main__":
    main()
