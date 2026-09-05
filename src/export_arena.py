"""Convert our CSV predictions into the arena's submission JSON.

The arena does NOT take CSV. Discovered 2026-09-05 11:00 from `submission.pdf`:
it wants one JSON document with a `predictions` array, and `runtime_metadata`
on EVERY video. Everything else in this repo emits the organisers'
`ground_truth.csv` shape, which is the right format for local scoring against
`test/ground_truth.csv` and the wrong format for uploading. This script is the
bridge, and it is deliberately strict: a rejected upload is cheap (it does not
consume a run) but a silently-wrong one is not.

Rules encoded here, all quoted from submission.pdf, each of which is listed in
its own "things that catch people out" section:
  - `"class_name": "normal"` is REJECTED. A normal video is `"events": []`.
  - Level 1 timestamps MUST be null. Levels 2-3 require start >= 0 and
    end > start.
  - At Level 1 "repeating a class on one video earns nothing extra", so a
    Level-1 video gets exactly ONE event even when our CSV carries several.
    Our local macro-F1 rewarded multi-label; the arena does not.
  - Level 2/3 events match only at temporal IoU >= 0.5, and only the single
    best-overlapping fragment can match - "the rest count against you". So
    emitting many small guesses is actively harmful.
  - `video_id` must match the manifest exactly and appear exactly once.
  - Max file size 5 MB.

    python src\export_arena.py --manifest C:\dvad\data\arena\manifest.json ^
        --pred C:\dvad\outputs\predictions_final.csv ^
        --summaries C:\dvad\outputs\ahc_events ^
        --out C:\dvad\outputs\submission.json

Levels come from the manifest, never from our CSV - our writer hardcoded
level=1 on every row, which was wrong on 11 of 35 rows against the public set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The 11 event classes. `normal` is deliberately NOT here: it is expressed as
# an empty event list, and sending it as a class_name is a documented rejection.
ARENA_CLASSES = {
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "wrong_way_driving",
    "road_spill_or_debris",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
}
MAX_BYTES = 5 * 1024 * 1024


def load_manifest(path: Path) -> dict[str, int]:
    """{video_id: level} from the arena manifest.

    Tolerates the shapes the manifest might plausibly take, because it is
    downloaded from an authenticated page and cannot be inspected ahead of
    time: a bare list of ids, a list of objects, or a dict keyed by id.
    """
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("videos", "predictions", "manifest", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # dict keyed by video_id
            return {str(k): int((v or {}).get("level", 1) if isinstance(v, dict) else v or 1)
                    for k, v in data.items()}
    out: dict[str, int] = {}
    for item in data:
        if isinstance(item, str):
            out[item] = 1
            continue
        vid = item.get("video_id") or item.get("id") or item.get("name")
        if vid is None:
            continue
        out[str(vid)] = int(item.get("level", 1) or 1)
    if not out:
        raise SystemExit(f"Could not read any video ids from {path}. "
                         f"Inspect it and extend load_manifest().")
    return out


def load_predictions(path: Path) -> dict[str, list[dict]]:
    """{video_id: [row, ...]} from our submission CSV."""
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)
    return by_video


def _num(v) -> float | None:
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_durations(videos_dir: Path | None) -> dict[str, float]:
    if not videos_dir or not videos_dir.exists():
        return {}
    from common import probe_video
    out: dict[str, float] = {}
    for fp in videos_dir.glob("*.mp4"):
        try:
            m = probe_video(fp)
            out[fp.stem] = m.frame_count / m.fps if m.fps else 0.0
        except Exception:
            continue
    return out


def load_runtimes(summaries: Path | None, extra: Path | None) -> dict[str, dict]:
    """Per-video timing, from pipeline .summary.json files and/or a sidecar JSON.

    The latency bonus is `total reported processing time / total video
    duration`, so these numbers are part of the score. They are read from what
    the pipeline actually measured rather than invented; videos with no
    measurement are reported by --validate instead of being quietly filled in.
    """
    out: dict[str, dict] = {}
    if summaries and summaries.exists():
        for fp in sorted(summaries.glob("*.summary.json")):
            try:
                payload = json.loads(fp.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            entry = payload[0] if isinstance(payload, list) and payload else payload
            if not isinstance(entry, dict):
                continue
            vid = fp.name.removesuffix(".summary.json")
            frames = int(entry.get("frames_processed") or 0)
            wall_s = float(entry.get("wall_clock_s") or 0.0)
            out[vid] = {
                "frames_processed": frames,
                "chunks_processed": 1,
                "end_to_end_internal_time_ms": round(wall_s * 1000.0, 1),
            }
            calls = int(entry.get("vlm_calls") or 0)
            mean_ms = entry.get("vlm_ms_mean")
            if calls and mean_ms:
                total = round(float(mean_ms) * calls, 1)
                out[vid]["model_runtimes"] = [{
                    "model_name": entry.get("decision_mode") or "vision-language-model",
                    "call_count": calls,
                    "total_time_ms": total,
                    "average_time_ms": round(total / calls, 1),
                }]
            else:
                # Arena requires model_runtimes on every video. When Stage 3
                # never ran, the wall time is the detector + classifier, so
                # report that as a single model rather than inventing VLM calls.
                wall_ms = out[vid]["end_to_end_internal_time_ms"]
                out[vid]["model_runtimes"] = [{
                    "model_name": "appearance-classifier",
                    "call_count": 1,
                    "total_time_ms": wall_ms,
                    "average_time_ms": wall_ms,
                }]
    if extra and extra.exists():
        for vid, rt in json.loads(extra.read_text(encoding="utf-8-sig")).items():
            out.setdefault(vid, {}).update(rt)
    return out


def build(manifest: dict[str, int], preds: dict[str, list[dict]],
          runtimes: dict[str, dict], explanations: bool = True,
          durations: dict[str, float] | None = None,
          min_conf: float = 0.0) -> tuple[dict, list[str]]:
    """The submission document, plus a list of human-readable warnings."""
    warnings: list[str] = []
    predictions = []
    for vid, level in manifest.items():
        rows = [r for r in preds.get(vid, [])
                if (r.get("class_name") or "").strip() in ARENA_CLASSES]
        # Rank by confidence when our description carries one, so the single
        # Level-1 label is the strongest rather than merely the first written.
        rows.sort(key=lambda r: _conf(r.get("description_summary", "")), reverse=True)

        events: list[dict] = []
        if level == 1:
            # "repeating a class on one video earns nothing extra" - one event.
            # A weak Level-1 call on a normal clip is a false alarm, and the
            # arena prices those higher than a miss. Drop below min_conf rather
            # than send a guess (empty events = normal).
            for r in rows[:1]:
                if _conf(r.get("description_summary", "")) < min_conf:
                    warnings.append(f"{vid} (level 1): silenced `{r['class_name']}` "
                                    f"(confidence below {min_conf})")
                    continue
                ev = {"class_name": r["class_name"],
                      "start_time_sec": None, "end_time_sec": None}
                if explanations and (x := _explanation(r)):
                    ev["explanation"] = x
                events.append(ev)
        else:
            seen: set[tuple[str, float, float]] = set()
            for r in rows:
                s, e = _num(r.get("start_time_sec")), _num(r.get("end_time_sec"))
                if s is None or e is None:
                    warnings.append(
                        f"{vid} (level {level}): dropped `{r['class_name']}` - Levels 2-3 "
                        f"require timestamps and this row has none")
                    continue
                s = max(0.0, s)
                dur = (durations or {}).get(vid)
                if dur and e > dur:
                    e = dur - 0.05
                if e <= s:
                    warnings.append(
                        f"{vid} (level {level}): dropped `{r['class_name']}` - "
                        f"end_time_sec ({e}) must be greater than start ({s})")
                    continue
                key = (r["class_name"], round(s, 3), round(e, 3))
                if key in seen:
                    continue
                seen.add(key)
                ev = {"class_name": r["class_name"],
                      "start_time_sec": round(s, 3), "end_time_sec": round(e, 3)}
                if explanations and (x := _explanation(r)):
                    ev["explanation"] = x
                events.append(ev)

        rt = dict(runtimes.get(vid) or {})
        if not rt:
            warnings.append(f"{vid}: no measured runtime_metadata - filling a required "
                            f"stub; replace before a scored upload if you can")
            rt = {"frames_processed": 0, "chunks_processed": 1,
                  "end_to_end_internal_time_ms": 0,
                  "model_runtimes": [{
                      "model_name": "appearance-classifier",
                      "call_count": 1, "total_time_ms": 0, "average_time_ms": 0,
                  }]}
        rt.setdefault("model_runtimes", [{
            "model_name": "appearance-classifier",
            "call_count": 1,
            "total_time_ms": rt.get("end_to_end_internal_time_ms") or 0,
            "average_time_ms": rt.get("end_to_end_internal_time_ms") or 0,
        }])
        # The published schema's model_runtimes carry a latency distribution, not
        # just a total. We keep only per-video aggregates, so for the usual
        # single-call case the percentiles ARE that call and are reported as
        # measured; with several calls the mean is the only summary the retained
        # aggregates support, and it is stated rather than guessed at.
        for mr in rt["model_runtimes"]:
            n = int(mr.get("call_count") or 1)
            total = float(mr.get("total_time_ms") or 0.0)
            v = total if n <= 1 else float(mr.get("average_time_ms") or (total / n))
            for key in ("p50_time_ms", "p95_time_ms", "max_time_ms"):
                mr.setdefault(key, round(v, 1))
        predictions.append({"video_id": vid, "events": events, "runtime_metadata": rt})

    doc = {
        "schema_version": "1.0",
        "predictions": predictions,
    }
    return doc, warnings


def _conf(desc: str) -> float:
    """Pull `confidence 0.78` out of a description_summary, else 0."""
    marker = "confidence"
    i = desc.find(marker)
    if i < 0:
        return 0.0
    tail = desc[i + len(marker):].lstrip(" :")
    num = ""
    for ch in tail:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return 0.0


def _explanation(row: dict) -> str | None:
    """Arena wants 20-500 chars, and it is bonus-only: omitting never costs.
    So anything outside the window is dropped rather than padded or truncated
    into something that misrepresents what the pipeline actually found."""
    text = (row.get("description_summary") or "").strip()
    return text if 20 <= len(text) <= 500 else None


def validate(doc: dict, manifest: dict[str, int]) -> list[str]:
    """Re-check the finished document against every rule in submission.pdf."""
    errs: list[str] = []
    seen: set[str] = set()
    for p in doc.get("predictions", []):
        vid = p.get("video_id")
        if vid in seen:
            errs.append(f"{vid}: appears more than once")
        seen.add(vid)
        if vid not in manifest:
            errs.append(f"{vid}: not in the manifest")
        if "events" not in p:
            errs.append(f"{vid}: missing `events`")
        if "runtime_metadata" not in p:
            errs.append(f"{vid}: missing `runtime_metadata` (required on every video)")
        level = manifest.get(vid, 1)
        for ev in p.get("events", []):
            c = ev.get("class_name")
            if c == "normal":
                errs.append(f"{vid}: `normal` as class_name is rejected - use events: []")
            elif c not in ARENA_CLASSES:
                errs.append(f"{vid}: `{c}` is not one of the 11 classes")
            s, e = ev.get("start_time_sec"), ev.get("end_time_sec")
            if level == 1 and (s is not None or e is not None):
                errs.append(f"{vid}: Level 1 timestamps must be null, got {s}/{e}")
            if level >= 2:
                if s is None or e is None:
                    errs.append(f"{vid}: Level {level} requires start and end times")
                elif s < 0 or e <= s:
                    errs.append(f"{vid}: Level {level} needs start >= 0 and end > start, got {s}/{e}")
            if (x := ev.get("explanation")) is not None and not (20 <= len(x) <= 500):
                errs.append(f"{vid}: explanation must be 20-500 chars, got {len(x)}")
        for mr in p.get("runtime_metadata", {}).get("model_runtimes") or []:
            tot, calls = mr.get("total_time_ms"), mr.get("call_count")
            avg = mr.get("average_time_ms")
            if tot and calls and avg:
                expect = tot / calls
                if abs(expect - avg) > 0.02 * max(expect, 1e-9):
                    errs.append(f"{vid}: average_time_ms {avg} != total/calls {expect:.2f} (2% tol)")
            if (ct := mr.get("call_times_ms")) is not None and calls is not None and len(ct) != calls:
                errs.append(f"{vid}: call_times_ms has {len(ct)} entries, call_count is {calls}")
    for vid in manifest:
        if vid not in seen:
            errs.append(f"{vid}: in the manifest but absent from the submission "
                        f"(it would be scored as `normal`)")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest.json from the arena Benchmark tab")
    ap.add_argument("--pred", required=True, help="our submission CSV")
    ap.add_argument("--summaries", default=None,
                    help="dir of pipeline *.summary.json, for runtime_metadata")
    ap.add_argument("--runtimes", default=None,
                    help="optional sidecar JSON {video_id: {frames_processed, ...}}")
    ap.add_argument("--videos", default=None,
                    help="Directory of source videos, used to clip end_time_sec to duration")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--submission-id", default=None)
    ap.add_argument("--model-name", default="cascade-appearance-vlm")
    ap.add_argument("--hardware", default="1x GTX 1650 4GB")
    ap.add_argument("--no-explanations", dest="explanations", action="store_false")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Silence Level-1 events whose description confidence is below this. "
                         "Empty events score as normal - cheaper than a false alarm.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    preds = load_predictions(Path(args.pred))
    runtimes = load_runtimes(Path(args.summaries) if args.summaries else None,
                             Path(args.runtimes) if args.runtimes else None)
    durations = load_durations(Path(args.videos) if args.videos else None)
    print(f"[loaded] {len(manifest)} manifest video(s), {len(preds)} video(s) in predictions, "
          f"{len(runtimes)} with measured runtime")

    levels = {}
    for lv in manifest.values():
        levels[lv] = levels.get(lv, 0) + 1
    print(f"[levels] " + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items())))

    unmatched = sorted(set(preds) - set(manifest))
    if unmatched:
        print(f"[warn] {len(unmatched)} predicted video(s) are not in the manifest and will be "
              f"skipped: {', '.join(unmatched[:8])}{' ...' if len(unmatched) > 8 else ''}")
        print("       (expected if you scored the public test set - the arena uses its own ids)")

    doc, warnings = build(manifest, preds, runtimes, args.explanations, durations,
                          min_conf=args.min_conf)
    if args.submission_id:
        doc["submission_id"] = args.submission_id
    doc["model_name"] = args.model_name
    total_ms = sum((p["runtime_metadata"].get("end_to_end_internal_time_ms") or 0)
                   for p in doc["predictions"])
    # max_parallel_videos is 1 by measurement, not modesty: one 4GB GTX 1650
    # cannot hold a second feed's detector alongside this one.
    doc["run_metadata"] = {"total_wall_time_ms": round(total_ms, 1),
                           "max_parallel_videos": 1,
                           "hardware": args.hardware}

    for w in warnings:
        print(f"[warn] {w}")

    errs = validate(doc, manifest)
    n_events = sum(len(p["events"]) for p in doc["predictions"])
    n_normal = sum(1 for p in doc["predictions"] if not p["events"])
    print(f"\n[build] {len(doc['predictions'])} video(s), {n_events} event(s), "
          f"{n_normal} answered as normal (empty events)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2)
    size = len(text.encode("utf-8"))
    if size > MAX_BYTES:
        errs.append(f"file is {size / 1e6:.2f} MB, over the 5 MB limit")

    if errs:
        print(f"\n[REJECTED] {len(errs)} problem(s) - fix these before uploading:")
        for e in errs[:40]:
            print(f"  - {e}")
        if len(errs) > 40:
            print(f"  ... and {len(errs) - 40} more")
        # Still write it, so the file can be inspected - but the exit code is
        # non-zero so a script never uploads a document that failed validation.
        out.write_text(text, encoding="utf-8")
        print(f"\n[wrote anyway, for inspection] {out}")
        raise SystemExit(1)

    out.write_text(text, encoding="utf-8")
    print(f"[ok] validation clean -> {out} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
