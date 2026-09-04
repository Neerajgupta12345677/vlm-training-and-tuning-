"""Turns our events.jsonl into the organisers' exact submission schema.

Columns (from "AHC Visual Intelligence Hackathon - Training and Public Test
Data", verbatim): video_id, level, is_anomaly, class_name, start_time_sec,
end_time_sec, description_summary.

Three things this has to get right, none of which are optional:
  1. A video with zero detections still needs a row - class_name=normal,
     is_anomaly=False. Our pipeline emits nothing for a clean video, so
     without this every normal video in the test set silently has no
     prediction at all.
  2. class_name must be one of the twelve official strings, exactly -
     label_map.validate_label is the gate that catches a typo before it
     costs accuracy on every row using it.
  3. Repeated events on the same track/label (our dwell rules re-trigger as
     a vehicle keeps sitting there) must MERGE into one interval, or one
     real incident becomes several duplicate rows.

    python src\\submission.py --events C:\\dvad\\outputs\\events_x.jsonl --video-id clip01 --duration 45.2
    python src\\submission.py --events-dir C:\\dvad\\outputs\\ahc_events --out C:\\dvad\\outputs\\predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from label_map import map_event_to_label, validate_label

CSV_COLUMNS = ["video_id", "level", "is_anomaly", "class_name",
              "start_time_sec", "end_time_sec", "description_summary"]

# How far back an event's real onset likely was, per kind - our dwell rules
# fire once a threshold is CROSSED, so the trigger frame is the end of the
# window, not the start. Keyed to the exact feature names each rule writes
# in context_state.py.
_ONSET_FEATURE = {
    "stopped_vehicle": "stationary_s",
    "loitering": "stationary_s",
    "traffic_congestion": "sustained_s",
    "person_in_roadway": "age_s",
}

# Two events of the same label on the same video merge into one interval if
# they are within this many seconds of each other. Wide enough to absorb the
# cooldown-driven re-triggers our dwell rules produce (measured: repeat
# triggers land 4-8s apart at typical --cooldown settings), narrow enough that
# two genuinely separate incidents of the same class do not get fused.
DEFAULT_MERGE_GAP_S = 20.0


def _event_onset_s(row: dict) -> float:
    """Best estimate of when this event's underlying condition actually began."""
    t = float(row.get("timestamp_s", 0.0))
    kind = row.get("kind", "")
    feat = row.get("features") or {}
    key = _ONSET_FEATURE.get(kind)
    if key and key in feat and feat[key] is not None:
        return max(0.0, t - float(feat[key]))
    return t


def _hazard_type(row: dict) -> str | None:
    obs = row.get("observation") or {}
    return obs.get("hazard_type")


def _survived(row: dict) -> bool:
    """True if the FUSED verdict (post-VLM) says this event is real.

    Uses `verdict.anomalous`, not `rule_anomalous` - the whole point of the
    hybrid decision path is that a VLM hazard can escalate a rule-benign
    event (crowd_density) and rule-anomalous events still need a verdict to
    exist at all (a --no-vlm harvest row has verdict=None and must not be
    treated as a detection).
    """
    v = row.get("verdict")
    return bool(v and v.get("anomalous"))


class Episode:
    """One merged occurrence of one label within one video."""

    __slots__ = ("label", "start", "end", "reasons")

    def __init__(self, label: str, start: float, end: float, reason: str):
        self.label = label
        self.start = start
        self.end = end
        self.reasons: list[str] = [reason] if reason else []

    def absorb(self, start: float, end: float, reason: str) -> None:
        self.start = min(self.start, start)
        self.end = max(self.end, end)
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def summary(self) -> str:
        # Keep the first (usually most specific) reason plus a count, rather
        # than concatenating every repeat trigger's near-identical sentence.
        if not self.reasons:
            return ""
        extra = f" ({len(self.reasons) - 1} more observation(s))" if len(self.reasons) > 1 else ""
        return self.reasons[0][:300] + extra


def build_episodes(events: list[dict], merge_gap_s: float = DEFAULT_MERGE_GAP_S) -> list[Episode]:
    """Group surviving events into merged (label, start, end) episodes."""
    by_label: dict[str, list[Episode]] = {}
    # Process in time order so merging only ever looks at the most recent
    # open episode for that label.
    for row in sorted(events, key=lambda r: r.get("timestamp_s", 0.0)):
        if not _survived(row):
            continue
        label = map_event_to_label(row.get("kind", ""), _hazard_type(row))
        if label is None:
            continue  # crowd_density unescalated, scene_sweep with no hazard, etc.
        start = _event_onset_s(row)
        end = float(row.get("timestamp_s", start))
        reason = (row.get("verdict") or {}).get("reason", "")

        bucket = by_label.setdefault(label, [])
        if bucket and start - bucket[-1].end <= merge_gap_s:
            bucket[-1].absorb(start, end, reason)
        else:
            bucket.append(Episode(label, start, end, reason))

    return [ep for eps in by_label.values() for ep in eps]


def build_rows(events: list[dict], video_id: str, level: int = 3,
              merge_gap_s: float = DEFAULT_MERGE_GAP_S) -> list[dict]:
    """The full set of submission rows for one video.

    `level` controls how much temporal detail is kept:
      3 -> one row per merged episode, full start/end timestamps
      2 -> same as 3 (both are "populated" per the organisers' schema note)
      1 -> one row per distinct label for the whole video, NO timestamps
           (matches "empty on Level 1" in the schema)
    A video with zero surviving episodes gets exactly one `normal` row.
    """
    episodes = build_episodes(events, merge_gap_s=merge_gap_s)

    if not episodes:
        return [{
            "video_id": video_id, "level": level, "is_anomaly": False,
            "class_name": validate_label("normal"),
            "start_time_sec": "", "end_time_sec": "", "description_summary": "",
        }]

    if level == 1:
        # Collapse to one row per distinct label - Level 1 is video-level
        # classification, not event-level localisation.
        seen: dict[str, Episode] = {}
        for ep in episodes:
            if ep.label not in seen or (ep.end - ep.start) > (seen[ep.label].end - seen[ep.label].start):
                seen[ep.label] = ep
        return [{
            "video_id": video_id, "level": 1, "is_anomaly": True,
            "class_name": validate_label(ep.label),
            "start_time_sec": "", "end_time_sec": "", "description_summary": ep.summary(),
        } for ep in seen.values()]

    return [{
        "video_id": video_id, "level": level, "is_anomaly": True,
        "class_name": validate_label(ep.label),
        "start_time_sec": round(ep.start, 2), "end_time_sec": round(ep.end, 2),
        "description_summary": ep.summary(),
    } for ep in sorted(episodes, key=lambda e: e.start)]


def write_csv(rows: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Build organiser-format submission rows from events.jsonl.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--events", help="A single video's events.jsonl.")
    src.add_argument("--events-dir", help="A directory of <video_id>.jsonl files, one per video.")
    p.add_argument("--video-id", default=None, help="Required with --events; the video's id/filename stem.")
    p.add_argument("--duration", type=float, default=None,
                   help="Video duration in seconds. Not required for row-building, kept for "
                        "future use (e.g. clamping end_time_sec) and self-documentation.")
    p.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP_S)
    p.add_argument("--out", default=str(Path(r"C:\dvad\outputs") / "predictions.csv"))
    p.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()  # always start clean; --append semantics live in write_csv for batch callers

    if args.events:
        if not args.video_id:
            raise SystemExit("--events requires --video-id")
        events = read_jsonl(Path(args.events))
        rows = build_rows(events, args.video_id, level=args.level, merge_gap_s=args.merge_gap)
        write_csv(rows, out_path)
        print(f"[ok] {len(rows)} row(s) for video_id={args.video_id!r} -> {out_path}")
        return

    events_dir = Path(args.events_dir)
    files = sorted(events_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in {events_dir}")
    total = 0
    for i, f in enumerate(files):
        video_id = f.stem
        rows = build_rows(read_jsonl(f), video_id, level=args.level, merge_gap_s=args.merge_gap)
        write_csv(rows, out_path, append=(i > 0))
        total += len(rows)
    print(f"[ok] {total} row(s) across {len(files)} video(s) -> {out_path}")


if __name__ == "__main__":
    main()
