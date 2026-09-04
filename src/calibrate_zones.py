"""One-time per-video zone calibration. Never runs per frame.

Three modes, fastest first - which matters when a new dataset lands on the day:

  --auto        Derive zones from observed motion. Vehicles that move mark
                driving lanes; vehicles that sit still mark parking. Each lane
                also gets a mean flow direction, which powers the wrong-way
                rule in Stage 2. No human input required.

  --draw        Draw polygons by hand with the mouse (native Windows window).

  --whole-frame Label the entire frame as one driving lane. Crude but instant,
                and enough to get the pipeline running on unseen footage.

    python src\\calibrate_zones.py --auto --source C:\\dvad\\data\\vehicles.mp4
    python src\\calibrate_zones.py --draw --source C:\\dvad\\data\\vehicles.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import DATA_DIR, iter_frames, probe_video

ZONE_KINDS = {
    ord("1"): "driving_lane",
    ord("2"): "shoulder",
    ord("3"): "parking",
    ord("4"): "sidewalk",
    ord("5"): "restricted",
}
KIND_COLOURS = {
    "driving_lane": (0, 200, 0),
    "shoulder": (0, 200, 200),
    "parking": (200, 120, 0),
    "sidewalk": (200, 0, 200),
    "restricted": (0, 0, 220),
    "unknown": (150, 150, 150),
}


def save_zones(path: Path, zones: list[dict], default_kind: str, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"default_kind": default_kind, "source": meta, "zones": zones}, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] {len(zones)} zone(s) written to {path}")
    for z in zones:
        flow = f", flow={z['flow_deg']:.0f}deg" if z.get("flow_deg") is not None else ""
        print(f"     {z['name']:<16} {z['kind']:<14} {len(z['polygon'])} pts{flow}")


# --- Mode 1: automatic, from observed motion ------------------------------
def auto_calibrate(video: Path, args) -> list[dict]:
    from context_state import VEHICLE_NAMES
    from detect_track import Stage1Tracker

    meta = probe_video(video)
    scale = args.heat_scale
    hh, hw = meta.height // scale, meta.width // scale
    moving = np.zeros((hh, hw), np.float32)
    static = np.zeros((hh, hw), np.float32)
    # Per-heatmap-cell heading accumulators, for lane flow direction.
    flow_sin = np.zeros((hh, hw), np.float32)
    flow_cos = np.zeros((hh, hw), np.float32)

    stage1 = Stage1Tracker(weights=args.weights, device=args.device, conf=args.conf, fps=meta.fps)
    history: dict[int, tuple[float, float, float]] = {}  # tid -> (t, cx, cy)
    frames = 0

    print(f"[auto] observing up to {args.frames} frames of {video.name} ...")
    for frame_idx, frame in iter_frames(video, max_frames=args.frames, stride=args.stride):
        det = stage1.process(frame_idx, frame)
        frames += 1
        for i, tid in enumerate(det.track_ids):
            tid = int(tid)
            name = det.class_names[i] if i < len(det.class_names) else ""
            if name not in VEHICLE_NAMES:
                continue
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            diag = float(np.hypot(x2 - x1, y2 - y1)) or 1.0

            gy, gx = int(cy) // scale, int(cx) // scale
            if not (0 <= gy < hh and 0 <= gx < hw):
                continue

            prev = history.get(tid)
            history[tid] = (det.timestamp_s, cx, cy)
            if prev is None:
                continue
            dt = det.timestamp_s - prev[0]
            if dt <= 1e-6:
                continue
            dx, dy = cx - prev[1], cy - prev[2]
            norm_speed = (float(np.hypot(dx, dy)) / diag) / dt

            if norm_speed >= args.move_speed:
                moving[gy, gx] += 1.0
                ang = np.arctan2(dy, dx)
                flow_sin[gy, gx] += np.sin(ang)
                flow_cos[gy, gx] += np.cos(ang)
            elif norm_speed < args.still_speed:
                static[gy, gx] += 1.0

        if frames % 50 == 0:
            print(f"  {frames} frames observed, {len(history)} vehicle tracks seen")

    if moving.max() <= 0:
        raise SystemExit(
            "[auto] no vehicle motion observed - cannot derive lanes. "
            "Try more --frames, a lower --conf, or use --whole-frame."
        )

    moving_s = cv2.GaussianBlur(moving, (0, 0), args.blur)
    static_s = cv2.GaussianBlur(static, (0, 0), args.blur)

    zones: list[dict] = []

    def contours_from(mask: np.ndarray) -> list[np.ndarray]:
        m = (mask * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (hh * hw) * args.min_area_frac
        return [c for c in cnts if cv2.contourArea(c) >= min_area]

    lane_mask = (moving_s > moving_s.max() * args.lane_thresh).astype(np.float32)
    # Vehicle centroids trace a thin line down the middle of a lane, so the raw
    # mask hugs that line and misses the lane's actual width - a vehicle stopped
    # slightly off the centreline then reads as "unmapped" and the VLM will not
    # commit to a verdict. Dilating widens each lane to a realistic footprint.
    if args.dilate > 0:
        k = 2 * args.dilate + 1
        lane_mask = cv2.dilate(lane_mask, np.ones((k, k), np.uint8))
    for i, cnt in enumerate(contours_from(lane_mask), 1):
        poly = cv2.approxPolyDP(cnt, args.epsilon * cv2.arcLength(cnt, True), True)
        pts = (poly.reshape(-1, 2) * scale).astype(int)
        if len(pts) < 3:
            continue
        cell_mask = np.zeros((hh, hw), np.uint8)
        cv2.drawContours(cell_mask, [cnt], -1, 1, -1)
        s, c = float((flow_sin * cell_mask).sum()), float((flow_cos * cell_mask).sum())
        flow_deg = float(np.degrees(np.arctan2(s, c)) % 360.0) if (s or c) else None
        # flow_sin/flow_cos accumulate UNIT vectors, so the length of their sum
        # divided by the sample count is the circular resultant R in [0, 1]:
        # 1.0 = every observed vehicle went the same way, ~0 = directions cancel.
        # Without this a lane derived from a few seconds of scattered motion
        # yields a confident-looking flow_deg that is pure noise, and every
        # vehicle then reads as wrong-way. Measured: 5 false wrong-way alerts
        # and 0 true ones on the public test set before this was recorded.
        n_samples = float((moving * cell_mask).sum())
        flow_consistency = (float(np.hypot(s, c)) / n_samples) if n_samples > 0 else 0.0
        zones.append(
            {"name": f"lane{i}", "kind": "driving_lane", "polygon": pts.tolist(),
             "flow_deg": flow_deg,
             "flow_consistency": round(flow_consistency, 4),
             "flow_samples": int(n_samples)}
        )

    # Parking = consistently occupied but not a travel path.
    if static_s.max() > 0:
        park_mask = (
            (static_s > static_s.max() * args.park_thresh) & (moving_s < moving_s.max() * args.lane_thresh)
        ).astype(np.float32)
        for i, cnt in enumerate(contours_from(park_mask), 1):
            poly = cv2.approxPolyDP(cnt, args.epsilon * cv2.arcLength(cnt, True), True)
            pts = (poly.reshape(-1, 2) * scale).astype(int)
            if len(pts) < 3:
                continue
            zones.append({"name": f"parking{i}", "kind": "parking", "polygon": pts.tolist(), "flow_deg": None})

    print(f"[auto] derived {sum(1 for z in zones if z['kind'] == 'driving_lane')} driving lane(s), "
          f"{sum(1 for z in zones if z['kind'] == 'parking')} parking area(s) from {frames} frames")

    if args.preview:
        _write_preview(video, zones, moving_s, scale, args)
    return zones


def _write_preview(video: Path, zones: list[dict], heat: np.ndarray, scale: int, args) -> None:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    heat_up = cv2.resize(heat / (heat.max() or 1.0), (frame.shape[1], frame.shape[0]))
    colour = cv2.applyColorMap((heat_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.6, colour, 0.4, 0)
    for z in zones:
        pts = np.array(z["polygon"], np.int32)
        cv2.polylines(blended, [pts], True, KIND_COLOURS[z["kind"]], 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(blended, f"{z['name']}:{z['kind']}", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KIND_COLOURS[z["kind"]], 2, cv2.LINE_AA)
        if z.get("flow_deg") is not None:
            ang = np.radians(z["flow_deg"])
            tip = (int(cx + 60 * np.cos(ang)), int(cy + 60 * np.sin(ang)))
            cv2.arrowedLine(blended, (cx, cy), tip, (255, 255, 255), 2, tipLength=0.3)
    out = Path(args.preview_out) if args.preview_out else video.with_name(video.stem + "_zones.jpg")
    cv2.imwrite(str(out), blended)
    print(f"[auto] preview image: {out}")


# --- Mode 2: manual drawing -----------------------------------------------
def draw_calibrate(video: Path, args) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame}")

    zones: list[dict] = []
    current: list[tuple[int, int]] = []
    state = {"pending_flow": None}
    win = "calibrate zones"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["pending_flow"] is not None:
                z = zones[state["pending_flow"]]
                pts = np.array(z["polygon"], np.int32)
                cx, cy = pts.mean(axis=0)
                z["flow_deg"] = float(np.degrees(np.arctan2(y - cy, x - cx)) % 360.0)
                print(f"  flow for {z['name']}: {z['flow_deg']:.0f}deg")
                state["pending_flow"] = None
            else:
                current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print(
        "\nControls:\n"
        "  left click   add point        right click  undo point\n"
        "  1..5         close polygon as 1=driving_lane 2=shoulder 3=parking 4=sidewalk 5=restricted\n"
        "  f            set flow direction for the last zone (then click a point)\n"
        "  r            reset current polygon   u  remove last zone\n"
        "  s            save and exit           q  quit without saving\n"
    )

    while True:
        canvas = base.copy()
        for z in zones:
            pts = np.array(z["polygon"], np.int32)
            cv2.polylines(canvas, [pts], True, KIND_COLOURS[z["kind"]], 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, f"{z['name']}:{z['kind']}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KIND_COLOURS[z["kind"]], 2, cv2.LINE_AA)
            if z.get("flow_deg") is not None:
                ang = np.radians(z["flow_deg"])
                cv2.arrowedLine(canvas, (cx, cy), (int(cx + 60 * np.cos(ang)), int(cy + 60 * np.sin(ang))),
                                (255, 255, 255), 2, tipLength=0.3)
        for i, pt in enumerate(current):
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
            if i:
                cv2.line(canvas, current[i - 1], pt, (0, 255, 255), 1)
        hint = "click a point to set flow direction" if state["pending_flow"] is not None else \
               f"{len(current)} pts | {len(zones)} zones | 1-5 close, s save, q quit"
        cv2.putText(canvas, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted, nothing saved")
        if key == ord("s"):
            break
        if key == ord("r"):
            current.clear()
        if key == ord("u") and zones:
            print(f"  removed {zones.pop()['name']}")
        if key == ord("f") and zones:
            state["pending_flow"] = len(zones) - 1
        if key in ZONE_KINDS and len(current) >= 3:
            kind = ZONE_KINDS[key]
            n = sum(1 for z in zones if z["kind"] == kind) + 1
            zones.append({"name": f"{kind}{n}", "kind": kind, "polygon": [list(p) for p in current],
                          "flow_deg": None})
            print(f"  added {zones[-1]['name']} ({len(current)} pts)")
            current.clear()

    cv2.destroyAllWindows()
    return zones


def main() -> None:
    p = argparse.ArgumentParser(description="Per-video zone calibration (auto, manual, or whole-frame).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto", action="store_true", help="Derive zones from observed vehicle motion.")
    mode.add_argument("--draw", action="store_true", help="Draw polygons by hand.")
    mode.add_argument("--whole-frame", action="store_true", help="One driving_lane covering everything.")
    p.add_argument("--source", required=True)
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--out", default=None, help="Zones JSON output (default: <video>_zones.json).")
    p.add_argument("--default-zone", default="unknown", help="Zone kind for points in no polygon.")
    # auto
    p.add_argument("--frames", type=int, default=400, help="Frames to observe in --auto.")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--heat-scale", type=int, default=8, help="Heatmap downscale factor.")
    p.add_argument("--blur", type=float, default=3.0)
    p.add_argument("--move-speed", type=float, default=0.15, help="Above this = moving.")
    p.add_argument("--still-speed", type=float, default=0.03, help="Below this = stationary.")
    p.add_argument("--lane-thresh", type=float, default=0.06, help="Fraction of peak motion to call a lane.")
    p.add_argument("--dilate", type=int, default=6,
                   help="Widen lanes by this many heatmap cells (cell = --heat-scale px). "
                        "Centroid tracks are thin; lanes are not.")
    p.add_argument("--park-thresh", type=float, default=0.35)
    p.add_argument("--min-area-frac", type=float, default=0.004)
    p.add_argument("--epsilon", type=float, default=0.01, help="Polygon simplification strength.")
    p.add_argument("--preview", action="store_true", default=True, help="Write a heatmap preview image.")
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument("--preview-out", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.3)
    # draw
    p.add_argument("--frame", type=int, default=0, help="Frame index to draw on.")
    args = p.parse_args()

    video = Path(args.source)
    meta = probe_video(video)

    if args.whole_frame:
        zones = [{
            "name": "frame",
            "kind": "driving_lane",
            "polygon": [[0, 0], [meta.width, 0], [meta.width, meta.height], [0, meta.height]],
            "flow_deg": None,
        }]
    elif args.auto:
        zones = auto_calibrate(video, args)
    else:
        zones = draw_calibrate(video, args)

    if not zones:
        raise SystemExit("no zones produced")

    out = Path(args.out) if args.out else video.with_name(video.stem + "_zones.json")
    save_zones(out, zones, args.default_zone,
               {"video": video.name, "width": meta.width, "height": meta.height})


if __name__ == "__main__":
    main()
