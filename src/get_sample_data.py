"""Fetch a real traffic video and synthesise a stopped-vehicle clip with ground truth.

We need a clip that provably contains the target anomaly, or eval has nothing to
score. Rather than hand-labelling, we take a real vehicle patch from the footage
and composite it at a fixed road position from a chosen frame onward: traffic
keeps flowing while one genuine-looking vehicle sits still. That gives an exact
ground-truth frame index and bounding box.

    python src\\get_sample_data.py --download
    python src\\get_sample_data.py --inject --source C:\\dvad\\data\\vehicles.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from common import DATA_DIR, ensure_dirs, probe_video


@contextmanager
def chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def download_sample(asset: str = "VEHICLES") -> Path:
    """Download a supervision sample video into DATA_DIR."""
    from supervision.assets import VideoAssets, download_assets

    ensure_dirs()
    try:
        enum_member = getattr(VideoAssets, asset)
    except AttributeError as e:
        available = [a for a in dir(VideoAssets) if a.isupper()]
        raise SystemExit(f"Unknown asset {asset!r}. Available: {available}") from e

    with chdir(DATA_DIR):
        download_assets(enum_member)
    path = DATA_DIR / enum_member.value
    if not path.exists():
        raise SystemExit(f"download reported success but {path} is missing")
    print(f"[ok] {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _feathered_paste(dst: np.ndarray, patch: np.ndarray, x: int, y: int, feather: int = 6) -> None:
    """Alpha-blend `patch` into `dst` at (x, y) with soft edges, in place."""
    h, w = patch.shape[:2]
    H, W = dst.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return
    mask = np.ones((h, w), np.float32)
    f = max(1, min(feather, h // 3, w // 3))
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    mask[:f, :] *= ramp[:, None]
    mask[-f:, :] *= ramp[::-1, None]
    mask[:, :f] *= ramp[None, :]
    mask[:, -f:] *= ramp[::-1][None, :]
    m3 = mask[:, :, None]
    region = dst[y : y + h, x : x + w].astype(np.float32)
    dst[y : y + h, x : x + w] = (patch.astype(np.float32) * m3 + region * (1.0 - m3)).astype(np.uint8)


def pick_vehicle_patch(video: Path, patch_frame: int, device: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Detect vehicles on one frame and return the best patch plus its box."""
    from detect_track import VEHICLE_CLASSES, load_detector

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, patch_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {patch_frame} of {video}")

    model, name = load_detector(None, device)
    res = model.predict(frame, verbose=False, conf=0.5, classes=sorted(VEHICLE_CLASSES), device=device)[0]
    if res.boxes is None or len(res.boxes) == 0:
        raise SystemExit(f"no vehicles detected on frame {patch_frame}; try a different --patch-frame")

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Prefer a mid-sized, confident vehicle: big enough to be visible, not a crop-edge blob.
    H, W = frame.shape[:2]
    score = confs * np.clip(areas / (W * H) / 0.01, 0, 3)
    best = int(np.argmax(score))
    x1, y1, x2, y2 = (int(v) for v in boxes[best])
    print(f"[patch] {name}: chose vehicle at ({x1},{y1})-({x2},{y2}) conf={confs[best]:.2f}")
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def inject_stopped_vehicle(video: Path, args) -> Path:
    meta = probe_video(video)
    patch, (x1, y1, x2, y2) = pick_vehicle_patch(video, args.patch_frame, args.device)

    out_path = Path(args.out) if args.out else DATA_DIR / f"{video.stem}_stopped.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height)
    )
    cap = cv2.VideoCapture(str(video))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and written >= args.max_frames):
            break
        if written >= args.start_frame:
            _feathered_paste(frame, patch, x1, y1)
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()

    gt = {
        "video": out_path.name,
        "source_video": video.name,
        "fps": meta.fps,
        "frames": written,
        "anomalies": [
            {
                "kind": "stopped_vehicle",
                "start_frame": args.start_frame,
                "start_time_s": round(args.start_frame / meta.fps, 2),
                "end_frame": written - 1,
                "bbox": [x1, y1, x2, y2],
                "anomalous": True,
                "note": "Composited stationary vehicle; traffic continues to flow around it.",
            }
        ],
    }
    gt_path = out_path.with_name(out_path.stem + "_ground_truth.json")
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"[ok] clip        : {out_path} ({written} frames)")
    print(f"[ok] ground truth: {gt_path}")
    print(f"[ok] anomaly from frame {args.start_frame} ({args.start_frame / meta.fps:.1f}s) at bbox {[x1, y1, x2, y2]}")
    return out_path


def inject_hazard(video: Path, args) -> Path:
    """Composite a static hazard (smoke / fire / flood) onto the road surface.

    The scene-sweep path is the only way this system can report static
    conditions, and a sweep that never cries wolf proves only half of what
    matters. This gives the positive control: a hazard that is definitely
    present, at a known frame, so a miss is unambiguous.

    Deliberately crude but visually honest - soft-edged, correct colour family,
    composited into perspective on the road rather than pasted as a flat
    rectangle. Not a substitute for real footage of a flood, and labelled as
    such wherever the results are quoted.
    """
    meta = probe_video(video)
    out_path = Path(args.out) if args.out else DATA_DIR / f"{video.stem}_{args.hazard}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height)
    )

    # Place it low-centre, where the road surface is in these views.
    cx = int(meta.width * args.hazard_x)
    cy = int(meta.height * args.hazard_y)
    rx = int(meta.width * args.hazard_scale)
    ry = int(rx * 0.45)  # foreshortened, as a patch on the ground would be

    yy, xx = np.mgrid[0:meta.height, 0:meta.width]
    d = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - d, 0.0, 1.0) ** 1.5  # soft-edged falloff
    rng = np.random.default_rng(7)
    mask *= 0.75 + 0.25 * rng.random(mask.shape)  # break up the perfect ellipse
    mask3 = np.dstack([mask] * 3).astype(np.float32)

    colour = {
        "smoke": (205.0, 205.0, 200.0),   # BGR, near-neutral grey plume
        "fire": (40.0, 120.0, 240.0),     # BGR -> orange
        "flood": (150.0, 120.0, 85.0),    # BGR -> muddy brown-grey standing water
    }[args.hazard]
    layer = np.zeros((meta.height, meta.width, 3), np.float32)
    layer[:] = colour
    alpha = {"smoke": 0.80, "fire": 0.85, "flood": 0.70}[args.hazard]

    cap = cv2.VideoCapture(str(video))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and written >= args.max_frames):
            break
        if written >= args.start_frame:
            f = frame.astype(np.float32)
            m = mask3 * alpha
            if args.hazard == "flood":
                # Standing water darkens and mirrors rather than covering.
                f = f * (1 - m) + (f * 0.45 + layer * 0.55) * m
            else:
                f = f * (1 - m) + layer * m
            frame = np.clip(f, 0, 255).astype(np.uint8)
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()

    gt = {
        "video": out_path.name,
        "source_video": video.name,
        "fps": meta.fps,
        "frames": written,
        "synthetic": True,
        "note": "Composited hazard for a positive control. NOT real footage - "
                "quote results as a smoke-test of the detection path, not as accuracy.",
        "anomalies": [{
            "kind": args.hazard,
            "start_frame": args.start_frame,
            "start_time_s": round(args.start_frame / meta.fps, 2),
            "end_frame": written - 1,
            "bbox": [cx - rx, cy - ry, cx + rx, cy + ry],
            "anomalous": True,
        }],
    }
    gt_path = out_path.with_name(out_path.stem + "_ground_truth.json")
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"[ok] {args.hazard} clip  : {out_path} ({written} frames)")
    print(f"[ok] ground truth   : {gt_path}")
    print(f"[ok] hazard from frame {args.start_frame} at bbox {gt['anomalies'][0]['bbox']}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch sample footage and build a ground-truth anomaly clip.")
    p.add_argument("--download", action="store_true", help="Download a supervision sample video.")
    p.add_argument("--asset", default="VEHICLES", help="supervision VideoAssets name (e.g. VEHICLES, PEOPLE_WALKING).")
    p.add_argument("--inject", action="store_true", help="Composite a stationary vehicle into --source.")
    p.add_argument("--inject-hazard", action="store_true",
                   help="Composite a static hazard (smoke/fire/flood) as a positive control "
                        "for the scene-sweep path.")
    p.add_argument("--hazard", default="smoke", choices=["smoke", "fire", "flood"])
    p.add_argument("--hazard-x", type=float, default=0.45, help="Centre x as a frame fraction.")
    p.add_argument("--hazard-y", type=float, default=0.62, help="Centre y as a frame fraction.")
    p.add_argument("--hazard-scale", type=float, default=0.16, help="Radius as a frame fraction.")
    p.add_argument("--source", default=None, help="Video to inject into.")
    p.add_argument("--data_dir", default=str(DATA_DIR), help="Where videos live.")
    p.add_argument("--out", default=None)
    p.add_argument("--patch-frame", type=int, default=0, help="Frame to lift the vehicle patch from.")
    p.add_argument("--start-frame", type=int, default=60, help="Frame the vehicle 'stops' at.")
    p.add_argument("--max-frames", type=int, default=0, help="Truncate output (0 = whole video).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not args.download and not args.inject and not args.inject_hazard:
        p.print_help()
        return

    video = Path(args.source) if args.source else None
    if args.download:
        video = download_sample(args.asset)
    if args.inject:
        if video is None:
            raise SystemExit("--inject needs --source (or run with --download first)")
        inject_stopped_vehicle(video, args)
    if args.inject_hazard:
        if video is None:
            raise SystemExit("--inject-hazard needs --source (or run with --download first)")
        inject_hazard(video, args)


if __name__ == "__main__":
    main()
