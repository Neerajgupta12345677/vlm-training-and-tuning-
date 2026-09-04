"""STAGE 1 - every frame: YOLO detection + ByteTrack tracking. No VLM here.

Runs standalone to verify real-time FPS:
    python src\\detect_track.py --source C:\\dvad\\data\\vehicles.mp4
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from common import MODELS_DIR, OUTPUTS_DIR, iter_frames, iter_frames_threaded, probe_video

# COCO ids we care about for traffic/crowd scenes.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASSES = {0: "person"}
DEFAULT_CLASSES = sorted({*VEHICLE_CLASSES, *PERSON_CLASSES})

# Preference order: YOLO26n is the Jan-2026 release (NMS-free, faster on CPU);
# YOLO11n is the proven fallback if the installed ultralytics can't fetch it.
WEIGHT_PREFERENCE = ["yolo26n.pt", "yolo11n.pt"]


def resolve_weights(name: str) -> Path:
    """Return a local path to `name`, downloading into MODELS_DIR if needed.

    Ultralytics downloads relative to the process CWD, so we download from
    inside MODELS_DIR to keep multi-MB weights out of the OneDrive folder.
    """
    target = MODELS_DIR / name
    if target.exists():
        return target
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(MODELS_DIR)
    try:
        YOLO(name)  # triggers the download into MODELS_DIR
    finally:
        os.chdir(prev)
    if not target.exists():
        raise FileNotFoundError(f"ultralytics did not produce {target}")
    return target


def load_detector(weights: str | None = None, device: str = "cuda") -> tuple[YOLO, str]:
    """Load the detector, trying YOLO26n then YOLO11n. Returns (model, name)."""
    candidates = [weights] if weights else WEIGHT_PREFERENCE
    errors: list[str] = []
    for name in candidates:
        try:
            path = Path(name) if Path(name).exists() else resolve_weights(name)
            model = YOLO(str(path))
            model.to(device)
            return model, Path(name).name
        except Exception as exc:  # noqa: BLE001 - we genuinely want the next candidate
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No detector weights could be loaded:\n  " + "\n  ".join(errors))


@dataclass
class FrameDetections:
    """One frame's tracked detections, in the form Stage 2 consumes."""

    frame_idx: int
    timestamp_s: float
    track_ids: np.ndarray
    xyxy: np.ndarray
    class_ids: np.ndarray
    class_names: list[str]
    confidences: np.ndarray
    frame_diag: float = 0.0  # lets Stage 2 judge object size relative to the frame

    def __len__(self) -> int:
        return len(self.track_ids)


@dataclass
class Stage1Tracker:
    """Detector + tracker pair. Reused by pipeline.py so behaviour is identical."""

    weights: str | None = None
    device: str = "cuda"
    conf: float = 0.3
    imgsz: int = 640
    classes: list[int] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    fps: float = 30.0
    stride: int = 1
    # Class-agnostic NMS. At the low confidence aerial footage needs (0.10), one
    # vehicle routinely yields several overlapping boxes under different labels -
    # measured: a single stopped truck came back as truck+truck+bus, became three
    # ByteTrack ids, and raised three alerts for one object. Per-class NMS cannot
    # merge those because they are different classes; agnostic NMS can.
    agnostic_nms: bool = True

    def __post_init__(self) -> None:
        self.model, self.model_name = load_detector(self.weights, self.device)
        # Two different rates, deliberately: timestamps come from the true frame
        # index at source fps (so dwell times stay real), while ByteTrack sees
        # the rate frames actually arrive at, which is what sizes its track
        # buffer. Conflating them makes tracks expire wrongly under --stride.
        self.tracker = sv.ByteTrack(frame_rate=max(1, int(round(self.fps / max(self.stride, 1)))))
        self.last_detections: sv.Detections | None = None

    def process(self, frame_idx: int, frame: np.ndarray) -> FrameDetections:
        result = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            agnostic_nms=self.agnostic_nms,
        )[0]
        dets = sv.Detections.from_ultralytics(result)
        dets = self.tracker.update_with_detections(dets)
        self.last_detections = dets

        names = list(dets.data.get("class_name", [])) if dets.data else []
        if len(names) != len(dets):
            names = [self.model.names.get(int(c), str(c)) for c in (dets.class_id if dets.class_id is not None else [])]

        return FrameDetections(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps if self.fps else 0.0,
            track_ids=dets.tracker_id if dets.tracker_id is not None else np.empty(0, dtype=int),
            xyxy=dets.xyxy if dets.xyxy is not None else np.empty((0, 4)),
            class_ids=dets.class_id if dets.class_id is not None else np.empty(0, dtype=int),
            class_names=names,
            confidences=dets.confidence if dets.confidence is not None else np.empty(0),
            frame_diag=float(np.hypot(frame.shape[1], frame.shape[0])),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1: YOLO + ByteTrack, reports achieved FPS.")
    p.add_argument("--source", required=True, help="Path to a video file.")
    p.add_argument("--data_dir", default=None, help="Unused here; kept for interface parity.")
    p.add_argument("--weights", default=None, help="Override weights (default: yolo26n.pt then yolo11n.pt).")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=5,
                   help="Frames excluded from timing stats (CUDA/model warmup costs ~1.5s on frame 0).")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame. Dwell-based anomalies do not need 25fps sampling.")
    p.add_argument("--threaded", action="store_true", default=True,
                   help="Decode on a background thread so decode overlaps inference (default on).")
    p.add_argument("--no-threaded", dest="threaded", action="store_false")
    p.add_argument("--save", default=None, help="Optional annotated .mp4 output path.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-frame logging.")
    args = p.parse_args()

    meta = probe_video(args.source)
    print(f"[video] {args.source}")
    print(f"[video] {meta.width}x{meta.height} @ {meta.fps:.2f}fps, {meta.frame_count} frames")

    tracker = Stage1Tracker(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        fps=meta.fps,
        stride=args.stride,
    )
    print(f"[model] {tracker.model_name} on {args.device}")

    writer = None
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height))
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    seen_ids: set[int] = set()
    latencies: list[float] = []
    t_start = time.perf_counter()
    t_warm = None  # wall clock at the point warmup ends

    reader = iter_frames_threaded if args.threaded else iter_frames
    for n_done, (frame_idx, frame) in enumerate(
        reader(args.source, max_frames=args.max_frames, stride=args.stride)
    ):
        if n_done == args.warmup:
            t_warm = time.perf_counter()
        t0 = time.perf_counter()
        det = tracker.process(frame_idx, frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        seen_ids.update(int(t) for t in det.track_ids)

        if not args.quiet and frame_idx % 30 == 0:
            ids = ", ".join(str(int(t)) for t in det.track_ids[:8])
            print(f"  frame {frame_idx:>5} | {len(det):>2} tracked | ids: [{ids}] | {latencies[-1]:6.1f} ms")

        if writer is not None and tracker.last_detections is not None:
            dets = tracker.last_detections
            labels = [
                f"#{int(tid)} {name}"
                for tid, name in zip(det.track_ids, det.class_names or [""] * len(det))
            ]
            annotated = box_annotator.annotate(frame.copy(), detections=dets)
            annotated = label_annotator.annotate(annotated, detections=dets, labels=labels)
            writer.write(annotated)

    elapsed = time.perf_counter() - t_start
    if writer is not None:
        writer.release()

    n = len(latencies)
    if n == 0:
        print("[error] no frames processed")
        return
    lat = np.array(latencies)
    # Frame 0 pays CUDA context + model warmup (~1.5s). Including it makes the
    # mean exceed the p95 and understates steady-state throughput, so the
    # headline number is the warm one.
    warm = lat[args.warmup :] if n > args.warmup else lat
    warm_elapsed = (time.perf_counter() - t_warm) if t_warm else elapsed
    warm_fps = len(warm) / warm_elapsed if warm_elapsed > 0 else 0.0

    print("\n=== Stage 1 result ===")
    print(f"frames processed  : {n}  (first {args.warmup} excluded from stats as warmup)")
    print(f"wall clock        : {elapsed:.2f}s total")
    print(f"cold frame 0      : {lat[0]:.0f} ms  (one-off CUDA + model init)")
    print(f"warm latency      : mean {warm.mean():.1f} ms | p95 {np.percentile(warm, 95):.1f} ms | "
          f"max {warm.max():.1f} ms")
    print(f"warm FPS          : {warm_fps:.1f}   (source is {meta.fps:.0f}fps)")
    print(f"unique track ids  : {len(seen_ids)}")
    print(f"decode            : {'threaded (overlapped)' if args.threaded else 'serial'}"
          f"{f' | stride {args.stride}' if args.stride > 1 else ''}")
    # With stride, one processed frame covers `stride` source frames, so the
    # rate we must beat to keep up with a live feed drops accordingly.
    required_fps = meta.fps / max(args.stride, 1)
    print(f"REAL-TIME?        : {'YES' if warm_fps >= required_fps else 'NO'} "
          f"({warm_fps / required_fps:.2f}x the {required_fps:.1f}fps needed to keep up)")
    print(f"resolution        : {meta.width}x{meta.height} "
          f"(detector runs at imgsz={args.imgsz}; decode cost scales with source res)")
    if args.save:
        print(f"annotated video   : {args.save}")


if __name__ == "__main__":
    main()
