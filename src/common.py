"""Shared foundation: paths, the anomaly verdict schema, and video I/O.

The verdict schema lives here and nowhere else. Stage 3 (local VLM), the
distillation teacher, and the eval harness all import it, so a schema change
can never desync them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
from pydantic import BaseModel, Field

# --- Paths -----------------------------------------------------------------
# Heavy artifacts live outside OneDrive so it never tries to sync gigabytes.
DVAD_ROOT = Path(os.environ.get("DVAD_ROOT", r"C:\dvad"))
MODELS_DIR = DVAD_ROOT / "models"
DATA_DIR = DVAD_ROOT / "data"
OUTPUTS_DIR = DVAD_ROOT / "outputs"
PROJECT_DIR = Path(__file__).resolve().parent.parent


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DATA_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- Verdict schema (single source of truth) -------------------------------
class AnomalyVerdict(BaseModel):
    """What Stage 3 and the teacher model both must return."""

    anomalous: bool = Field(description="True if this is a genuine anomaly worth an operator's attention")
    severity: float = Field(ge=0.0, le=1.0, description="0.0 = benign, 1.0 = critical/emergency")
    reason: str = Field(description="One short sentence of concrete visual justification")


def verdict_json_schema() -> dict:
    """JSON schema for runtimes that constrain decoding (Ollama's `format`)."""
    return AnomalyVerdict.model_json_schema()


class SceneObservation(BaseModel):
    """What a small VLM is actually reliable at: describing what is visible.

    Measured on qwen2.5vl:3b (Q4) and moondream: asked to decide `anomalous`
    from a still frame plus tracker text, both models tracked whichever rule the
    prompt emphasised rather than the input facts, scoring at chance across four
    prompt revisions. Perception was consistently good; the conditional boolean
    was not. So the boolean is decided by Stage 2's deterministic rules, and the
    VLM is asked only for observations - hazards it can literally see, plus a
    human-readable description.
    """

    hazard_visible: bool = Field(
        description="True only if fire, smoke, a collision, spilled debris, a person on "
        "the carriageway, or a gathering crowd is actually visible"
    )
    hazard_type: str = Field(description="Which hazard, or 'none'")
    surroundings: str = Field(
        description="One short sentence naming where the boxed object is (live lane, hard "
        "shoulder, lay-by, parking area, junction) and what is around it"
    )


def observation_json_schema() -> dict:
    return SceneObservation.model_json_schema()


# --- JSONL helpers ---------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Video I/O -------------------------------------------------------------
@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def diagonal(self) -> float:
        return (self.width**2 + self.height**2) ** 0.5


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FRAME_SEQ_FPS = float(os.environ.get("DVAD_FRAME_SEQ_FPS", "25"))


def frame_sequence(path: str | Path) -> list[Path] | None:
    """Return sorted frames if `path` is a directory of images, else None.

    Public anomaly benchmarks (UCSD Ped, CUHK Avenue, ShanghaiTech, UCF-Crime)
    ship as numbered frame folders rather than video files, so every entry point
    accepts a directory wherever it accepts an .mp4.
    """
    p = Path(path)
    if not p.is_dir():
        return None
    frames = sorted((f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda f: f.name)
    return frames or None


def probe_video(path: str | Path) -> VideoMeta:
    frames = frame_sequence(path)
    if frames is not None:
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise FileNotFoundError(f"Cannot read first frame: {frames[0]}")
        h, w = first.shape[:2]
        # Frame folders carry no timing, so dwell thresholds depend on this
        # assumed rate. Override with DVAD_FRAME_SEQ_FPS if the dataset says otherwise.
        return VideoMeta(width=w, height=h, fps=FRAME_SEQ_FPS, frame_count=len(frames))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return VideoMeta(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # A container reporting 0/NaN fps would poison every dwell-time
            # calculation downstream, so fall back to a sane default.
            fps=fps if fps and fps > 0 else 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames_threaded(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
    queue_size: int = 3,
) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Same contract as iter_frames, but decodes on a background thread.

    Decode and GPU inference are otherwise serial, so their costs add: on 4K
    footage that measured 23.5ms decode + 29.3ms detect = ~50ms/frame (0.8x
    real-time). Overlapping them hides the decode behind the inference.

    queue_size is deliberately tiny - a 4K frame is ~25MB, so a long queue
    would blow a meaningful hole in 7.35GB of RAM.
    """
    import queue
    import threading

    q: "queue.Queue[tuple[int, object] | None]" = queue.Queue(maxsize=queue_size)
    stop = threading.Event()

    def producer() -> None:
        try:
            for item in iter_frames(path, max_frames=max_frames, stride=stride):
                if stop.is_set():
                    break
                q.put(item)
        finally:
            q.put(None)  # sentinel: always sent, even on error, so we never hang

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item  # type: ignore[misc]
    finally:
        stop.set()
        # Drain so a blocked producer can observe `stop` and exit.
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:  # noqa: BLE001
                break
        thread.join(timeout=2.0)


def iter_frames(path: str | Path, max_frames: int | None = None, stride: int = 1) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Stream frames one at a time. Never loads the whole video (7.35GB RAM).

    Accepts a video file or a directory of numbered frames.
    """
    frames = frame_sequence(path)
    if frames is not None:
        emitted = 0
        for idx, fp in enumerate(frames):
            if idx % stride:
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            yield idx, img
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        idx = 0
        emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
