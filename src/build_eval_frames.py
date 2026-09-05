"""Pack an evaluation set into JPEG frames + eval.jsonl for a Kaggle VLM run.

Local VLM inference is not an option: one 4GB card cannot hold YOLO and a 3B
VLM at once, and warm latency is 27-45s per call. Kaggle's T4 does the whole
set in well under an hour. But the videos must not go with it - the eval set is
1.27GB and one clip (E024) is 718MB, which is a slow upload on venue wifi and
pointless besides, because the model only ever sees decoded frames.

Two sampling regimes, because the two levels ask different questions:

  * Level 1 wants ONE class for a short clip, so frames are motion-aware
    (`sample_window`) - a 5-second clip's anomaly is a transient and a uniform
    grid can sit either side of it.
  * Level 2/3 want WHEN, not just what. Those get a strictly UNIFORM time grid,
    because the per-frame labels are turned back into intervals afterwards and
    a motion-biased grid would distort the timeline it is meant to measure.
    This is the same sampling contract `windows_for_label` already uses.

Frames are capped at 768px on the long side to match how the fine-tuning set
was built - feeding the adapter a different resolution than it trained on is a
silent distribution shift.

    python src\\build_eval_frames.py --videos C:\\dvad\\data\\eval_ahc\\all\\test\\videos ^
        --manifest C:\\dvad\\outputs\\manifest_eval.json --out C:\\dvad\\data\\eval_frames_pack
    python src\\build_eval_frames.py ... --push
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from build_kaggle_dataset import kaggle_username
from frame_sample import _open_video, read_indices, sample_window

MAX_SIDE = 768
JPEG_Q = 88


def _save(img_rgb: np.ndarray, dest: Path) -> None:
    h, w = img_rgb.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / float(max(h, w))
        img_rgb = cv2.resize(img_rgb, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])


def _process(job: tuple) -> list[dict]:
    vid, path_s, level, frames_dir_s, l1_frames, sample_dt, max_frames = job
    path, frames_dir = Path(path_s), Path(frames_dir_s)
    cap, fps, n_total = _open_video(path)
    if cap is None or n_total <= 0:
        if cap is not None:
            cap.release()
        return []
    duration = n_total / fps
    rows: list[dict] = []

    if level >= 2:
        # Uniform time grid. Resolution is what an IoU>=0.5 gate needs; a 2.6s
        # event cannot be localised by a grid coarser than the event itself.
        n = int(min(max_frames, max(8, duration / sample_dt)))
        idxs = np.linspace(0, n_total - 1, num=n).astype(int).tolist()
        got = read_indices(path, idxs, cap=cap, n_total=n_total)
        cap.release()
        picked = [(i / fps, fr) for i, fr in got]
    else:
        cap.release()
        picked = sample_window(path, l1_frames, fps=fps, n_total=n_total)

    for t_sec, fr in picked:
        name = f"{vid}_t{t_sec:07.2f}.jpg"
        _save(fr, frames_dir / name)
        rows.append({"video_id": vid, "image": f"frames/{name}",
                     "frame_idx": int(round(t_sec * fps)),
                     "t_sec": round(float(t_sec), 2),
                     "level": level, "duration_s": round(duration, 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="Directory of <video_id>.mp4")
    ap.add_argument("--manifest", required=True, help="JSON list of {video_id, level}")
    ap.add_argument("--out", required=True, help="Pack directory to build")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--l1-frames", type=int, default=6)
    ap.add_argument("--sample-dt", type=float, default=5.0,
                    help="Target seconds between frames on L2/L3 clips.")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="Per-clip cap so a 10-minute video stays affordable.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--slug", default="dvad-eval-frames")
    ap.add_argument("--title", default="DVAD eval frames")
    ap.add_argument("--username", default=None)
    ap.add_argument("--push", action="store_true", help="Create the dataset on Kaggle.")
    ap.add_argument("--update", action="store_true", help="Push a new version.")
    ap.add_argument("--message", default="eval frame pack")
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if isinstance(manifest, dict):
        manifest = [{"video_id": k, "level": v} for k, v in manifest.items()]
    vdir = Path(args.videos)

    jobs = []
    for m in manifest:
        vid = m["video_id"]
        p = vdir / f"{vid}.mp4"
        if not p.exists():
            print(f"  [{vid}] missing {p} - skipped")
            continue
        jobs.append((vid, str(p), int(m["level"]), str(frames_dir),
                     args.l1_frames, args.sample_dt, args.max_frames))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for got in ex.map(_process, jobs):
            if got:
                rows.append(got)
    rows = [r for group in rows for r in group]
    rows.sort(key=lambda r: (r["video_id"], r["t_sec"]))

    (out / "eval.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    by_level: dict[int, int] = {}
    by_video: dict[str, int] = {}
    for r in rows:
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
        by_video[r["video_id"]] = by_video.get(r["video_id"], 0) + 1
    size_mb = sum(f.stat().st_size for f in frames_dir.glob("*.jpg")) / 1e6
    print(f"\n[pack] {len(rows)} frame(s) from {len(by_video)} video(s), {size_mb:.1f} MB")
    for lv in sorted(by_level):
        print(f"  L{lv}: {by_level[lv]} frames")
    for v in sorted(by_video):
        print(f"    {v}: {by_video[v]}")

    # Auth here is the standalone KGAT_ token, which stores no username on
    # disk - only the CLI session knows it. Getting this wrong writes a
    # dataset id under the wrong owner and the push 403s.
    user = args.username or kaggle_username()
    if not user:
        raise SystemExit("Could not resolve the Kaggle username. Pass --username, "
                         "or run: python src\\setup_kaggle.py --verify-only")
    meta = {"title": args.title, "id": f"{user}/{args.slug}", "licenses": [{"name": "CC0-1.0"}]}
    # write_bytes, NOT PowerShell / Path.write_text with a default encoding: a
    # UTF-8 BOM here makes the Kaggle CLI fail with the misleading
    # "Expecting value: line 1 column 1 (char 0)".
    (out / "dataset-metadata.json").write_bytes(
        json.dumps(meta, indent=2).encode("utf-8"))
    print(f"[meta] {meta['id']}")

    if args.push or args.update:
        kaggle = Path(sys.executable).with_name("kaggle.exe")
        cmd = ([str(kaggle), "datasets", "version", "-p", str(out), "-m", args.message,
                "--dir-mode", "zip"] if args.update else
               [str(kaggle), "datasets", "create", "-p", str(out), "--dir-mode", "zip"])
        print("[kaggle]", " ".join(cmd))
        raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
