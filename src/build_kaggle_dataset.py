"""Package teacher labels + frames into an uploadable Kaggle dataset.

The prompt text the student will be trained on is baked into train.jsonl here,
copied straight from src\\vlm_reason.py. The training notebook therefore needs
no prompt knowledge of its own, and the training text cannot drift away from
what Stage 3 actually sends at inference time.

    python src\\build_kaggle_dataset.py --labels C:\\dvad\\data\\pseudo_labels.jsonl
    python src\\build_kaggle_dataset.py --labels ... --push          # create
    python src\\build_kaggle_dataset.py --labels ... --push --update # new version
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import DATA_DIR, OUTPUTS_DIR, read_jsonl, write_jsonl
from vlm_reason import SYSTEM_PROMPT, build_prompt_for_training

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def kaggle_username() -> str | None:
    """Resolve the Kaggle username under either auth scheme.

    Legacy auth stores it in kaggle.json. The newer standalone access token
    (KGAT_...) has no username anywhere on disk, so ask the CLI, which knows it
    from the authenticated session.
    """
    if KAGGLE_JSON.exists():
        try:
            user = json.loads(KAGGLE_JSON.read_text(encoding="utf-8")).get("username")
            if user:
                return user
        except Exception:  # noqa: BLE001
            pass

    exe = Path(sys.executable).parent / "kaggle.exe"
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "kaggle"]
    try:
        proc = subprocess.run(cmd + ["config", "view"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
        for line in (proc.stdout or "").splitlines():
            if "username:" in line:
                user = line.split("username:", 1)[1].strip()
                if user and user.lower() != "none":
                    return user
    except Exception:  # noqa: BLE001
        pass
    return None


def build(args) -> Path:
    labels = read_jsonl(Path(args.labels))
    rows = [r for r in labels if "error" not in r]
    if not rows:
        raise SystemExit(f"No usable rows in {args.labels}")

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(args.frames_dir) if args.frames_dir else OUTPUTS_DIR / "events"
    train_rows: list[dict] = []
    missing = 0

    for r in rows:
        src = frames_dir / r["image"]
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, images_dir / r["image"])
        # The exact string the student must learn to emit.
        target = json.dumps(
            {
                "anomalous": bool(r["anomalous"]),
                "severity": round(float(r["severity"]), 2),
                "reason": r["reason"],
            },
            ensure_ascii=False,
        )
        train_rows.append(
            {
                "image": f"images/{r['image']}",
                "system": SYSTEM_PROMPT,
                "instruction": build_prompt_for_training(r["context"]),
                "target": target,
                "context": r["context"],
                "anomalous": bool(r["anomalous"]),
                "severity": float(r["severity"]),
                "teacher_model": r.get("teacher_model"),
                "event_kind": r.get("event_kind"),
                "source_video": r.get("source_video"),
            }
        )

    if not train_rows:
        raise SystemExit(f"No frames found in {frames_dir}; nothing to package.")

    write_jsonl(out_dir / "train.jsonl", train_rows)

    user = kaggle_username() or args.username or "YOUR_KAGGLE_USERNAME"
    meta = {
        "title": args.title,
        "id": f"{user}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pos = sum(1 for r in train_rows if r["anomalous"])
    print(f"[ok] dataset dir : {out_dir}")
    print(f"[ok] samples     : {len(train_rows)} ({pos} anomalous / {len(train_rows) - pos} benign)")
    if missing:
        print(f"[warn] {missing} label row(s) had no matching frame in {frames_dir}")
    print(f"[ok] kaggle id   : {meta['id']}")
    if pos == 0 or pos == len(train_rows):
        print("[warn] single-class dataset - the student cannot learn a decision boundary.")
    return out_dir


def push(out_dir: Path, update: bool, message: str) -> None:
    exe = Path(sys.executable).parent / "kaggle.exe"
    kaggle_cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "kaggle"]
    if update:
        cmd = kaggle_cmd + ["datasets", "version", "-p", str(out_dir), "-m", message, "--dir-mode", "zip"]
    else:
        cmd = kaggle_cmd + ["datasets", "create", "-p", str(out_dir), "--dir-mode", "zip"]
    print(f"[push] {' '.join(cmd)}")
    # encoding/errors are load-bearing on Windows: the kaggle CLI emits progress
    # bytes that cp1252 cannot decode, and the default decoding raised
    # UnicodeDecodeError *after* a successful upload - a success that looked
    # like a failure.
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
        raise SystemExit(
            "[push] kaggle CLI failed. Check that ~/.kaggle/kaggle.json exists and that the "
            "slug is not already taken (use --update to push a new version)."
        )
    print("[push] done - confirm it appears under your Kaggle account.")


def main() -> None:
    p = argparse.ArgumentParser(description="Package and optionally upload the Kaggle training dataset.")
    p.add_argument("--labels", default=str(DATA_DIR / "pseudo_labels.jsonl"))
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--out-dir", default=str(DATA_DIR / "kaggle_dataset"))
    p.add_argument("--data_dir", default=str(DATA_DIR), help="Kept for interface parity.")
    p.add_argument("--slug", default="dvad-pseudo-labels")
    p.add_argument("--title", default="DVAD drone anomaly pseudo-labels")
    p.add_argument("--username", default=None, help="Override if kaggle.json is absent.")
    p.add_argument("--clean", action="store_true", help="Wipe the output dir first.")
    p.add_argument("--push", action="store_true", help="Upload via the kaggle CLI.")
    p.add_argument("--update", action="store_true", help="Push a new version of an existing dataset.")
    p.add_argument("--message", default="updated pseudo-labels")
    args = p.parse_args()

    out_dir = build(args)
    if args.push:
        push(out_dir, args.update, args.message)
    else:
        print("\nTo upload:")
        print(f"  python src\\build_kaggle_dataset.py --labels {args.labels} --push")


if __name__ == "__main__":
    main()
