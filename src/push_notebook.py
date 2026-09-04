"""Push the training notebook to Kaggle and run it unattended (Stage 7).

This is the "Save & Run All (Commit)" flow, driven from the CLI so it is
repeatable on the day rather than a sequence of browser clicks.

    python src\\push_notebook.py --push          # upload + start a GPU run
    python src\\push_notebook.py --status        # poll it
    python src\\push_notebook.py --pull          # download the adapter when done

The kernel is private, GPU-enabled, internet-enabled (Unsloth installs at
runtime) and has the pseudo-label dataset attached.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from build_kaggle_dataset import kaggle_username
from common import MODELS_DIR, PROJECT_DIR

DEFAULT_NOTEBOOK = PROJECT_DIR / "notebooks" / "finetune_kaggle.ipynb"
PUSH_DIR = Path(r"C:\dvad\outputs\kernel_push")


def kaggle_cmd() -> list[str]:
    exe = Path(sys.executable).parent / "kaggle.exe"
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "kaggle"]


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    # Two separate Windows encoding problems, both load-bearing:
    #  * utf-8/replace here, because the CLI emits bytes cp1252 cannot DECODE.
    #  * PYTHONUTF8=1 in the child, because the CLI itself dies with
    #    UnicodeEncodeError while WRITING the kernel log, leaving a 0-byte file
    #    and hiding the very traceback you need.
    import os

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def build_push_dir(args) -> Path:
    user = kaggle_username()
    if not user:
        raise SystemExit("Could not resolve the Kaggle username. Run: python src\\setup_kaggle.py")
    notebook = Path(args.notebook) if args.notebook else DEFAULT_NOTEBOOK
    if not notebook.exists():
        raise SystemExit(f"Notebook not found: {notebook}")

    # One push dir per kernel, or two kernels would overwrite each other's code.
    push_dir = PUSH_DIR / args.slug
    if push_dir.exists():
        shutil.rmtree(push_dir)
    push_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook, push_dir / notebook.name)
    NOTEBOOK = notebook

    meta = {
        "id": f"{user}/{args.slug}",
        "title": args.title,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,          # required: Unsloth will not run on CPU
        # MUST be T4, not P100. Kaggle's default GPU is a P100 (sm_60, Pascal),
        # and current bitsandbytes/Unsloth 4-bit kernels are no longer built for
        # it: the run dies with "CUDA error: no kernel image is available for
        # execution on the device". Kaggle's own docs warn P100 is incompatible
        # with the default image. T4 is sm_75 and works.
        "machine_shape": args.accelerator,
        "enable_internet": True,     # required: the notebook pip-installs Unsloth
        # The YOLO notebook downloads VisDrone itself, so it needs no attachment.
        "dataset_sources": [] if args.no_dataset else [f"{user}/{args.dataset}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {meta['id']}  gpu={args.accelerator}  dataset={meta['dataset_sources']}")
    return push_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Push and run the Kaggle training notebook.")
    p.add_argument("--push", action="store_true", help="Upload and start a run.")
    p.add_argument("--status", action="store_true", help="Print the current run status.")
    p.add_argument("--wait", action="store_true", help="Poll until the run finishes.")
    p.add_argument("--pull", action="store_true", help="Download outputs (the LoRA adapter).")
    p.add_argument("--log", action="store_true",
                   help="Download and print the run log, jumping to the traceback.")
    # Kaggle derives the URL slug from the TITLE, not the id. Keep them
    # consistent or --status/--pull will 404 on a kernel that exists.
    p.add_argument("--slug", default="dvad-finetune-qwen25vl")
    p.add_argument("--title", default="dvad finetune qwen25vl")
    p.add_argument("--notebook", default=None, help="Notebook to push (default: finetune_kaggle.ipynb).")
    p.add_argument("--dataset", default="dvad-pseudo-labels")
    p.add_argument("--no-dataset", action="store_true",
                   help="Attach no dataset (the YOLO notebook fetches VisDrone itself).")
    p.add_argument("--accelerator", default="NvidiaTeslaT4",
                   choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                   help="Leave on T4. P100 is sm_60 and current 4-bit kernels "
                        "are not built for it.")
    p.add_argument("--out", default=str(MODELS_DIR / "kaggle_output"))
    p.add_argument("--poll", type=int, default=60, help="Seconds between status polls.")
    p.add_argument("--timeout-min", type=int, default=90)
    args = p.parse_args()

    user = kaggle_username()
    kid = f"{user}/{args.slug}"

    if args.push:
        d = build_push_dir(args)
        code, out = run(kaggle_cmd() + ["kernels", "push", "-p", str(d)])
        print(out)
        if code != 0:
            raise SystemExit("[push] failed - see the message above.")
        print(f"\n[ok] running at https://www.kaggle.com/code/{kid}")
        print("It runs unattended; you can close everything.")

    if args.status or args.wait:
        deadline = time.time() + args.timeout_min * 60
        while True:
            code, out = run(kaggle_cmd() + ["kernels", "status", kid])
            print(f"[{time.strftime('%H:%M:%S')}] {out}")
            low = out.lower()
            if any(s in low for s in ("complete", "error", "cancel")):
                if "error" in low:
                    print("\n[!] The run errored. Fetch the log with:")
                    print(f"    kaggle kernels output {kid} -p {args.out}")
                break
            if not args.wait or time.time() > deadline:
                break
            time.sleep(args.poll)

    if args.log:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remove any 0-byte log left behind by a previous encoding crash.
        for stale in out_dir.glob("*.log"):
            if stale.stat().st_size == 0:
                stale.unlink()
        run(kaggle_cmd() + ["kernels", "output", kid, "-p", str(out_dir)])
        logs = sorted(out_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not logs:
            print("[!] No log downloaded. The run may not have started yet.")
            return
        raw = logs[0].read_text(encoding="utf-8", errors="replace")
        try:
            txt = "".join(e.get("data", "") for e in json.loads(raw))
        except Exception:  # noqa: BLE001 - plain-text logs are fine too
            txt = raw
        marker = txt.find("Exception encountered")
        if marker < 0:
            marker = txt.find("Traceback (most recent call last)")
        print(txt[max(0, marker - 400): marker + 3000] if marker >= 0 else txt[-3000:])

    if args.pull:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        code, out = run(kaggle_cmd() + ["kernels", "output", kid, "-p", str(out_dir)])
        print(out)
        files = sorted(f for f in out_dir.rglob("*") if f.is_file())
        print(f"\n{len(files)} file(s) in {out_dir}:")
        for f in files[:25]:
            print(f"  {f.relative_to(out_dir)}  {f.stat().st_size/1e6:.2f} MB")
        adapter = [f for f in files if f.suffix == ".zip" or "adapter" in f.name.lower()]
        if adapter:
            print(f"\n[ok] adapter artifact: {adapter[0]}")
            print("Unzip it into C:\\dvad\\models\\lora_adapter\\ to cache it offline.")


if __name__ == "__main__":
    main()
