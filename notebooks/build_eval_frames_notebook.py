"""Generate eval_frames_kaggle.ipynb - run the fine-tuned VLM over a packed
frame set (dvad-eval-frames) on a Kaggle T4.

Different question from the older eval_kaggle.ipynb, which asked "what is this
video's class" and collapsed everything to one label per video. That answers
Level 1 only. Levels 2 and 3 are 75 of the 100 marks and they ask WHEN, so this
notebook keeps every frame's label next to its timestamp and does NOT collapse
the long clips - the per-frame timeline is the product, and intervals are built
from it locally where they can be scored and tuned without burning GPU time.

It also matters that the VLM knows `stalled_or_broken_down_vehicle`. The
11-class appearance classifier deliberately excludes it (too few training
videos), which is why eval clips E022 and E025 were predicted stalled by the
rules and then produced "no window" - nothing downstream could time a class the
window-maker cannot emit.

    python notebooks\\build_eval_frames_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "id": f"c{len(CELLS):02d}",
                  "metadata": {}, "source": _src(text)})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "id": f"c{len(CELLS):02d}", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _src(text)})


md(r"""
# DVAD - VLM pass over the packed eval frames

Inference only. Loads Qwen2.5-VL-3B (4-bit) + a LoRA adapter and labels every
frame in the `dvad-eval-frames` pack.

Outputs `eval_frames.jsonl` (one row per frame, with `t_sec`) and
`eval_results.jsonl` (per-video rollup, Level 1 only). The long clips are left
uncollapsed on purpose - their intervals get built locally from the timeline.
""")

code(r'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
''')

code(r'''
!pip install -q --upgrade unsloth unsloth_zoo
import torch
assert torch.cuda.is_available(), "No GPU. Session options -> Accelerator -> GPU T4."
cc = torch.cuda.get_device_capability(0)
# P100 is sm_60 and current bitsandbytes 4-bit kernels are not built for it -
# it fails later with an opaque "no kernel image is available" instead.
assert cc[0] >= 7, f"Need sm_70+, got {cc}. Switch the accelerator to T4."
print("gpu:", torch.cuda.get_device_name(0), "cc:", cc)
''')

code(r'''
import glob, json
from pathlib import Path

ckpt = sorted(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True))
pack = sorted(glob.glob("/kaggle/input/**/eval.jsonl", recursive=True))
assert ckpt, "No adapter_config.json - attach the adapter dataset."
assert pack, "No eval.jsonl - attach the dvad-eval-frames dataset."
CKPT_DIR = Path(ckpt[0]).parent
PACK = Path(pack[0]).parent
print("adapter:", CKPT_DIR)
print("frames :", PACK)

rows = [json.loads(l) for l in (PACK / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by_level = {}
for r in rows:
    by_level[r["level"]] = by_level.get(r["level"], 0) + 1
print(f"{len(rows)} frames over {len({r['video_id'] for r in rows})} videos -> {by_level}")
''')

code(r'''
from unsloth import FastVisionModel
from peft import PeftModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
model = PeftModel.from_pretrained(model, str(CKPT_DIR))
FastVisionModel.for_inference(model)
print("adapter attached")
''')

code(r'''
from PIL import Image
import torch, time, json as _json

CLASSES = ("normal, traffic_accident, traffic_congestion, "
           "stalled_or_broken_down_vehicle, vehicle_blocking_traffic, "
           "wrong_way_driving, road_spill_or_debris, waterlogging_or_flood, "
           "fire, smoke, fighting_or_violence, loitering_or_suspicious_presence")

INSTRUCTION = (
    "You are a drone and CCTV video anomaly analyst. Look at this frame from a "
    "traffic or public-space camera and decide whether it shows an anomaly that "
    "an operator must respond to.\n"
    "Answer with JSON only, using exactly these keys:\n"
    '{"is_anomaly": true|false, "class_name": "<one of: ' + CLASSES + '>", '
    '"description_summary": "<one short sentence describing what you see>"}'
)

def predict(image_path, max_new_tokens=128):
    """Explicit kwargs + inference_mode + autocast. Passing image/text
    positionally on a 4-bit T4 previously produced NaN logits (decoded as
    repeated '!'), which looks exactly like a ruined fine-tune but is not."""
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

per_frame, t0 = [], time.time()
for i, r in enumerate(rows):
    raw = predict(PACK / r["image"])
    try:
        parsed = _json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception:
        parsed = None
    per_frame.append({"video_id": r["video_id"], "level": r["level"],
                      "t_sec": r.get("t_sec"), "frame_idx": r.get("frame_idx"),
                      "duration_s": r.get("duration_s"),
                      "raw": raw[:300], "parsed": parsed})
    if (i + 1) % 25 == 0 or i == 0:
        el = time.time() - t0
        rate = el / (i + 1)
        print(f"[{i+1}/{len(rows)}] {r['video_id']} t={r.get('t_sec')}s -> "
              f"{parsed.get('class_name') if parsed else 'UNPARSEABLE'}  "
              f"({rate:.1f}s/frame, ~{rate*(len(rows)-i-1)/60:.0f} min left)")

el = time.time() - t0
print(f"\ndone in {el/60:.1f} min, {el/max(len(rows),1):.2f}s/frame")

with open("/kaggle/working/eval_frames.jsonl", "w", encoding="utf-8") as f:
    for row in per_frame:
        f.write(_json.dumps(row) + "\n")
print("wrote eval_frames.jsonl")
''')

code(r'''
from collections import defaultdict, Counter

by_video = defaultdict(list)
for r in per_frame:
    by_video[r["video_id"]].append(r)

results = []
for vid, frames in sorted(by_video.items()):
    frames.sort(key=lambda f: f["t_sec"] if f["t_sec"] is not None else 0)
    labels = [f["parsed"]["class_name"] for f in frames
              if f["parsed"] and f["parsed"].get("class_name")]
    anomalies = [l for l in labels if l != "normal"]
    if anomalies:
        label = Counter(anomalies).most_common(1)[0][0]
        best = next(f for f in frames
                    if f["parsed"] and f["parsed"].get("class_name") == label)
        desc, is_anom = best["parsed"].get("description_summary", ""), True
    else:
        label, desc, is_anom = "normal", "", False
    results.append({"video_id": vid, "level": frames[0]["level"],
                    "class_name": label, "is_anomaly": is_anom,
                    "description_summary": desc, "n_frames": len(frames),
                    "n_anomaly_frames": len(anomalies),
                    "votes": dict(Counter(labels))})
    share = len(anomalies) / max(len(frames), 1)
    print(f"{vid:<6} L{frames[0]['level']} {len(anomalies):3d}/{len(frames):3d} "
          f"({share:4.0%}) -> {label:<32} {dict(Counter(labels))}")

with open("/kaggle/working/eval_results.jsonl", "w", encoding="utf-8") as f:
    for row in results:
        f.write(_json.dumps(row) + "\n")

ok = sum(1 for r in per_frame if r["parsed"] is not None)
print(f"\nparseable {ok}/{len(per_frame)}")
print("NOTE: for L2/L3 the per-video rollup above is only a sanity check - "
      "intervals are built locally from eval_frames.jsonl.")
''')

md(r"""
## Pull the results

    python src\push_notebook.py --pull --slug dvad-eval-frames-run --out C:\dvad\models\kaggle_evalframes

Then build the timeline into intervals with `src\vlm_windows.py`.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "eval_frames_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells, {out.stat().st_size/1024:.1f} KB)")

bad = [i for i, c in enumerate(CELLS)
       if c["cell_type"] == "code" and any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
