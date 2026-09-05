"""Generate eval_kaggle.ipynb - loads the base VLM + checkpoint-500 adapter
and runs inference on the REAL public test set (one frame per video), so the
fine-tune's end-to-end contribution can be scored through the same
score_submission.py pipeline as the classifier, not just eyeballed on a
handful of held-out training samples.

This is deliberately an EVAL-ONLY notebook - no training, no gradient steps,
so it costs a few minutes of GPU time rather than another ~75 min run, and it
answers the actual question this session needs answered: does checkpoint-500
(reached independently twice, healthy internal trainer state, but from an
INTERRUPTED run that never finished all 800 steps) produce a macro-F1 on the
real test set that's worth wiring into the cascade, or is 0.262 (the
classifier alone) still the number to submit.

    python notebooks\\build_eval_notebook.py
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
# DVAD - checkpoint-500 eval on the real public test set

EVAL ONLY - no training. Loads the base Qwen2.5-VL-3B (4-bit) + the
checkpoint-500 LoRA adapter (from the interrupted fine-tune run) and runs
inference on 33 real test-video frames, so this can be scored macro-F1
against ground_truth.csv exactly like the classifier was.
""")

code(r'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
''')

code(r'''
!pip install -q --upgrade unsloth unsloth_zoo
import torch
assert torch.cuda.is_available(), "No GPU. Set Accelerator to GPU in session options."
print("gpu:", torch.cuda.get_device_name(0), "cc:", torch.cuda.get_device_capability(0))
''')

code(r'''
# Locate the two attached datasets without hardcoding mount depth.
import glob, json
from pathlib import Path

ckpt_candidates = sorted(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True))
eval_candidates = sorted(glob.glob("/kaggle/input/**/eval.jsonl", recursive=True))
assert ckpt_candidates, "No adapter_config.json found - attach the dvad-vlm-ckpt500 dataset."
assert eval_candidates, "No eval.jsonl found - attach the dvad-vlm-eval-test dataset."
CKPT_DIR = Path(ckpt_candidates[0]).parent
EVAL_ROOT = Path(eval_candidates[0]).parent
print("checkpoint:", CKPT_DIR)
print("eval data :", EVAL_ROOT)

rows = [json.loads(l) for l in (EVAL_ROOT / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{len(rows)} test video frames to evaluate")
''')

code(r'''
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
print("base model loaded")

from peft import PeftModel
model = PeftModel.from_pretrained(model, str(CKPT_DIR))
print("checkpoint-500 adapter attached")
FastVisionModel.for_inference(model)
''')

code(r'''
INSTRUCTION = (
    "You are a drone and CCTV video anomaly analyst. Look at this frame from a "
    "traffic or public-space camera and decide whether it shows an anomaly that "
    "an operator must respond to.\n"
    "Answer with JSON only, using exactly these keys:\n"
    '{"is_anomaly": true|false, "class_name": "<one of: normal, traffic_accident, '
    "traffic_congestion, stalled_or_broken_down_vehicle, vehicle_blocking_traffic, "
    "wrong_way_driving, road_spill_or_debris, waterlogging_or_flood, fire, smoke, "
    'fighting_or_violence, loitering_or_suspicious_presence>", '
    '"description_summary": "<one short sentence describing what you see>"}'
)

def predict(image_path, max_new_tokens=96):
    """Same explicit-kwargs + inference_mode + autocast pattern proven during
    training-time eval - positional image/text args on a 4-bit T4 previously
    produced NaN logits ("!!!!!!!" repeated), not a bad fine-tune."""
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

from PIL import Image
import torch, time, json as _json

results = []
t0 = time.time()
for i, r in enumerate(rows):
    img_path = EVAL_ROOT / r["image"]
    raw = predict(img_path)
    parsed = None
    try:
        parsed = _json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception as e:
        parsed = None
    results.append({"video_id": r["video_id"], "raw": raw[:300], "parsed": parsed})
    label = parsed.get("class_name") if parsed else "UNPARSEABLE"
    print(f"[{i+1}/{len(rows)}] {r['video_id']:<8} -> {label}")

elapsed = time.time() - t0
print(f"\ndone in {elapsed/60:.1f} min, {elapsed/len(rows):.1f}s/video average")

with open("/kaggle/working/eval_results.jsonl", "w", encoding="utf-8") as f:
    for row in results:
        f.write(_json.dumps(row) + "\n")

parseable = sum(1 for r in results if r["parsed"] is not None)
print(f"parseable: {parseable}/{len(results)}")
''')

md(r"""
## Download the results

    kaggle kernels output <owner>/<slug> -p C:\dvad\models\kaggle_eval_output

Then locally: build a predictions.csv from eval_results.jsonl and score it
with score_submission.py against the real ground_truth.csv - the same
apples-to-apples comparison used for the classifier.
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

out = Path(__file__).parent / "eval_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells, {out.stat().st_size/1024:.1f} KB)")

bad = [i for i, c in enumerate(CELLS)
       if c["cell_type"] == "code" and any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
