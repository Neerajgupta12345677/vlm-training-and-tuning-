"""Generate finetune_kaggle.ipynb.

Kept in the repo so the notebook can be regenerated on the day.

    python notebooks\\build_notebook.py

NBFORMAT GOTCHA (this cost a failed Kaggle run): every entry in a cell's
`source` list must END with "\\n". Using text.split("\\n") strips them, so Kaggle
joins the whole cell onto one line. A cell starting with "#" then becomes a
single comment and silently does NOTHING, while a code cell dies with
"SyntaxError: incomplete input". Always use splitlines(keepends=True).
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
# DVAD - Qwen2.5-VL-3B LoRA fine-tune (Kaggle, unattended)

Distills a large teacher VLM's anomaly judgements into a small VLM that runs on
a 4GB laptop GPU.

Pushed and run from the CLI via `src/push_notebook.py`, so this needs no clicks.
If running by hand: attach the `dvad-pseudo-labels` dataset, set Accelerator to
GPU, then **Save & Run All (Commit)**.
""")

code(r'''
# MUST run before torch is imported anywhere.
# Unsloth's open-source path is effectively single-GPU: if the T4 x2 accelerator
# is selected, pin to one device rather than debugging multi-GPU mid-hackathon.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("python:", sys.version.split()[0])
''')

code(r'''
# Unsloth install. If this fails, copy the install cell from Unsloth's official
# Kaggle notebook - their pinning changes with each Kaggle base image.
!pip install -q --upgrade unsloth unsloth_zoo

import importlib, torch
for mod in ("torch", "unsloth", "trl", "peft", "transformers", "accelerate", "bitsandbytes"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<16} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:<16} MISSING ({type(e).__name__})")
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
assert torch.cuda.is_available(), "No GPU. Set Accelerator to GPU in session options."
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"gpu: {name}  sm_{cc[0]}{cc[1]}")
# Fail fast and legibly rather than 3 cells later with an opaque CUDA error.
if cc[0] < 7:
    raise SystemExit(
        f"{name} is sm_{cc[0]}{cc[1]} (Pascal). Current bitsandbytes/Unsloth 4-bit "
        "kernels are not built for it and you will get 'no kernel image is "
        "available for execution on the device'. Use the T4 accelerator instead."
    )
''')

code(r'''
# SMOKE_TEST=True caps the run to a small slice of data and few steps -
# a ~10 min sanity check that the precision fix below actually holds before
# spending the full ~75 min GPU budget again. Flip to False for the real run.
SMOKE_TEST = False

CONFIG = dict(
    base_model   = "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit = True,     # QLoRA - fits comfortably on a single T4 (NOT P100)
    # 4371 real samples at effective batch 4 is ~1090 steps/epoch. 60 was sized
    # for the old n=15 synthetic set and would cover under 6% of one epoch.
    # SMOKE_TEST now runs 200 steps, not 40 - the v1 fp16 fix produced a clean
    # loss curve for 30 steps, then wild oscillation (0.004-0.65) from step 30
    # to a second NaN at step 141. A 40-step smoke test cannot catch a failure
    # that only appears past step 100; 200 does, at a real but bounded cost
    # (~35-45 min instead of another blind ~75 min full run).
    max_steps    = 200 if SMOKE_TEST else 800,
    # Also raised with the step count - 60 samples cycled 13x over 200 steps
    # risks memorising the smoke subset rather than testing real dynamics.
    smoke_n      = 200,
    batch_size   = 1,        # vision models with dynamic resolution spike; 1x4 is known-good
    grad_accum   = 4,
    # LOWERED AGAIN, 1e-4 -> 5e-5. The v1 fix (fp16 loss scaling + 1e-4) fixed
    # the FIRST failure mode (early NaN, no loss trace at all) cleanly - steps
    # 1-30 here are a smooth monotonic descent, proving it. The oscillation
    # that starts at step 30 is a SECOND, later-onset issue: once the model
    # gets good at the majority templated pattern (~25% of data is the
    # `normal` class, itself a short fixed-shape JSON target), it starts
    # nailing easy samples to near-zero loss while overshooting on harder
    # ones - a classic sign the LR is still too large for the fine-grained
    # regime once coarse fitting is done, even though it was fine at the start.
    lr           = 5e-5,
    # LENGTHENED, 5 -> 20 steps. A 5-step ramp reaches full LR while the model
    # is still only seeing the easiest, most templated samples, so the full
    # LR is already "too much" by the time harder samples arrive. A longer
    # ramp gives the optimiser more time before committing to full step size.
    warmup_steps = 20,
    max_grad_norm = 0.3,     # Unsloth's own recipe uses this for LoRA vision SFT
    lora_r       = 16,
    seed         = 3407,
    holdout      = 12,       # by-video val samples to eyeball after training
)
for k, v in CONFIG.items():
    print(f"{k:<14} {v}")
if SMOKE_TEST:
    print("\n*** SMOKE TEST MODE - small data, few steps, ~10 min. "
          "Set SMOKE_TEST=False for the real run once this passes. ***")
''')

code(r'''
# Find the attached dataset without hardcoding its slug OR its mount depth.
# Kaggle mounts datasets at different depths depending on the image - this run
# saw /kaggle/input/datasets/<owner>/<slug>/, which a two-level glob missed.
# Search recursively and never assume a layout.
import glob, json
from pathlib import Path

candidates = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
if not candidates:
    tree = glob.glob("/kaggle/input/**/*", recursive=True)[:40]
    raise SystemExit(
        "No train.jsonl under /kaggle/input. Attach the dvad-pseudo-labels dataset.\n"
        f"Found instead:\n  " + "\n  ".join(tree)
    )
TRAIN_JSONL = Path(candidates[0])
DATA_ROOT = TRAIN_JSONL.parent
print("dataset:", DATA_ROOT)

def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

rows = load_rows(TRAIN_JSONL)
# ROOT CAUSE of BOTH divergences (v2 and v3 failed at the identical step 141,
# across two different LR/warmup settings): src/build_vlm_dataset.py writes
# train.jsonl grouped by class (`for cls in sorted(per_class)`), so training
# marches through hundreds of consecutive near-duplicate examples per class
# before this shuffle existed - step 141 lands deep inside a long run of
# loitering_or_suspicious_presence samples sharing an near-identical caption.
# Long runs of repeated targets let the model overfit locally to near-zero
# loss, then destabilise on the next distinct example - consistent with the
# oscillation observed from step ~30 onward in both failed runs. Shuffling
# is not just a smoke-test convenience, it is required for the real run too.
import random as _random
_random.Random(CONFIG["seed"]).shuffle(rows)
if SMOKE_TEST:
    rows = rows[: CONFIG["smoke_n"]]
    print(f"[smoke test] subsampled to {len(rows)} rows")
else:
    print(f"[shuffled] {len(rows)} rows (fixes the class-block ordering that "
          f"produced the step-141 divergence in both prior runs)")
# src/build_vlm_dataset.py emits rows already in conversation form:
#   {image, video_id, class_name, messages:[user(image+text), assistant(json)]}
# The label lives in `class_name`, so anomaly balance is derived from it rather
# than a separate boolean - `normal` is the only benign class of the twelve.
pos = sum(1 for r in rows if r["class_name"] != "normal")
print(f"rows: {len(rows)}   balance: {pos} anomalous / {len(rows) - pos} normal")
if pos in (0, len(rows)):
    print("WARNING: single-class dataset - the model cannot learn a decision boundary.")

from collections import Counter
print("\n--- per-class ---")
for c, n in sorted(Counter(r["class_name"] for r in rows).items()):
    print(f"  {c:<34} {n}")

# The val split is held out BY VIDEO upstream. Holding out a slice of train
# instead would put near-duplicate frames of one clip on both sides and report
# an eval number that means nothing.
VAL_JSONL = DATA_ROOT / "val.jsonl"
val_rows = load_rows(VAL_JSONL) if VAL_JSONL.exists() else []
print(f"\nval rows (held out by video): {len(val_rows)}")

print("\n--- sample row ---")
s0 = rows[0]
print("image     :", s0["image"])
print("class     :", s0["class_name"])
print("target    :", s0["messages"][-1]["content"][0]["text"][:200])
''')

code(r'''
from unsloth import FastVisionModel, is_bfloat16_supported

# THE root-cause fix for tonight's loss=nan run. A T4 is compute capability
# 7.5 (Turing) with no native bfloat16 support, so Unsloth loads compute
# weights as float16. float16 has a much smaller dynamic range than bf16 and
# NEEDS automatic-mixed-precision loss scaling to stay numerically stable -
# but that only turns on when SFTConfig is told fp16=True explicitly. Without
# it (the previous run's config), HF Trainer runs plain float16 arithmetic
# with NO loss-scaling safety net at all: every official Unsloth notebook
# sets this pair, ours did not.
USE_FP16 = not is_bfloat16_supported()
USE_BF16 = is_bfloat16_supported()
print(f"precision: fp16={USE_FP16} bf16={USE_BF16} (bf16 support: {is_bfloat16_supported()})")

model, tokenizer = FastVisionModel.from_pretrained(
    CONFIG["base_model"],
    load_in_4bit               = CONFIG["load_in_4bit"],
    use_gradient_checkpointing = "unsloth",
)
print(type(model).__name__, "loaded")
''')

code(r'''
# finetune_vision_layers=False: we are teaching a decision policy and a strict
# JSON output format, which are language-side behaviours. Freezing the vision
# tower trains faster and keeps the adapter small - it has to come back down a
# slow link before Saturday. modules_to_save is omitted for the same reason.
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r              = CONFIG["lora_r"],
    lora_alpha     = CONFIG["lora_r"],
    lora_dropout   = 0,
    bias           = "none",
    random_state   = CONFIG["seed"],
    use_rslora     = False,
    loftq_config   = None,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
''')

code(r'''
from PIL import Image

def to_conversation(row):
    """Rows already carry the full `messages` conversation from
    src/build_vlm_dataset.py, whose instruction text and JSON target match the
    submission schema in src/submission.py exactly - so the student's raw
    output drops straight in with no parsing layer to drift.

    The only thing missing on disk is the decoded image: the user turn holds a
    bare {"type": "image"} placeholder, and Unsloth's collator needs a real PIL
    object in that slot. Everything else passes through untouched.
    """
    img = Image.open(DATA_ROOT / row["image"]).convert("RGB")
    msgs = []
    for m in row["messages"]:
        content = [({"type": "image", "image": img} if part.get("type") == "image" else part)
                   for part in m["content"]]
        msgs.append({"role": m["role"], "content": content})
    return {"messages": msgs}

train_dataset = [to_conversation(r) for r in rows]
# held_rows comes from the upstream by-VIDEO val split, never a slice of train:
# frames from one clip are near-duplicates, so holding out part of train would
# report a number that means nothing.
held_rows = val_rows[: CONFIG["holdout"]] if val_rows else []
print(f"train samples: {len(train_dataset)} | held out (by video): {len(held_rows)}")
''')

code(r'''
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainerCallback
import json as _json

FastVisionModel.for_training(model)

# Catches a diverging run at step 1, not step 800. Tonight's failed run only
# reported "loss: nan" in the FINAL summary, after burning the full ~75 min
# budget with no visibility into when it went wrong. This writes every step's
# loss to disk immediately (so it survives even if the kernel is later killed)
# and raises the instant a non-finite loss appears, rather than training on
# through it.
class NaNGuard(TrainerCallback):
    """Tolerates an ISOLATED non-finite loss, only halts on a PERSISTENT one.

    v2/v3/v4 each hit a single non-finite loss report after hundreds of steps
    of bounded, healthy-looking oscillation (many near-zero losses mixed with
    moderate ones, no rising trend beforehand) - the signature of the model
    becoming very confident on easy samples, then one batch pushing softmax
    cross-entropy into log(~0) territory in fp16. That is a normal, KNOWN
    hazard of fp16 training that PyTorch's own gradient scaler is specifically
    built to survive: it checks post-backward gradients for inf/nan and, if
    found, SKIPS just that optimizer step (leaving weights unchanged) and
    lowers its scale factor - no crash needed. The previous version of this
    guard stopped on the very first non-finite report, which meant it was
    aborting training before ever finding out whether the built-in scaler
    would have recovered on its own - v4 got 4.6x further than v2/v3 with the
    same "stop immediately" policy, so this was plausibly killing otherwise-
    fine runs at the exact edge of what fp16 training can tolerate.

    Now: only escalate to a hard stop if non-finite loss appears on
    CONSECUTIVE logged steps (PERSIST_THRESHOLD in a row) - that pattern means
    the model itself is stuck producing garbage every step, which the scaler
    cannot fix by skipping one update, and is the same signal the old guard
    was built to catch in the first place.
    """
    PERSIST_THRESHOLD = 5

    def __init__(self, path="/kaggle/working/loss_trace.jsonl"):
        self.f = open(path, "a", encoding="utf-8")
        self.consecutive_bad = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]
        self.f.write(_json.dumps({"step": state.global_step, "loss": loss}) + "\n")
        self.f.flush()
        import math
        bad = loss is None or math.isnan(loss) or math.isinf(loss)
        if not bad:
            self.consecutive_bad = 0
            return
        self.consecutive_bad += 1
        self.f.write(_json.dumps({"step": state.global_step,
                                  "warning": f"non-finite loss, consecutive={self.consecutive_bad}"}) + "\n")
        self.f.flush()
        print(f"\n!!! non-finite loss at step {state.global_step} "
              f"(consecutive={self.consecutive_bad}/{self.PERSIST_THRESHOLD}) !!!")
        if self.consecutive_bad >= self.PERSIST_THRESHOLD:
            self.f.write(_json.dumps({"step": state.global_step,
                                      "FATAL": f"{self.PERSIST_THRESHOLD} consecutive non-finite losses, stopping"}) + "\n")
            self.f.flush()
            print(f"!!! {self.PERSIST_THRESHOLD} consecutive non-finite losses - this is not "
                  f"recovering on its own, stopping now !!!")
            control.should_training_stop = True

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer, resize="min"),
    train_dataset = train_dataset,
    callbacks     = [NaNGuard()],
    args = SFTConfig(
        per_device_train_batch_size = CONFIG["batch_size"],
        gradient_accumulation_steps = CONFIG["grad_accum"],
        warmup_steps       = CONFIG["warmup_steps"],
        max_steps          = CONFIG["max_steps"],
        learning_rate      = CONFIG["lr"],
        logging_steps      = 1,
        optim              = "adamw_8bit",
        weight_decay       = 0.01,
        lr_scheduler_type  = "linear",
        seed               = CONFIG["seed"],
        output_dir         = "/kaggle/working/checkpoints",
        report_to          = "none",
        remove_unused_columns = False,   # required for vision SFT
        dataset_text_field    = "",
        dataset_kwargs        = {"skip_prepare_dataset": True},
        max_length            = 2048,
        # THE fix for tonight's divergence - see the model-loading cell above.
        fp16               = USE_FP16,
        bf16               = USE_BF16,
        max_grad_norm      = CONFIG["max_grad_norm"],
    ),
)
print("trainer ready")
print(f"  fp16={USE_FP16} bf16={USE_BF16} lr={CONFIG['lr']} max_grad_norm={CONFIG['max_grad_norm']}")
''')

code(r'''
import time, torch, math, json as _json2

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
stats = trainer.train()
elapsed_min = (time.time() - t0) / 60
print(f"\ntrained in {elapsed_min:.1f} min")
print(f"peak VRAM    : {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

# stats.training_loss is an AVERAGE over every logged step. With NaNGuard now
# tolerating isolated non-finite steps (see the guard's own docstring - v2/v3/
# v4 each hit a lone non-finite report after hundreds of otherwise-healthy
# steps, which PyTorch's own gradient scaler is built to survive by skipping
# just that update), even ONE tolerated bad step poisons this average to nan
# even on an otherwise-successful run. Recompute from the trace, ignoring
# non-finite entries, so a false alarm here doesn't discard a genuinely fine
# adapter over noise the guard was already designed to absorb.
finite_losses = []
fatal = False
with open("/kaggle/working/loss_trace.jsonl", encoding="utf-8") as f:
    for line in f:
        row = _json2.loads(line)
        if "FATAL" in row:
            fatal = True
        loss = row.get("loss")
        if isinstance(loss, (int, float)) and not (math.isnan(loss) or math.isinf(loss)):
            finite_losses.append(loss)

n_total = sum(1 for _ in open("/kaggle/working/loss_trace.jsonl", encoding="utf-8"))
n_bad = n_total - len(finite_losses)
print(f"loss steps   : {len(finite_losses)} finite / {n_total} total "
      f"({n_bad} tolerated non-finite, isolated events)")
if finite_losses:
    print(f"final loss   : {finite_losses[-1]:.4f}  (last finite value)")
    print(f"mean of last 20 finite losses: {sum(finite_losses[-20:])/len(finite_losses[-20:]):.4f}")

if fatal:
    raise SystemExit(
        "NaNGuard hit its persistence threshold (several non-finite losses IN A "
        "ROW, not just an isolated one) - this is the kind of divergence a "
        "gradient-scaler skip cannot fix on its own. Check "
        "/kaggle/working/loss_trace.jsonl for the exact step and pattern."
    )
if not finite_losses:
    raise SystemExit("Every single logged loss was non-finite - something is "
                     "wrong from step 1, not a late transient event.")
print("training completed - proceeding to the eval cell below.")
''')

code(r'''
# Save the adapter. This is the small artifact you cache locally as the offline
# fallback before judging.
import shutil
from pathlib import Path

OUT = Path("/kaggle/working/lora_adapter")
model.save_pretrained(str(OUT))
tokenizer.save_pretrained(str(OUT))

size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
print(f"adapter: {OUT}  ({size_mb:.1f} MB)")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT)}  {f.stat().st_size/1e6:.2f} MB")

shutil.make_archive("/kaggle/working/lora_adapter", "zip", str(OUT))
print("\nzip MB:", Path("/kaggle/working/lora_adapter.zip").stat().st_size / 1e6)
''')

code(r'''
# Sanity check: does the tuned model emit parseable JSON, and does it agree with
# the teacher on samples it never trained on?
import json
from PIL import Image

FastVisionModel.for_inference(model)

def predict(row, max_new_tokens=96):
    """Explicit kwargs + inference_mode + autocast.

    A previous run emitted "!!!!!!!" forever - that is token id 0 repeated,
    i.e. NaN logits, not a bad fine-tune. Passing the image/text positionally
    and generating outside autocast on a 4-bit T4 is the usual cause.
    """
    import torch

    img = Image.open(DATA_ROOT / row["image"]).convert("RGB")
    # Reuse the stored conversation verbatim, minus the assistant turn, so the
    # prompt at eval is byte-identical to the prompt seen in training.
    messages = [m for m in row["messages"] if m["role"] != "assistant"]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

probe = held_rows if held_rows else rows[:3]
agree = parseable = 0
for r in probe:
    raw = predict(r)
    truth = r["messages"][-1]["content"][0]["text"]
    print(f"\n--- {r['image']}  [{r['class_name']}]")
    print(f"teacher : {truth}")
    print(f"student : {raw[:300]}")
    try:
        pred = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        parseable += 1
        # Compare the CLASS, not just a boolean: the class is what the
        # submission is scored on, and macro-F1 weights all twelve equally.
        if pred.get("class_name") == r["class_name"]:
            agree += 1
            print("        -> class matches")
        else:
            print(f"        -> MISMATCH (said {pred.get('class_name')!r})")
    except Exception as e:
        print(f"        -> unparseable ({type(e).__name__})")

print(f"\nparseable: {parseable}/{len(probe)}   exact class match: {agree}/{len(probe)}")
print("An unparseable output is unusable downstream no matter how good the class "
      "is, so the JSON contract holding on every sample matters most here.")
''')

md(r"""
## Getting the adapter onto your laptop

```
python src\push_notebook.py --pull
```

Then unzip `lora_adapter.zip` into `C:\dvad\models\lora_adapter\`.

Do this **before Saturday** - venue wifi is the riskiest dependency in the plan,
and a cached adapter is what makes the demo wifi-independent.

### Serving it locally
- **For the demo**: keep the stock Ollama model for Stage 3. The pipeline is
  already wifi-independent, and the fine-tune becomes the "we distilled a
  teacher into a small model" story you *show* with eval numbers.
- **To put the tuned weights in the live path**: merge the adapter, convert to
  GGUF, register with Ollama via a Modelfile. Budget real time - VLM GGUF
  conversion is the fiddliest step here and is NOT on the critical path.
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

out = Path(__file__).parent / "finetune_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells, {out.stat().st_size/1024:.1f} KB)")

# Fail loudly here rather than on Kaggle: every source line except the last of a
# cell must end with a newline.
bad = [i for i, c in enumerate(CELLS)
       if any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad} - Kaggle would join them into one line")
print("newline check: OK")
