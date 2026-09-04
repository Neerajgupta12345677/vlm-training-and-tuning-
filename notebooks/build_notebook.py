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
CONFIG = dict(
    base_model   = "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit = True,     # QLoRA - fits comfortably on a single T4 (NOT P100)
    max_steps    = 60,       # raise once you see the loss curve behave
    batch_size   = 1,
    grad_accum   = 4,
    lr           = 2e-4,
    lora_r       = 16,
    seed         = 3407,
    holdout      = 3,        # samples withheld to eyeball after training
)
for k, v in CONFIG.items():
    print(f"{k:<14} {v}")
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

rows = [json.loads(l) for l in TRAIN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
pos = sum(1 for r in rows if r["anomalous"])
print(f"rows: {len(rows)}   class balance: {pos} anomalous / {len(rows) - pos} benign")
if pos in (0, len(rows)):
    print("WARNING: single-class dataset - the model cannot learn a decision boundary.")
print("\n--- sample row ---")
s = rows[0]
print("image      :", s["image"])
print("instruction:", s["instruction"][:200])
print("target     :", s["target"])
''')

code(r'''
from unsloth import FastVisionModel

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
    """`system` and `instruction` came from src/vlm_reason.py via
    build_kaggle_dataset.py, so the wording is byte-identical to what Stage 3
    sends at inference. `target` is the exact JSON the student must emit."""
    img = Image.open(DATA_ROOT / row["image"]).convert("RGB")
    return {"messages": [
        {"role": "system",    "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user",      "content": [{"type": "image", "image": img},
                                          {"type": "text",  "text": row["instruction"]}]},
        {"role": "assistant", "content": [{"type": "text",  "text": row["target"]}]},
    ]}

holdout_n = min(CONFIG["holdout"], max(0, len(rows) - 4))
train_rows, held_rows = (rows[holdout_n:], rows[:holdout_n]) if holdout_n else (rows, [])
train_dataset = [to_conversation(r) for r in train_rows]
print(f"train samples: {len(train_dataset)} | held out: {len(held_rows)}")
''')

code(r'''
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer, resize="min"),
    train_dataset = train_dataset,
    args = SFTConfig(
        per_device_train_batch_size = CONFIG["batch_size"],
        gradient_accumulation_steps = CONFIG["grad_accum"],
        warmup_steps       = 5,
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
    ),
)
print("trainer ready")
''')

code(r'''
import time, torch
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
stats = trainer.train()
print(f"\ntrained in {(time.time()-t0)/60:.1f} min")
print(f"final loss   : {stats.training_loss:.4f}")
print(f"peak VRAM    : {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
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
    messages = [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user",   "content": [{"type": "image"},
                                       {"type": "text", "text": row["instruction"]}]},
    ]
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
    print(f"\n--- {r['image']}")
    print(f"teacher : {r['target']}")
    print(f"student : {raw[:300]}")
    try:
        pred = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        parseable += 1
        if bool(pred.get("anomalous")) == bool(r["anomalous"]):
            agree += 1
            print("        -> agrees with teacher")
        else:
            print("        -> DISAGREES with teacher")
    except Exception as e:
        print(f"        -> unparseable ({type(e).__name__})")

print(f"\nparseable: {parseable}/{len(probe)}   agreement: {agree}/{len(probe)}")
print("(A tiny pseudo-label set will not give strong numbers - what matters is "
      "that the JSON contract holds and the training loop completes.)")
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
