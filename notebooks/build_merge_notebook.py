"""Generate merge_gguf.ipynb - merge the QLoRA adapter and export a GGUF for Ollama.

Why this runs on Kaggle rather than locally
-------------------------------------------
The adapter (`dvad-vlm-ckpt400b2`, batch-2 checkpoint-400, macro-F1 0.437) has
never influenced a submitted prediction. Local inference is Ollama running the
*stock* `qwen2.5vl:3b` GGUF, and an HF/PEFT adapter cannot be applied to a GGUF
at runtime - it has to be merged into fp16 weights and re-exported.

That merge cannot happen on the laptop:
  * `Qwen/Qwen2.5-VL-3B-Instruct` in fp16 is ~7.5GB and is not cached (the HF
    cache is 260KB).
  * Merging needs ~6.2GB of RAM. The machine has 7.3GB TOTAL, ~1GB free.
  * transformers / peft / accelerate are not even installed in the venv.

Kaggle's T4 has 16GB VRAM, ~30GB disk and fast internet, so the whole chain runs
there and only the finished quantised GGUF comes back.

Two things that make this cheaper than it looks:
  * The adapter targets LANGUAGE layers only (task_type CAUSAL_LM; the
    target_modules regex matches `language`/`text` attention and MLP
    projections). The vision tower is untouched.
  * It was trained against `unsloth/qwen2.5-vl-3b-instruct-unsloth-bnb-4bit`,
    which is a 4-bit copy of the official base with unchanged module names, so
    it applies cleanly to the official fp16 weights. The notebook VERIFIES that
    rather than assuming it - if PEFT injects zero LoRA layers it stops, instead
    of silently exporting the un-finetuned base and letting us ship a "fine-tuned"
    model that is nothing of the sort.

    python notebooks\\build_merge_notebook.py
    python src\\push_notebook.py --push --slug dvad-merge-gguf ^
        --title "dvad merge gguf" --notebook notebooks\\merge_gguf.ipynb ^
        --dataset dvad-vlm-ckpt400b2
"""

import json
from pathlib import Path

CELLS = []


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(t): CELLS.append({"cell_type": "markdown", "id": f"m{len(CELLS):02d}",
                         "metadata": {}, "source": _src(t)})


def code(t): CELLS.append({"cell_type": "code", "id": f"m{len(CELLS):02d}",
                           "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _src(t)})


md(r"""
# Merge QLoRA adapter into Qwen2.5-VL-3B and export GGUF for Ollama

Chain: fp16 base -> apply LoRA -> merge -> convert to GGUF -> quantise Q4_K_M.

Only the quantised model and the vision projector are kept in the output, so the
local pull stays around 2-3GB instead of the ~15GB of intermediates.
""")

code(r'''
import os, subprocess, sys, shutil, json
from pathlib import Path

BASE     = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER  = None          # resolved below from the attached dataset
MERGED   = Path("/kaggle/temp/merged")
WORK     = Path("/kaggle/working")
LLAMA    = Path("/kaggle/temp/llama.cpp")

# Intermediates go in /kaggle/temp, NOT /kaggle/working: anything left in
# working becomes kernel output and would make the download ~15GB.
MERGED.parent.mkdir(parents=True, exist_ok=True)

for root in Path("/kaggle/input").glob("*"):
    hit = list(root.rglob("adapter_config.json"))
    if hit:
        ADAPTER = hit[0].parent
        break
assert ADAPTER is not None, "no adapter_config.json under /kaggle/input"
print("adapter:", ADAPTER)
print(json.dumps({k: v for k, v in json.load(open(ADAPTER / "adapter_config.json")).items()
                  if k in ("base_model_name_or_path", "r", "lora_alpha", "task_type")}, indent=2))
''')

code(r'''
# Pinned so a surprise upstream release cannot change behaviour mid-hackathon.
!pip -q install "transformers==4.51.3" "peft==0.14.0" "accelerate==1.3.0" 2>&1 | tail -2
print("installed")
''')

code(r'''
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# CPU + fp16 keeps peak RAM near the 6.2GB weight size. Kaggle has ~30GB, so
# this is comfortable here even though it is exactly what fails on the laptop.
print("loading base (this pulls ~7.5GB the first time)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(BASE)
print("base loaded:", sum(p.numel() for p in model.parameters()) / 1e9, "B params")
''')

code(r'''
model = PeftModel.from_pretrained(model, str(ADAPTER), torch_dtype=torch.float16)

# VERIFY the adapter actually attached. The adapter was trained against
# unsloth's 4-bit copy of this base; if module naming ever diverged, PEFT would
# match nothing, merge would be a no-op, and we would ship the stock model
# believing it was fine-tuned. Fail loudly instead.
lora_mods = [n for n, _ in model.named_modules() if "lora_A" in n]
print(f"LoRA modules injected: {len(lora_mods)}")
assert lora_mods, "adapter matched ZERO modules - refusing to export the base as 'fine-tuned'"
print("sample:", lora_mods[:3])
''')

code(r'''
print("merging...")
model = model.merge_and_unload()
MERGED.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED, safe_serialization=True)
processor.save_pretrained(MERGED)
del model
import gc; gc.collect()
print("merged ->", MERGED)
!du -sh {MERGED}
''')

code(r'''
# llama.cpp supplies both the HF->GGUF converter and the quantiser.
if not LLAMA.exists():
    !git clone -q --depth 1 https://github.com/ggerganov/llama.cpp {LLAMA}
!pip -q install -r {LLAMA}/requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -1
print("llama.cpp ready")
''')

code(r'''
F16 = Path("/kaggle/temp/qwen25vl-3b-ahc-f16.gguf")
# Two artefacts come out of a vision model: the language GGUF and the mmproj
# vision projector. --mmproj writes the projector on its own pass.
!python {LLAMA}/convert_hf_to_gguf.py {MERGED} --outfile {F16} --outtype f16
!python {LLAMA}/convert_hf_to_gguf.py {MERGED} --outfile {WORK}/mmproj-qwen25vl-3b-f16.gguf --mmproj
!ls -la {F16} {WORK}/mmproj-qwen25vl-3b-f16.gguf
''')

code(r'''
# Q4_K_M matches the quantisation of the stock ollama qwen2.5vl:3b this replaces,
# so latency and VRAM stay where they were measured (27-45s/call on a GTX 1650).
Q4 = WORK / "qwen25vl-3b-ahc-q4_k_m.gguf"
!cmake -S {LLAMA} -B {LLAMA}/build -DLLAMA_CURL=OFF > /dev/null 2>&1 && \
 cmake --build {LLAMA}/build --target llama-quantize -j4 > /dev/null 2>&1
QUANT = f"{LLAMA}/build/bin/llama-quantize"
!{QUANT} {F16} {Q4} Q4_K_M
!ls -la {WORK}
''')

code(r'''
# The Modelfile ships with the weights so the local step is a single command
# and nobody has to remember the projector line.
(WORK / "Modelfile").write_text(
    "FROM ./qwen25vl-3b-ahc-q4_k_m.gguf\n"
    "FROM ./mmproj-qwen25vl-3b-f16.gguf\n"
    'PARAMETER temperature 0.1\n'
    'PARAMETER num_ctx 4096\n'
)

# Guard the download size: anything stray left in /kaggle/working is output.
print("FINAL OUTPUT:")
tot = 0
for f in sorted(WORK.rglob("*")):
    if f.is_file():
        tot += f.stat().st_size
        print(f"  {f.name:<44} {f.stat().st_size/1e9:6.2f} GB")
print(f"  {'TOTAL':<44} {tot/1e9:6.2f} GB")
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "merge_gguf.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")

bad = [i for i, c in enumerate(CELLS)
       if any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
