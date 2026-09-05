# Context-aware drone video anomaly detection

Real-time pipeline that flags **context-dependent** events in drone and CCTV
video (stopped vehicle in a live lane, smoke, flood, spill, congestion) on a
small GPU. A still frame cannot see motion, so cheap tracker arithmetic owns
the yes/no decision and a small vision-language model is invited in only when
something has already earned a look.

## Implementation document

The full write-up of what the system does, why it is built this way, and how
to run it is here:

**[document.pdf](document.pdf)** — implementation reference (also checked into this repo)

Same content in markdown: [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Architecture in one paragraph

Stage 1 detects and tracks every frame (aerial-tuned YOLO + ByteTrack).
Stage 2 keeps per-track dwell, speed and zone — **this layer owns the
boolean**. Stage 3 (appearance classifier + Qwen2.5-VL LoRA) runs on the
gated frames only; it may escalate a visible hazard, never silently clear a
stop the tracker measured. Stage 4 merges windows and writes the
timed, classified, explained alert.

```
video → detect/track → context rules (verdict) → gated VLM → fused event
```

## Fine-tuned LoRA

Private/public adapter on Hugging Face:

**https://huggingface.co/hackiit-neeraj/qwen25vl-ahc-lora-ckpt400**

- Repo root: batch-2 checkpoint-400 (Qwen2.5-VL-3B LoRA)
- `batch4/`: later rich-interval fine-tune

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_id, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, "hackiit-neeraj/qwen25vl-ahc-lora-ckpt400")
processor = AutoProcessor.from_pretrained(base_id)
```

## Setup

Python 3.11. Heavy artifacts (venv, weights, videos) live outside this repo
under `C:\dvad\` so OneDrive does not sync gigabytes. See `CLAUDE.md`.

```text
C:\dvad\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\dvad\.venv\Scripts\python.exe src\<script>.py --help
```

Every `src/` script is standalone and takes `--data_dir`. Swap the dataset
folder; do not hardcode paths.

## Repo map

| Path | What |
|---|---|
| [`document.pdf`](document.pdf) | Implementation reference |
| `src/` | Pipeline, scoring, training, export |
| `notebooks/` | Kaggle fine-tune / eval / GGUF merge |
| `submissions/` | Arena answer sheets |
| `ARCHITECTURE.md` | Stage-by-stage design |
| `HANDOVER.md` | Latest competition state |
| `PROGRESS.md` | Dated build log |

## License

Code in this repo is for the FlytBase AHC build. The LoRA adapter follows
the Apache-2.0 license of Qwen2.5-VL.
