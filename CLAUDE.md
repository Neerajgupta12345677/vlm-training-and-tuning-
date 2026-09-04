# PROJECT: Drone Video Anomaly Detection (Hackathon — FlytBase, Sept 5, 2026)

## Mission
Build a real-time pipeline that flags context-dependent anomalies (stopped
vehicle on highway, smoke, unusual crowd behavior) in drone video, using a
SMALL vision-language model, cheaply enough to scale across many feeds.
Judged Sept 5, 2026, FlytBase Labs, Pune. Real dataset arrives day-of —
pipeline must be dataset-agnostic (swap a folder path, nothing else).

## Hardware and environment split (hard constraints — do not violate)
Verified from Windows "About" panel — trust these numbers, not guesses:
- Laptop: AMD Ryzen 5 5600H, **NVIDIA GeForce GTX 1650 (4GB VRAM)**,
  8GB RAM (7.34GB usable), Windows 11. There is also an integrated AMD
  Radeon iGPU (496MB) — it is NOT CUDA-capable, ignore it entirely.
- The GTX 1650 (Turing, TU117 die) has **no dedicated Tensor Cores**
  (that's an RTX/Ampere-class feature — this card is the GTX 16-series,
  not RTX). Local FP16/INT8 inference will be noticeably slower than a
  Tensor-Core GPU. Confirmed real-world result on this exact card: ~1
  token/sec for small quantized LLMs via llama.cpp+CUDA. Set expectations
  accordingly — do not assume snappy local VLM calls per-frame.
- 8GB system RAM is tight. During dev and especially during the Saturday
  demo: close unnecessary background apps, avoid running the conda env +
  browser + video decode + VLM all fighting for RAM at once, and prefer
  streaming video frame-by-frame over loading full videos into memory.
- Storage: 477GB total, ~329GB free — not a constraint, don't worry about it.
- LOCAL = **native Windows, no WSL2** (decided Sept 3 after audit; see
  "Why no WSL2" below). Pipeline dev, detector/tracker, local quantized
  INFERENCE ONLY. Never attempt LoRA/QLoRA training locally — 4GB VRAM
  cannot hold weights + gradients + optimizer state for this, full stop.
- Verified driver state (Sept 3, 2026): NVIDIA driver 592.27, CUDA 13.1
  capable, GTX 1650 reporting 4096MiB with 0MiB in use. No driver work
  needed. The GPU is compute capability 7.5 (sm_75) — pin PyTorch to a
  CUDA build that still ships sm_75 kernels (cu128 verified good; treat
  the newest cu13x builds as suspect until a matmul actually runs).
- TRAINING = Kaggle Notebooks only. ~30 GPU-hours/week, sessions up to ~12
  hours, no credit card needed.
- **Use the T4 accelerator, NEVER the P100** (`machine_shape: "NvidiaTeslaT4"`).
  Verified the hard way on 2026-09-04: Kaggle's default GPU is a P100 (sm_60,
  Pascal), and current bitsandbytes/Unsloth 4-bit kernels are no longer built
  for it. The run dies at `get_peft_model` with
  `CUDA error: no kernel image is available for execution on the device`.
  Kaggle's own metadata docs warn P100 is incompatible with the default image.
  The notebook now asserts `compute capability >= 7` and fails with a readable
  message rather than an opaque CUDA error three cells later.
- If T4x2 is selected, restrict to one GPU via
  `os.environ["CUDA_VISIBLE_DEVICES"]="0"` — Unsloth's open-source path is
  still effectively single-GPU in practice; don't risk debugging a multi-GPU
  setup mid-hackathon.
- vLLM is Linux-only — it is NOT used anywhere in this project locally.
  If you ever think you need it, you don't; local inference is Ollama.

## Why no WSL2 (decided Sept 3, 2026 — do not re-litigate)
The original plan mandated WSL2. That was dropped after the Stage 0 audit:
- WSL was not installed, and installing it needs elevation + a reboot.
- The machine had 0.5GB of 7.35GB RAM free at audit time. WSL2 claims
  ~50% of RAM by default (~3.6GB here), which this laptop cannot spare.
- Nothing left in the LOCAL plan needs Linux: vLLM was already excluded,
  and Unsloth only ever runs on Kaggle. The justification had dissolved.
- Native OpenCV demo windows render directly instead of through WSLg,
  which matters for the live demo.

## Paths (important — this project is inside a OneDrive folder)
Code lives in the OneDrive project dir; **large/generated artifacts must
NOT**, or OneDrive will sync gigabytes and can saturate venue wifi mid-demo.
- Code / docs / notebooks (synced, small): this directory.
- Everything heavy lives OUTSIDE OneDrive under `C:\dvad\`:
  - `C:\dvad\.venv`  — Python 3.11 venv (interpreter:
    `C:\dvad\.venv\Scripts\python.exe`)
  - `C:\dvad\models` — model weights / GGUF / LoRA adapters
  - `C:\dvad\data`   — videos, frames, pseudo-labels
  - `C:\dvad\outputs`— run artifacts, annotated video, eval reports
- NOTE: the system default `py` is Python **3.13t (freethreaded)**, which
  has almost no ML wheels. Never use bare `python`/`py`. Always invoke
  `C:\dvad\.venv\Scripts\python.exe` explicitly.
- Assume unreliable internet at the venue after ~10am Saturday (Sept 5).
  All Kaggle training must be front-loaded **tonight and tomorrow (Sept
  3–4)** — that's the entire runway, it is not a lot of time. A fine-tuned
  adapter must be downloaded and cached locally as an offline fallback
  before judging starts.

## Kaggle workflow (specific, don't improvise)
- Local machine has `kaggle` CLI installed in the venv and authenticated
  via kaggle.json at `C:\Users\pragy\.kaggle\kaggle.json`.
- Datasets are uploaded from the local venv via `kaggle datasets create` /
  `kaggle datasets version` — never rely on manual browser drag-and-drop,
  it must be scriptable so the real Saturday dataset can be pushed fast.
- Training notebook must be written top-to-bottom, non-interactive, so it
  can be run via "Save & Run All (Commit)" and finish unattended in the
  background while local work continues.
- After training, the LoRA adapter (small file) is downloaded from the
  Kaggle notebook's output and copied into models/ locally for inference.

## Architecture (decided, do not redesign without asking)
Three-stage pipeline, decoupling cheap "noticing" from expensive "reasoning":
1. STAGE 1 — every frame: YOLO26n (try first — Ultralytics' Jan 2026
   release, NMS-free, faster than YOLO11n especially on CPU) or YOLO11n
   (fallback if YOLO26 has version/compatibility issues) detection +
   ByteTrack (via `supervision` library) tracking. No VLM here. Must run
   real-time on the GTX 1650 / CPU.
2. STAGE 2 — every frame, no model calls: per-track state (dwell time,
   speed, zone type e.g. highway/parking/shoulder). Zone comes from a
   one-time per-video calibration step, not per-frame inference.
3. STAGE 3 — event-triggered only (measured 0.19-1.9% of frames): when Stage 2
   crosses a threshold, consult the small VLM with the frame + context string.
   This gating is what makes the pipeline viable at all on a Tensor-Core-less
   4GB card — do not loosen the trigger threshold without re-checking latency.

## Division of labour in Stage 3 (measured Sept 4 — do not undo this)
The original design asked the VLM "is this anomalous?". That does not work,
for a reason that is structural rather than a prompt bug:

**A single still frame does not contain motion.** A moving car and a stopped
car are pixel-identical in one image. The tracker already knows the answer
with certainty; the VLM can only guess at it.

Measured across four prompt revisions on the injected stopped-truck clip
(3 positives / 3 negatives each time):
- qwen2.5vl:3b — all-true, then all-false, then all-true again. It tracked
  whichever rule the prompt emphasised, not the input facts. 3/6 every time,
  i.e. chance. Its *prose* was consistently accurate ("the truck is stationary
  in a live driving lane, surrounding traffic still flowing") while its boolean
  contradicted its own prose.
- moondream (1.8B) — cannot follow multi-clause conditional instructions at
  all, and collapses under schema-constrained decoding.
- Negation is especially bad: a prompt containing three "is NOT an anomaly"
  clauses and one "is an incident" made the model answer "not an incident" for
  a textbook incident.

So Stage 3 is split by what each component is actually good at:
- **Stage 2 rules own the boolean.** Dwell time, zone and neighbour state are
  arithmetic. `Event.rule_anomalous` / `rule_severity` carry that verdict.
- **The VLM only observes** (`SceneObservation`: hazard_visible, hazard_type,
  surroundings). Descriptive tasks it does well, and schema-constrained
  decoding is safe for description.
- **The VLM may escalate, never silently overturn.** A visible hazard (fire,
  smoke, collision, person on the carriageway, crowd) promotes an event to
  anomalous at severity >=0.9. A model saying "nothing visible" cannot clear a
  stop the tracker measured. See `vlm_reason.combine()`.

`--decision` selects the mode: `hybrid` (default), `rules` (no model),
`vlm` (model decides everything — kept only so the gap stays measurable).

Distillation is the path to making the small model trustworthy on the full
judgement: the teacher labels the full verdict, the student is trained on it,
and eval.py measures whether the gap closes. Until it does, rules own the
boolean.

## Measured performance (4K 25fps source, GTX 1650 — Sept 4)
- Stage 1 alone, threaded decode: **26.6 fps, 1.06x real-time on 4K**.
- Threaded decode matters: serial decode+detect was 20.0 fps because 4K decode
  (23.5ms/frame) and detection (29.3ms/frame) added up. Overlapping them hides
  the decode. Never revert `iter_frames_threaded`.
- Full pipeline, `--decision rules`, stride 2: **14.9 fps vs 12.5 needed —
  real-time, ~1.19 feeds per GPU.**
- `--decision hybrid` with qwen2.5vl:3b: VLM warm latency **27-45s per call**,
  and it contends with YOLO for the same 4GB. Even dispatched to a worker
  thread (which lifted 2.5 -> 6.8 fps) it does not reach real-time on one card.
  One 4GB GPU cannot run a real-time detector and a 3B VLM concurrently.
  For a strict real-time demo use `rules`; use `hybrid` for enrichment, a
  longer feed where alerts are rare, or a second GPU.
- Annotated 4K video encoding costs more than the whole pipeline (26.6 -> 9.7
  fps), hence `--save-width 1280`.

Distillation: use an API model (Claude/GPT/Gemini vision) OFFLINE to
auto-label sample frames with the same JSON schema as Stage 3 — these
pseudo-labels become the fine-tuning dataset, uploaded to Kaggle as a
Dataset, trained there, adapter pulled back down locally.

## Model choices
- Primary small VLM: **Qwen2.5-VL-3B-Instruct** — most proven Unsloth
  fine-tune → GGUF (with mmproj) → llama.cpp path as of 2026. This is the
  baseline; get it working end-to-end before touching anything else.
- Optional stretch upgrade (only if the baseline is done with time to
  spare): Qwen3-VL-2B-Instruct — newer, GGUF quants exist, but the
  llama.cpp vision path is less battle-tested. Do not start here.
- Local inference runtime: **Ollama for Windows** (native CUDA, no build
  step). Chosen over llama-cpp-python because it needs no compiler, no
  mmproj file juggling, and it holds the model in its own process — which
  keeps VLM weights out of our 7.35GB Python RAM budget.
- Local fallback models (CPU/tight VRAM): Moondream2 (1.8B) or
  SmolVLM2-500M, quantized — NOT vLLM locally.
- Detector: YOLO26n (primary) or YOLO11n (fallback) via `ultralytics`.

## Coding conventions
- Python 3.11 in the venv at `C:\dvad\.venv` (native Windows, not WSL).
- Every script in src/ must run standalone with
  `C:\dvad\.venv\Scripts\python.exe src\<file>.py --help`
  and accept a `--data_dir` argument — dataset path must never be
  hardcoded. Saturday's real dataset must be a one-flag swap.
- Prefer short, working scripts over abstracted frameworks. This is a
  hackathon — working end-to-end beats elegant and incomplete.
- After completing any meaningful step, append one short bullet to
  PROGRESS.md (what was done, verify command, what's next). Bullets only,
  no long explanations there.
- Before installing anything GPU/CUDA-related or starting a Kaggle run,
  state the plan and wait for confirmation if it's large (>2GB) or costs
  meaningful GPU-hour quota.
- When context gets long, re-read CLAUDE.md + PROGRESS.md rather than
  asking the user to re-explain the project.

## Current status
See PROGRESS.md for the live log. Start there every session.
