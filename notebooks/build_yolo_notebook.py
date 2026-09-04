"""Generate finetune_yolo_visdrone.ipynb - adapt the DETECTOR to aerial views.

Why this and not more VLM work: Stage 2/3 are arithmetic and already generalise,
but Stage 1 is COCO-trained and COCO is ground-level photography. Measured on
VisDrone val, stock YOLO26n recall was 0.152 overall (0.008 on small objects).
Resolution + confidence tuning lifted that to 0.462, but the remaining gap is a
domain gap that only training closes.

VisDrone also has classes COCO lacks entirely - tricycle / awning-tricycle -
which are the closest public proxy to Indian auto-rickshaws.

    python notebooks\\build_yolo_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(t): CELLS.append({"cell_type": "markdown", "id": f"y{len(CELLS):02d}",
                         "metadata": {}, "source": _src(t)})


def code(t): CELLS.append({"cell_type": "code", "id": f"y{len(CELLS):02d}",
                           "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _src(t)})


md(r"""
# Aerial detector fine-tune - YOLO26n on VisDrone

Stage 1 is the weak link on drone footage. Measured on VisDrone val with the
stock COCO detector:

| config | overall recall |
|---|---|
| imgsz 640, conf 0.25 | 0.152 |
| imgsz 1536, conf 0.10 | 0.462 |

Config tuning tripled it for free; this notebook closes the remaining domain gap
by training on actual aerial imagery. It also adds `tricycle` /
`awning-tricycle`, which COCO does not have at all.
""")

code(r'''
import os, subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
!pip install -q --upgrade ultralytics
import ultralytics, torch
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.cuda.is_available(), "No GPU - set the accelerator."

# Keep the ~2GB dataset OUT of /kaggle/working. Everything under /kaggle/working
# becomes kernel output, so the first run made `kaggle kernels output` pull the
# entire VisDrone dataset back down just to reach a 5MB weights file. /kaggle/temp
# is scratch and is not part of the output.
from ultralytics import settings
os.makedirs("/kaggle/temp/datasets", exist_ok=True)
settings.update({"datasets_dir": "/kaggle/temp/datasets", "sync": False})
print("datasets_dir:", settings["datasets_dir"], "| telemetry sync:", settings["sync"])
''')

code(r'''
CONFIG = dict(
    model   = "yolo26n.pt",   # same nano model the laptop runs - keep it small
    data    = "VisDrone.yaml",# ultralytics auto-downloads (~2GB)
    epochs  = 30,
    imgsz   = 1024,           # aerial objects are tiny; 640 loses them
    batch   = 16,
    workers = 4,
    patience = 10,
)
for k, v in CONFIG.items():
    print(f"{k:<9} {v}")
''')

code(r'''
from ultralytics import YOLO

model = YOLO(CONFIG["model"])
results = model.train(
    data       = CONFIG["data"],
    epochs     = CONFIG["epochs"],
    imgsz      = CONFIG["imgsz"],
    batch      = CONFIG["batch"],
    workers    = CONFIG["workers"],
    patience   = CONFIG["patience"],
    project    = "/kaggle/working/runs",
    name       = "visdrone",
    exist_ok   = True,
    plots      = True,
    val        = True,
)
print("training done")
''')

code(r'''
# Validate and print the numbers worth quoting.
metrics = model.val(data=CONFIG["data"], imgsz=CONFIG["imgsz"], split="val")
print("\n=== VisDrone val ===")
print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")
print(f"precision: {metrics.box.mp:.4f}")
print(f"recall   : {metrics.box.mr:.4f}")
print("\nper class:")
for i, c in enumerate(metrics.box.ap_class_index):
    print(f"  {model.names[int(c)]:<18} AP50 {metrics.box.ap50[i]:.4f}")
''')

code(r'''
# Export the weights. best.pt is what the laptop pipeline will load.
import shutil
from pathlib import Path

run = Path("/kaggle/working/runs/visdrone")
best = run / "weights" / "best.pt"
print("best.pt exists:", best.exists(), "-", best.stat().st_size / 1e6 if best.exists() else 0, "MB")
shutil.copy2(best, "/kaggle/working/yolo26n_visdrone.pt")

# Quick speed check at the resolution we will actually deploy at.
import time, torch
m = YOLO(str(best))
img = torch.zeros(1, 3, CONFIG["imgsz"], CONFIG["imgsz"])
for _ in range(3):
    m.predict(img, verbose=False, device=0)
t0 = time.time()
for _ in range(20):
    m.predict(img, verbose=False, device=0)
print(f"\n{1/((time.time()-t0)/20):.1f} fps at imgsz={CONFIG['imgsz']} on this GPU")

# Trim the output so pulling it back is fast: keep the weights, the results csv
# and the plots; drop optimiser-laden epoch checkpoints and any stray dataset.
import shutil
from pathlib import Path

keep = {"best.pt", "last.pt", "results.csv", "args.yaml"}
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_dir():
        continue
    if p.name in keep or p.suffix in {".png", ".jpg"} or p.name == "yolo26n_visdrone.pt":
        continue
    if p.suffix == ".pt" or "datasets" in p.parts:
        try:
            p.unlink()
        except Exception:
            pass
for d in Path("/kaggle/working").rglob("datasets"):
    shutil.rmtree(d, ignore_errors=True)

total = sum(f.stat().st_size for f in Path("/kaggle/working").rglob("*") if f.is_file())
print(f"\nkernel output trimmed to {total/1e6:.1f} MB")
print("Pull it with: python src\\push_notebook.py --pull --slug dvad-yolo-visdrone")
''')

md(r"""
## Using it locally

```
python src\pipeline.py --source <clip> --weights C:\dvad\models\yolo26n_visdrone.pt --imgsz 1024 --conf 0.15
```

Compare against stock before trusting it:
```
python src\eval_aerial.py --limit 100 --imgsz 1024 --conf 0.15
python src\eval_aerial.py --limit 100 --imgsz 1024 --conf 0.15 --weights C:\dvad\models\yolo26n_visdrone.pt
```

**Note the class ids change.** VisDrone is 0=pedestrian 1=people 2=bicycle 3=car
4=van 5=truck 6=tricycle 7=awning-tricycle 8=bus 9=motor, which is NOT COCO.
`src/detect_track.py` and `src/context_state.py` key off class *names*, so update
`VEHICLE_CLASSES` / `VEHICLE_NAMES` to include `van` and the tricycles before
using these weights.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "finetune_yolo_visdrone.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")

bad = [i for i, c in enumerate(CELLS)
       if any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
