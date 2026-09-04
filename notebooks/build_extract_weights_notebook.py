"""Generate extract_weights.ipynb - re-export just the weights from a finished run.

Why this exists: the first VisDrone training run left the ~2GB dataset inside
/kaggle/working, so its kernel output is multi-GB and pulling it locally means
downloading thousands of dataset images to reach one 5MB best.pt. Rather than
re-train for ~50 minutes, this attaches that finished kernel as an input source
and re-exports only the weights and metrics. Runs in well under a minute.

    python notebooks\\build_extract_weights_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(t): CELLS.append({"cell_type": "markdown", "id": f"x{len(CELLS):02d}",
                         "metadata": {}, "source": _src(t)})


def code(t): CELLS.append({"cell_type": "code", "id": f"x{len(CELLS):02d}",
                           "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _src(t)})


md(r"""
# Extract weights from the finished VisDrone run

The training kernel's output contains the whole VisDrone dataset (~2GB), which
makes it painful to download. This attaches that kernel as an input and copies
out only `best.pt` plus the metrics, so the local pull is a few MB.
""")

code(r'''
# Find whatever the attached source kernel produced, without assuming a layout.
import glob, shutil, os
from pathlib import Path

print("--- /kaggle/input tree (top levels) ---")
for p in sorted(glob.glob("/kaggle/input/*"))[:20]:
    print(" ", p)

pts = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
csvs = sorted(glob.glob("/kaggle/input/**/results.csv", recursive=True))
print(f"\nfound {len(pts)} .pt file(s):")
for p in pts:
    print(f"  {p}  ({os.path.getsize(p)/1e6:.2f} MB)")
print(f"found {len(csvs)} results.csv")
''')

code(r'''
# Copy out best.pt (preferred) or the largest .pt as a fallback, plus metrics.
out = Path("/kaggle/working")
best = [p for p in pts if Path(p).name == "best.pt"]
chosen = best[0] if best else (max(pts, key=os.path.getsize) if pts else None)
if chosen is None:
    raise SystemExit("No .pt found in the attached kernel output - check the source slug.")

shutil.copy2(chosen, out / "yolo26n_visdrone.pt")
print(f"exported {chosen} -> yolo26n_visdrone.pt "
      f"({os.path.getsize(chosen)/1e6:.2f} MB)")

for c in csvs:
    shutil.copy2(c, out / "results.csv")
    print(f"exported {c} -> results.csv")

# The last row of results.csv carries the final-epoch metrics worth quoting.
if csvs:
    import csv as _csv
    with open(csvs[0], newline="") as f:
        rows = list(_csv.DictReader(f))
    if rows:
        last = rows[-1]
        print(f"\nepochs completed: {len(rows)}")
        for k, v in last.items():
            k = k.strip()
            if any(t in k for t in ("mAP", "precision", "recall", "loss")):
                print(f"  {k:<28} {v}")

print("\noutput contents:")
for f in sorted(out.rglob("*")):
    if f.is_file():
        print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB")
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "extract_weights.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")

bad = [i for i, c in enumerate(CELLS)
       if any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
