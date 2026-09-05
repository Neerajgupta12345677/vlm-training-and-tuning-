"""Generate ingest_newdata.ipynb - pull the new dataset straight into Kaggle.

The constraint driving the design: the new data CANNOT come to this laptop.
It is a Google Drive folder, the laptop has ~1GB of free RAM and venue wifi is
unreliable after 10am. So the video data is pulled directly into the Kaggle
kernel, every heavy step happens there, and the ONLY thing that comes back is a
pair of ~6MB MobileNetV3 checkpoints for local inference.

The notebook is staged so a schema surprise costs two minutes, not a GPU run:

  CELL 1-2  pull + unpack
  CELL 3    INSPECT and ASSERT the schema, printing the tree and CSV headers.
            If the new data is not shaped like the AHC set, this stops here
            with a readable message instead of failing three cells later inside
            a cache builder.
  CELL 4-5  build the frame caches with OUR OWN tested code (attached as the
            `dvad-src` dataset, so build_motion_frames.py / train_appearance.py
            are the exact files that were debugged locally - not a re-typed copy
            that can silently drift)
  CELL 6    train, and write only the checkpoints to /kaggle/working

    python notebooks\\build_ingest_notebook.py
    python src\\push_notebook.py --push --slug dvad-ingest-newdata ^
        --title "dvad ingest newdata" --notebook notebooks\\ingest_newdata.ipynb ^
        --dataset dvad-src
"""

import json
from pathlib import Path

CELLS = []


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(t): CELLS.append({"cell_type": "markdown", "id": f"i{len(CELLS):02d}",
                         "metadata": {}, "source": _src(t)})


def code(t): CELLS.append({"cell_type": "code", "id": f"i{len(CELLS):02d}",
                           "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _src(t)})


md(r"""
# Ingest the new dataset directly into Kaggle

The data never touches the laptop. Videos and frame caches stay in
`/kaggle/temp`; only the trained checkpoints land in `/kaggle/working`.
""")

code(r'''
import os, sys, subprocess, shutil, json, csv
from pathlib import Path

DRIVE_FOLDER = "1CudRVY0UNIUPeV7p5vPKLfG4GXpRU8Hb"
RAW  = Path("/kaggle/temp/newdata")      # downloaded videos
WORK = Path("/kaggle/working")           # kernel output - keep SMALL
SRC  = None

# RECURSIVE search, deliberately. Kaggle does not always mount an input at
# /kaggle/input/<slug>: this account's kernels get /kaggle/input/datasets/
# <owner>/<slug>, so a direct `(root / "build_motion_frames.py").exists()`
# check finds nothing and the run dies claiming the dataset is not attached
# when it plainly is. Measured from the merge kernel's own log.
hits = list(Path("/kaggle/input").rglob("build_motion_frames.py"))
if hits:
    SRC = hits[0].parent
if SRC is None:
    print("contents of /kaggle/input:")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:40]:
        print("  ", p)
assert SRC is not None, "attach the `dvad-src` dataset - our pipeline code lives there"
sys.path.insert(0, str(SRC))
print("pipeline code:", SRC)

RAW.mkdir(parents=True, exist_ok=True)
''')

code(r'''
!pip -q install --upgrade gdown 2>&1 | tail -1
# No --remaining-ok: it is not accepted by the gdown build on this image, and
# passing it makes gdown exit on its usage message having downloaded NOTHING.
# Because `!` shell magics do not raise, that failure previously surfaced three
# cells later as a bogus "schema mismatch" on an empty directory.
!gdown --folder "https://drive.google.com/drive/folders/{DRIVE_FOLDER}" -O {RAW}
!du -sh {RAW}
''')

code(r'''
# Check the DOWNLOAD at the download step, so a download problem is reported as
# a download problem. gdown returns success for a folder it could not read, and
# a private folder yields an empty directory rather than an error.
got = [p for p in RAW.rglob("*") if p.is_file()]
print(f"files downloaded: {len(got)}")
for p in got[:20]:
    print("  ", p.relative_to(RAW), f"{p.stat().st_size/1e6:.1f} MB")
assert got, (
    "gdown downloaded NOTHING from the Drive folder.\n"
    "Usual cause: the folder is not shared as 'Anyone with the link'.\n"
    "Open it in an incognito window - if it asks you to sign in, Kaggle cannot\n"
    "read it either, and the sharing setting has to change."
)
''')

code(r'''
# Anything zipped gets expanded in place, then the archive is removed so it does
# not double the disk footprint.
for z in list(RAW.rglob("*.zip")) + list(RAW.rglob("*.tar.gz")):
    print("unpacking", z.name)
    shutil.unpack_archive(str(z), str(z.parent))
    z.unlink()

vids = sorted(RAW.rglob("*.mp4")) + sorted(RAW.rglob("*.avi")) + sorted(RAW.rglob("*.mov"))
csvs = sorted(RAW.rglob("*.csv"))
print(f"\nvideos: {len(vids)}   csvs: {len(csvs)}")
print("\ntree (first 40 entries):")
for i, p in enumerate(sorted(RAW.rglob("*"))):
    if i >= 40: print("  ..."); break
    print("  ", p.relative_to(RAW), f"({p.stat().st_size/1e6:.1f} MB)" if p.is_file() else "/")

for c in csvs:
    print(f"\n--- {c.relative_to(RAW)} ---")
    with c.open(encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        for j, row in enumerate(rd):
            print("  ", row)
            if j >= 3: break
''')

code(r'''
# Fail here, loudly, rather than inside a cache builder three cells down.
AHC_COLS = {"video_id", "class_name", "start_time_sec", "end_time_sec"}
gt = None
for c in csvs:
    with c.open(encoding="utf-8-sig") as f:
        cols = set(h.strip() for h in next(csv.reader(f), []))
    if AHC_COLS <= cols:
        gt = c
        break

if gt is None:
    print("SCHEMA MISMATCH - no CSV carries", sorted(AHC_COLS))
    print("Columns seen:")
    for c in csvs:
        with c.open(encoding="utf-8-sig") as f:
            print("  ", c.name, next(csv.reader(f), []))
    raise SystemExit(
        "The new data is not in AHC schema. Stopping BEFORE the GPU work.\n"
        "Send the column mapping and I will add an adapter cell - this costs\n"
        "two minutes here instead of a wasted training run."
    )

print("ground truth:", gt)
rows = list(csv.DictReader(gt.open(encoding="utf-8-sig")))
from collections import Counter
print("rows:", len(rows))
print("classes:", Counter(r["class_name"].strip() for r in rows).most_common())
timed = sum(1 for r in rows if (r.get("start_time_sec") or "").strip())
print(f"rows with intervals: {timed}/{len(rows)}")
''')

code(r'''
# Reshape into the train/<class>/videos/ + ground_truth.csv layout our builders
# expect, so build_motion_frames.py and train_appearance.py run UNCHANGED.
DATA = Path("/kaggle/temp/ahc_new")
by_vid = {v.stem: v for v in vids}
per_class = {}
for r in rows:
    per_class.setdefault(r["class_name"].strip() or "normal", []).append(r)

made = 0
for cls, crows in per_class.items():
    vdir = DATA / "train" / cls / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    keep = []
    for r in crows:
        src = by_vid.get(r["video_id"].strip())
        if src is None:
            continue
        dst = vdir / f"{r['video_id'].strip()}.mp4"
        if not dst.exists():
            os.symlink(src, dst)      # symlink, not copy: disk is finite
        keep.append(r); made += 1
    with (DATA / "train" / cls / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
print(f"staged {made} videos across {len(per_class)} classes -> {DATA}")
''')

code(r'''
CACHE_M = Path("/kaggle/temp/motion_frames")
CACHE_A = Path("/kaggle/temp/interval_frames")

# Kaggle gives 4 CPUs; shard the (CPU-bound) cache build across them.
import multiprocessing as mp
N = min(4, mp.cpu_count())
procs = [subprocess.Popen([sys.executable, str(SRC / "build_motion_frames.py"),
                           "--data_dir", str(DATA), "--out", str(CACHE_M),
                           "--num-shards", str(N), "--shard", str(i), "--skip-existing"])
         for i in range(N)]
for p in procs: p.wait()
!ls {CACHE_M} && find {CACHE_M} -name '*.jpg' | wc -l
''')

code(r'''
!python {SRC}/build_interval_frames.py --data_dir {DATA} --out {CACHE_A}
!find {CACHE_A} -name '*.jpg' | wc -l
''')

code(r'''
# Only the checkpoints go to /kaggle/working - a few MB each, so the local pull
# is trivial even on bad wifi. The caches and videos stay behind in /kaggle/temp.
!python {SRC}/train_motion.py --cache {CACHE_M} --out {WORK}/motion_classifier_new.pt --epochs 10 --save-every-epoch
!python {SRC}/train_appearance.py --data_dir {DATA} --cache {CACHE_A} --out {WORK}/appearance_new.pt --epochs 8 --save-every-epoch

print("\nOUTPUT (this is what comes down locally):")
tot = 0
for f in sorted(WORK.rglob("*")):
    if f.is_file():
        tot += f.stat().st_size
        print(f"  {f.name:<44} {f.stat().st_size/1e6:7.1f} MB")
print(f"  {'TOTAL':<44} {tot/1e6:7.1f} MB")
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "ingest_newdata.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")

bad = [i for i, c in enumerate(CELLS)
       if any(not ln.endswith("\n") for ln in c["source"][:-1])]
if bad:
    raise SystemExit(f"BROKEN newlines in cells {bad}")
print("newline check: OK")
