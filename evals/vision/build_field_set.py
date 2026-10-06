#!/usr/bin/env python3
"""Build evals/vision/field/ from a PlantDoc checkout.

Field-condition test set: real-world tomato leaf photos (cluttered backgrounds,
mixed light), unlike lab sets such as PlantVillage. Selection is deterministic,
so re-running against the same PlantDoc commit gives the same set.

  git clone https://github.com/pratikkayal/PlantDoc-Dataset.git /tmp/pd
  git -C /tmp/pd checkout 5467f6012d78d1c446145d5f582da6096f852ae8
  python evals/vision/build_field_set.py --plantdoc /tmp/pd

Non-PlantDoc rows in labels.csv (own photos, fetch_web.py) are kept across re-runs.
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"
LICENSE = "CC-BY-4.0 (PlantDoc, Singh et al. 2020)"

# PlantDoc folder -> provisional FarmLens label. Final names are fixed in
# labels_v1.json (task V2); update this map and re-run if they change.
CLASS_MAP = {
    "Tomato leaf": "healthy",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato mold leaf": "leaf_mold",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
    # "Tomato two spotted spider mites leaf" is left out: its 2 images are
    # multi-disease collages. Pest classes need own photos.
}

# All of PlantDoc's test split is used; train images top each class up to this.
PER_CLASS_TARGET = 12

FIELDS = ["file", "label", "source", "source_class", "source_split",
          "source_path", "sha256", "license", "verified"]

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "field"
LABELS_CSV = OUT_DIR / "labels.csv"
# Train-split images used here; ml/ training must skip these (see README).
EXCLUDE_TXT = ROOT / "field_exclude_from_training.txt"
# Images reviewed and rejected (slides, collages, wrong crop, wrong label).
REJECTS_CSV = ROOT / "field_rejects.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_image(path: Path) -> bool:
    head = path.read_bytes()[:8]
    return head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG")


def candidates(plantdoc: Path, split: str, folder: str):
    d = plantdoc / split / folder
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and is_image(p)]
    # Order by content hash: stable and not biased by file naming.
    return sorted(((sha256(p), p) for p in files), key=lambda t: t[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantdoc", type=Path, required=True)
    args = ap.parse_args()

    own_rows = []
    if LABELS_CSV.exists():
        with LABELS_CSV.open() as f:
            own_rows = [r for r in csv.DictReader(f) if r["source"] != "plantdoc"]

    for p in OUT_DIR.glob("plantdoc_*"):
        p.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Hashes across the whole PlantDoc tomato tree, so a train image that is a
    # copy of a test image is never picked twice.
    seen = {r["sha256"] for r in own_rows}
    with REJECTS_CSV.open() as f:
        seen |= {r["sha256"] for r in csv.DictReader(f)}
    rows, excluded = [], []

    for folder, label in CLASS_MAP.items():
        picked = 0
        for split in ("test", "train"):
            for digest, src in candidates(args.plantdoc, split, folder):
                if split == "train" and picked >= PER_CLASS_TARGET:
                    break
                if digest in seen:
                    continue
                seen.add(digest)
                name = f"plantdoc_{label}_{digest[:12]}{src.suffix.lower()}"
                shutil.copyfile(src, OUT_DIR / name)
                rel = src.relative_to(args.plantdoc).as_posix()
                rows.append({
                    "file": name, "label": label, "source": "plantdoc",
                    "source_class": folder, "source_split": split,
                    "source_path": rel, "sha256": digest,
                    "license": LICENSE, "verified": "0",
                })
                if split == "train":
                    excluded.append(f"{digest}  {rel}")
                picked += 1

    rows += own_rows
    rows.sort(key=lambda r: (r["label"], r["file"]))
    with LABELS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    EXCLUDE_TXT.write_text(
        f"# sha256  path in PlantDoc@{PLANTDOC_COMMIT}\n"
        + "\n".join(sorted(excluded)) + "\n")

    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    for label, n in sorted(counts.items()):
        print(f"{label:24s} {n}")
    print(f"{'total':24s} {len(rows)}  (non-PlantDoc: {len(own_rows)})")


if __name__ == "__main__":
    main()
