#!/usr/bin/env python3
"""Build evals/vision/ood/ from ood_sources.csv.

Out-of-domain images the app must not diagnose: other crops, soil, hands
(expected "unknown") and blurry tomato shots (expected "retake" from the
quality gate). Blurry images are made from sharp source photos with a fixed
transform, so the build is reproducible.

  pip install pillow
  python evals/vision/build_ood_set.py
"""

import csv
import hashlib
import io
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ood"
SOURCES_CSV = ROOT / "ood_sources.csv"
LABELS_CSV = OUT_DIR / "labels.csv"

# Long side of blurred images, so blur strength is independent of source size.
BLUR_SIZE = 1024

FIELDS = ["file", "category", "subject", "expected", "source_path",
          "transform", "sha256", "license"]


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "farmlens-evals/0.1"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=90).read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def motion_blur(im: Image.Image, length: int) -> Image.Image:
    # Average of horizontally shifted copies: a camera shake along one axis.
    acc = im
    for i in range(1, length):
        shifted = ImageChops.offset(im, i - length // 2, 0)
        acc = Image.blend(acc, shifted, 1.0 / (i + 1))
    return acc


def apply(data: bytes, transform: str) -> bytes:
    if not transform:
        return data
    kind, amount = transform.split(":")
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((BLUR_SIZE, BLUR_SIZE))
    if kind == "gaussian":
        im = im.filter(ImageFilter.GaussianBlur(int(amount)))
    elif kind == "motion":
        im = motion_blur(im, int(amount))
    else:
        raise SystemExit(f"unknown transform: {transform}")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for p in OUT_DIR.glob("*.jpg"):
        p.unlink()

    with SOURCES_CSV.open() as f:
        sources = list(csv.DictReader(f))

    rows, counts = [], {}
    for s in sources:
        data = apply(download(s["url"]), s["transform"])
        if not data.startswith(b"\xff\xd8\xff"):
            raise SystemExit(f"not a JPEG: {s['url']}")
        n = counts[s["category"]] = counts.get(s["category"], 0) + 1
        name = f"{s['category']}_{n:02d}.jpg"
        (OUT_DIR / name).write_bytes(data)
        rows.append({
            "file": name, "category": s["category"], "subject": s["subject"],
            "expected": s["expected"], "source_path": s["page"],
            "transform": s["transform"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "license": f"{s['license']} ({s['attribution']})",
        })
        time.sleep(0.5)

    with LABELS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for cat, n in sorted(counts.items()):
        print(f"{cat:12s} {n}")
    print(f"{'total':12s} {len(rows)}")


if __name__ == "__main__":
    main()
