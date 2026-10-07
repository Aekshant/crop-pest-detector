#!/usr/bin/env python3
"""Quality gate (design doc, vision pipeline step 2): no ML, runs before the model.

  python ml/quality_gate.py --calibrate     # fit thresholds on training field photos
  python ml/quality_gate.py photo.jpg ...   # check photos

Blur: no-reference "blur effect" (Crete-Roffet et al., 2007) on the grayscale
image resized to a 512 px short side. The photo is compared with a re-blurred
copy of itself: re-blurring changes a sharp photo a lot and a blurry one
barely. sharpness = 1 - blur effect (0 = blurry, 1 = sharp). Unlike Laplacian
variance, it hardly depends on how much texture the scene has (on training
field photos: 80% of blurred copies caught vs 42%, at 5% false retakes).
Exposure: mean luminance. A photo that fails either asks for a retake.

This is the reference implementation; the on-device version in core/ must
match it (same resize, kernel and thresholds from quality_gate.json).
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

ML = Path(__file__).resolve().parent
THRESHOLDS = ML / "quality_gate.json"
SHORT_SIDE = 512
REBLUR = 4  # box blur radius used for the comparison copy


def _abs_diff_sum(im, dx, dy):
    d = ImageChops.difference(im, ImageChops.offset(im, dx, dy))
    return d, sum(i * c for i, c in enumerate(d.histogram()))


def measure(im):
    g = im.convert("L")
    scale = SHORT_SIDE / min(g.size)
    g = g.resize((max(1, round(g.width * scale)), max(1, round(g.height * scale))), Image.BILINEAR)
    reblurred = g.filter(ImageFilter.BoxBlur(REBLUR))
    blur = 0.0
    for dx, dy in ((1, 0), (0, 1)):  # horizontal and vertical neighbour differences
        d_orig, s_orig = _abs_diff_sum(g, dx, dy)
        d_blur, _ = _abs_diff_sum(reblurred, dx, dy)
        lost = ImageChops.subtract(d_orig, d_blur)  # max(0, d_orig - d_blur)
        s_lost = sum(i * c for i, c in enumerate(lost.histogram()))
        blur += (s_orig - s_lost) / max(s_orig, 1)
    sharpness = 1 - blur / 2
    brightness = ImageStat.Stat(g).mean[0]
    return sharpness, brightness


def check(im, t):
    sharpness, brightness = measure(im)
    if sharpness < t["min_sharpness"]:
        return "retake", "blurry"
    if brightness < t["min_brightness"]:
        return "retake", "too dark"
    if brightness > t["max_brightness"]:
        return "retake", "too bright"
    return "ok", None


def load_thresholds():
    return json.loads(THRESHOLDS.read_text())


def motion_blur(im, length):
    acc = im
    for i in range(1, length):
        acc = Image.blend(acc, ImageChops.offset(im, i - length // 2, 0), 1.0 / (i + 1))
    return acc


def calibrate(n, max_false_retake, seed):
    """Thresholds from TRAINING field photos only (never the eval sets)."""
    field = {"plantdoc", "inaturalist", "pakistan", "fieldfruit", "bangladesh", "tomatovillage"}
    with (ML / "data" / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] in field and r["label"] != "unknown"]
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    sharp, blurred, bright = [], [], []
    for r in rows:
        with Image.open(ML / "data" / r["path"]) as im:
            im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        s, b = measure(im)
        sharp.append(s)
        bright.append(b)
        # Blur at a strength comparable to a missed focus / shaky hand on a phone.
        k = rng.choice(["g", "m"])
        radius = max(im.size) / 1024
        bl = im.filter(ImageFilter.GaussianBlur(rng.uniform(4, 8) * radius)) if k == "g" \
            else motion_blur(im, int(rng.uniform(20, 35) * radius))
        blurred.append(measure(bl)[0])
    sharp_sorted = sorted(sharp)
    min_sharpness = sharp_sorted[int(max_false_retake * len(sharp_sorted))]
    b_sorted = sorted(bright)
    t = {
        "min_sharpness": round(min_sharpness, 4),
        "min_brightness": round(max(15.0, b_sorted[int(0.005 * len(b_sorted))] - 5), 1),
        "max_brightness": round(min(245.0, b_sorted[int(0.995 * len(b_sorted)) - 1] + 5), 1),
        "calibration": {
            "photos": len(rows),
            "sharp_photos_sent_to_retake": round(sum(s < min_sharpness for s in sharp) / len(sharp), 3),
            "blurred_copies_caught": round(sum(s < min_sharpness for s in blurred) / len(blurred), 3),
            "source": "training field photos from ml/data/manifest.csv; eval sets not used",
        },
    }
    THRESHOLDS.write_text(json.dumps(t, indent=2) + "\n")
    print(json.dumps(t, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="*", type=Path)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max-false-retake", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.n, args.max_false_retake, args.seed)
        return 0
    t = load_thresholds()
    for p in args.photos:
        with Image.open(p) as im:
            result, why = check(im, t)
        s, b = measure(Image.open(p))
        print(f"{p}: {result}{' (' + why + ')' if why else ''}  sharpness {s:.3f} brightness {b:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
