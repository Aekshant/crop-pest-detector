#!/usr/bin/env python3
"""Write golden images + Python reference outputs for the C++ core (A4-A6).

  ml/.venv-export/bin/python ml/golden.py --run ml/runs/v3b_lite0_cpu

Picks validation photos (never eval photos) that the FP16 and INT8 models both
classify correctly with high confidence, one per class where possible, and
writes to core/tests/golden/:
  <name>.rgb        raw RGB8 pixels (max side 480 px), the input
  <name>.pre        224x224x3 uint8: Python preprocessing (export.preprocess)
  golden.json       sizes, expected quality (quality_gate.measure), expected
                    label, logits per model file
The C++ host tests compare preprocessing and quality against these; the
Android instrumented test checks the labels on device.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = REPO / "core" / "tests" / "golden"
sys.path.insert(0, str(ML))
import export as E  # noqa: E402
import quality_gate as Q  # noqa: E402
import train as T  # noqa: E402

MAX_SIDE = 480
WANT = ["healthy", "early_blight", "late_blight", "septoria_leaf_spot", "yellow_leaf_curl_virus",
        "spider_mites", "whitefly", "unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-prob", type=float, default=0.8)
    args = ap.parse_args()
    ck = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    labels = ck["labels"]
    cal = json.loads((args.run / "export" / "calibration.json").read_text())
    models = {k: E.interpreter(args.run / "export" / f"vision_{k}.tflite", 4) for k in ("fp16", "int8")}

    with (ML / "data" / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] not in {"pakistan", "fieldfruit", "bangladesh", "tomatovillage"}]
    _, val_rows = T.split_rows(rows, 0.15, 120, 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*"):
        p.unlink()

    chosen = []
    for label in WANT:
        cands = [r for r in val_rows if r["label"] == label]
        # Prefer field photos (what the app sees), then lab photos.
        cands.sort(key=lambda r: (r["source"] not in T.IN_SITU_SOURCES, r["sha256"]))
        for r in cands[:60]:
            with Image.open(ML / "data" / r["path"]) as im:
                im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            tmp = OUT / "_tmp.png"
            im.save(tmp)
            x = E.preprocess(tmp, 224)
            logits = {k: E.run(it, x)[0] for k, it in models.items()}
            probs = {k: E.softmax(z, cal[k]["temperature"]) for k, z in logits.items()}
            if all(int(p.argmax()) == labels.index(label) and p.max() >= args.min_prob for p in probs.values()):
                name = f"{len(chosen):02d}_{label}"
                (OUT / f"{name}.rgb").write_bytes(np.asarray(im, dtype=np.uint8).tobytes())
                (OUT / f"{name}.pre").write_bytes(x[0].astype(np.uint8).tobytes())
                sharp, bright = Q.measure(im)
                chosen.append({
                    "name": name, "width": im.width, "height": im.height, "label": label,
                    "label_index": labels.index(label), "source": r["source"], "manifest_path": r["path"],
                    "sharpness": round(sharp, 6), "brightness": round(bright, 4),
                    "logits": {k: [round(float(v), 5) for v in z] for k, z in logits.items()},
                    "top_prob": {k: round(float(p.max()), 4) for k, p in probs.items()},
                })
                print(f"{name}: {r['source']} p_fp16={probs['fp16'].max():.2f} p_int8={probs['int8'].max():.2f}")
                break
        else:
            print(f"{label}: no confident correct validation photo found")
        (OUT / "_tmp.png").unlink(missing_ok=True)
    gate = json.loads((ML / "quality_gate.json").read_text())
    (OUT / "golden.json").write_text(json.dumps({
        "labels": labels, "input_size": 224, "calibration": cal,
        "quality_thresholds": {k: gate[k] for k in ("min_sharpness", "min_brightness", "max_brightness")},
        "images": chosen}, indent=2) + "\n")
    # Same facts as TSV, so C++ tests need no JSON parser.
    with (OUT / "golden.tsv").open("w") as f:
        f.write("name\twidth\theight\tlabel_index\tsharpness\tbrightness\n")
        for c in chosen:
            f.write(f"{c['name']}\t{c['width']}\t{c['height']}\t{c['label_index']}\t{c['sharpness']}\t{c['brightness']}\n")
    print(f"{len(chosen)} golden images -> {OUT}")


if __name__ == "__main__":
    main()
