#!/usr/bin/env python3
"""Export a trained checkpoint to LiteRT (.tflite): FP32, FP16 and INT8 (V7).

  ml/.venv-export/bin/python ml/export.py --run ml/runs/v3b_lite0_cpu

Uses its own venv (litert-torch pulls TensorFlow-era deps):
  uv venv -p 3.11 ml/.venv-export
  uv pip install -p ml/.venv-export/bin/python --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu litert-torch "transformers>=4.45" timm pillow

Model I/O (what the C++ core feeds and reads):
  input  [1, 224, 224, 3] float32, RGB 0..255 (resize short side to 256, center crop 224)
  output [1, 12] float32 logits in labels_v1 order; app applies temperature + softmax

INT8: static int8 weights + activations for CONV_2D and DEPTHWISE_CONV_2D only
(per-channel weights, min/max ranges from --calib training images stratified
by class). Quantizing every op broke the model (40% top-1 agreement with FP32:
the pooling SUM / GATHER_ND / classifier do not survive it); conv-only gives
92% and is the same size, because the convs hold almost all weights. Since
quantization shifts probabilities, INT8 gets its own temperature and reject
threshold, refit on in-situ field VALIDATION photos (V8). The eval sets are
never used for calibration.
Each file is checked against PyTorch, scored on the eval sets (quality gate +
the run's thresholds) and timed on this machine's CPU.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

ML = Path(__file__).resolve().parent
REPO = ML.parent
EVAL_VISION = REPO / "evals" / "vision"
sys.path.insert(0, str(ML))
import quality_gate  # noqa: E402


class Deployable(torch.nn.Module):
    """NHWC 0..255 in, logits out: normalization lives inside the model."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.net((x - self.mean) / self.std)


def preprocess(path, size):
    """Python reference of the on-device preprocessing (C++ must match it)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        short = round(size * 256 / 224)
        scale = short / min(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
        left, top = (im.width - size) // 2, (im.height - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.float32)[None]


def representative(n, seed):
    with (ML / "data" / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    per = max(1, n // len(by_label))
    picked = []
    for label in sorted(by_label):
        picked += rng.sample(by_label[label], min(per, len(by_label[label])))
    return [ML / "data" / r["path"] for r in picked[:n]]


def interpreter(path, threads):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


def run(it, x):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(inp["index"], x.astype(inp["dtype"]))
    it.invoke()
    return it.get_tensor(out["index"]).astype(np.float32)


def eval_files():
    files = []
    for sub in ("field", "ood"):
        with (EVAL_VISION / sub / "labels.csv").open(newline="") as f:
            files += [f"{sub}/{r['file']}" for r in csv.DictReader(f)]
    return files


def inline_buffers(path):
    """Rewrite a .tflite so every weight buffer lives inside the flatbuffer.

    ai-edge-quantizer stores buffers *after* the flatbuffer, referenced by
    offset/size. Newer LiteRT runtimes read that; LiteRT 1.x (what the app
    links) does not and fails with "Input tensor N lacks data". Models here
    are far below the 2 GB limit that external buffers exist for.
    """
    import flatbuffers
    from ai_edge_litert import schema_py_generated as S
    raw = path.read_bytes()
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(raw, 0))
    moved = 0
    for b in m.buffers:
        if b.offset and b.offset > 1:
            b.data = np.frombuffer(raw[b.offset:b.offset + b.size], dtype=np.uint8)
            b.offset, b.size = 0, 0
            moved += 1
    if not moved:
        return 0
    builder = flatbuffers.Builder(len(raw) + (1 << 20))
    builder.Finish(m.Pack(builder), file_identifier=b"TFL3")
    path.write_bytes(bytes(builder.Output()))
    return moved


def softmax(z, t):
    z = z / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = args.run / "export"
    out.mkdir(exist_ok=True)

    ck = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    net = timm.create_model(ck["arch"], pretrained=False, num_classes=len(ck["labels"]))
    net.load_state_dict(ck["state_dict"])
    model = Deployable(net, ck["mean"], ck["std"]).eval()
    size, labels, temperature = ck["img"], ck["labels"], ck["temperature"]
    reject, accept = ck["thresholds"]["reject"], ck["thresholds"]["accept"]

    # ----- convert -----
    import litert_torch
    from ai_edge_quantizer import quantizer, recipe, recipe_manager
    sample = torch.rand(1, size, size, 3) * 255
    fp32 = out / "vision_fp32.tflite"
    litert_torch.convert(model, (sample,)).export(str(fp32))
    print(f"fp32: {fp32.stat().st_size / 1e6:.2f} MB", flush=True)

    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=recipe.TFLOperationName.ALL_SUPPORTED,
                              num_bits=16, algorithm_key=recipe.AlgorithmName.FLOAT_CASTING)
    fp16 = out / "vision_fp16.tflite"
    q = quantizer.Quantizer(str(fp32), rm.get_quantization_recipe())
    q.quantize().export_model(str(fp16), overwrite=True)
    print(f"fp16: moved {inline_buffers(fp16)} external buffers inline (LiteRT 1.x)", flush=True)
    print(f"fp16: {fp16.stat().st_size / 1e6:.2f} MB", flush=True)

    calib_paths = representative(args.calib, args.seed)
    rm = recipe_manager.RecipeManager()
    for op in (recipe.TFLOperationName.CONV_2D, recipe.TFLOperationName.DEPTHWISE_CONV_2D):
        rm.add_static_config(regex=".*", operation_name=op, activation_num_bits=8, weight_num_bits=8)
    q = quantizer.Quantizer(str(fp32), rm.get_quantization_recipe())
    sig = interpreter(fp32, 1).get_signature_list()
    key = next(iter(sig))
    input_name = sig[key]["inputs"][0]
    calib = q.calibrate({key: [{input_name: preprocess(p, size)} for p in calib_paths]})
    int8 = out / "vision_int8.tflite"
    q.quantize(calib).export_model(str(int8), overwrite=True)
    print(f"int8: moved {inline_buffers(int8)} external buffers inline (LiteRT 1.x)", flush=True)
    print(f"int8: {int8.stat().st_size / 1e6:.2f} MB (calibrated on {len(calib_paths)} training images)", flush=True)

    # ----- verify, score, time -----
    files = eval_files()
    xs = [preprocess(EVAL_VISION / f, size) for f in files]
    with torch.no_grad():
        ref = np.concatenate([model(torch.from_numpy(x)).numpy() for x in xs])
    gate = quality_gate.load_thresholds()
    quality = {}
    for f in files:
        with Image.open(EVAL_VISION / f) as im:
            quality[f] = quality_gate.check(im, gate)[0]

    # In-situ field validation rows, exactly as train.py splits them.
    import train as T
    with (ML / "data" / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] not in {s for s in ck["args"].get(
            "exclude_sources", "pakistan,fieldfruit,bangladesh,tomatovillage").split(",") if s}]
    _, val_rows = T.split_rows(rows, float(ck["args"]["val_frac_field"]), int(ck["args"]["val_max"]), int(ck["args"]["seed"]))
    val_rows = [r for r in val_rows if r["source"] in T.IN_SITU_SOURCES]
    val_x = [preprocess(ML / "data" / r["path"], size) for r in val_rows]
    val_y = torch.tensor([labels.index(r["label"]) for r in val_rows])
    print(f"refitting calibration per variant on {len(val_rows)} in-situ field val photos", flush=True)

    table, calibration = [], {}
    for name, path in (("fp32", fp32), ("fp16", fp16), ("int8", int8)):
        it = interpreter(path, args.threads)
        vl = torch.from_numpy(np.concatenate([run(it, x) for x in val_x]))
        temperature = T.fit_temperature(vl, val_y)
        reject = T.choose_reject_threshold(torch.softmax(vl / temperature, 1), val_y, labels.index("unknown"), 0.2)
        calibration[name] = {"temperature": round(temperature, 4), "reject_below": reject, "confident_at_or_above": accept}
        logits = np.concatenate([run(it, x) for x in xs])
        agree = float((logits.argmax(1) == ref.argmax(1)).mean())
        preds = args.run / f"eval_preds_{name}.jsonl"
        with preds.open("w") as fh:
            for f, z in zip(files, logits):
                p = softmax(z, temperature)
                fh.write(json.dumps({"file": f, "probs": {l: round(float(v), 5) for l, v in zip(labels, p)},
                                     "quality": quality[f]}) + "\n")
        rep = subprocess.run([sys.executable, str(REPO / "evals" / "score_vision.py"), str(preds),
                              "--reject", str(reject), "--accept", str(accept),
                              "--json", str(args.run / f"eval_report_{name}.json")], capture_output=True, text=True)
        (args.run / f"eval_report_{name}.txt").write_text(rep.stdout)
        r = json.loads((args.run / f"eval_report_{name}.json").read_text())
        for _ in range(5):
            run(it, xs[0])
        times = []
        for x in xs[:50]:
            t0 = time.perf_counter()
            run(it, x)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        row = {"variant": name, "size_mb": round(path.stat().st_size / 1e6, 2),
               "temperature": round(temperature, 3), "reject_below": reject,
               "top1_agreement_with_pytorch": round(agree, 3),
               "field_accuracy": round(r["field"]["accuracy"], 3),
               "confident_accuracy": round(r["field"]["confident_accuracy"], 3),
               "ece": round(r["field"]["ece"], 3), "ood_rejected": round(r["ood"]["reject_rate"], 3),
               "cpu_ms_p50": round(times[len(times) // 2], 1), "cpu_ms_p90": round(times[int(len(times) * 0.9)], 1)}
        table.append(row)
        print(json.dumps(row), flush=True)
    (out / "export_report.json").write_text(json.dumps({"threads": args.threads, "rows": table}, indent=2) + "\n")
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")


if __name__ == "__main__":
    main()
