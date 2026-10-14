#!/usr/bin/env python3
"""Export the on-device embedding model: multilingual-e5-small, int8.

  uvx --with onnx --with onnxruntime --with huggingface_hub python kb/export_embedder.py

Writes kb/dist/e5_small_int8.onnx, kb/dist/e5_small_tokenizer.json and
kb/dist/e5_small_vocab.tsv (the vocabulary the phone's tokenizer reads). The pack
(build_pack.py) and the phone (EmbeddingModel.kt) both use these exact files, so
questions and passages are embedded with the same weights.
"""

import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

DIST = Path(__file__).resolve().parent / "dist"
REPO_ID = "intfloat/multilingual-e5-small"
REVISION = "main"


def main():
    DIST.mkdir(exist_ok=True)
    fp32 = hf_hub_download(REPO_ID, "onnx/model.onnx", revision=REVISION)
    tok = hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION)
    out = DIST / "e5_small_int8.onnx"
    # Dynamic int8 (weights int8, activations quantized at run time): portable to ARM, ~4x smaller.
    quantize_dynamic(fp32, out, weight_type=QuantType.QInt8)
    shutil.copy(tok, DIST / "e5_small_tokenizer.json")
    # The phone reads a compact vocabulary instead of the 17 MB tokenizer.json: line i = piece i, "piece<TAB>score".
    vocab = json.loads(Path(tok).read_text(encoding="utf-8"))["model"]["vocab"]
    assert not any("\t" in p or "\n" in p for p, _ in vocab)
    (DIST / "e5_small_vocab.tsv").write_text("".join(f"{p}\t{s}\n" for p, s in vocab), encoding="utf-8")
    print(f"{Path(fp32).stat().st_size / 2**20:.0f} MB fp32 -> {out.stat().st_size / 2**20:.0f} MB int8: {out}")


if __name__ == "__main__":
    main()
