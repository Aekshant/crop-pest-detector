#!/usr/bin/env python3
"""Export multilingual-e5-small for LiteRT (replaces the ONNX Runtime embedder).

  uvx --python 3.11 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
      --with "torch==2.4.1+cpu" --with tensorflow-cpu==2.17.0 --with tf-keras==2.17.0 --with transformers==4.44.2 \\
      python kb/export_litert.py

The model is ~96M parameters of word-embedding table plus a ~21M parameter network, so
it ships in two parts (kb/dist/):
  e5_small_table.bin       the word table, int8 with a float32 scale per row, looked up
                           directly by the app (memory-mapped; no runtime needed)
  e5_small_tokenizer.json  the reference tokenizer (pack builder, parity fixtures)
  e5_small_vocab.tsv       its vocabulary for the phone's UnigramTokenizer ("piece<TAB>score", line = id)
  e5_small_encoder.tflite  the network: word vectors in -> mean-pooled, L2-normalized
                           sentence vector out; weights dynamic-range int8; runs on the
                           LiteRT interpreter the app already ships for vision
Table format (little-endian): b"E5T1", int32 rows, int32 dim, float32 scale[rows], int8 q[rows*dim];
the word vector of token t is q[t] * scale[t].
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from transformers import TFAutoModel

DIST = Path(__file__).resolve().parent / "dist"
MODEL = "intfloat/multilingual-e5-small"


def main():
    DIST.mkdir(exist_ok=True)
    tok = hf_hub_download(MODEL, "tokenizer.json")
    shutil.copy(tok, DIST / "e5_small_tokenizer.json")
    vocab = json.loads(Path(tok).read_text(encoding="utf-8"))["model"]["vocab"]
    assert not any("\t" in p or "\n" in p for p, _ in vocab)
    (DIST / "e5_small_vocab.tsv").write_text("".join(f"{p}\t{s}\n" for p, s in vocab), encoding="utf-8")

    m = TFAutoModel.from_pretrained(MODEL, from_pt=True)
    table = m.bert.embeddings.weight.numpy().astype(np.float32)  # word_embeddings [rows, dim]
    rows, dim = table.shape
    scale = np.maximum(np.abs(table).max(axis=1), 1e-12) / 127.0
    q = np.clip(np.rint(table / scale[:, None]), -127, 127).astype(np.int8)
    with (DIST / "e5_small_table.bin").open("wb") as f:
        f.write(b"E5T1" + struct.pack("<ii", rows, dim))
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())
    err = np.abs(q.astype(np.float32) * scale[:, None] - table).max()
    print(f"table {rows} x {dim}: max abs dequantization error {err:.5f}")

    @tf.function(input_signature=[tf.TensorSpec([1, None, dim], tf.float32, name="word_vectors")])
    def encode(word_vectors):
        n = tf.shape(word_vectors)[1]
        out = m({"inputs_embeds": word_vectors, "attention_mask": tf.ones([1, n], tf.int32)}, training=False).last_hidden_state
        pooled = tf.reduce_mean(out, axis=1)                                  # mean over tokens (one text, no padding)
        return {"embedding": tf.math.l2_normalize(pooled, axis=-1)}

    conv = tf.lite.TFLiteConverter.from_concrete_functions([encode.get_concrete_function()], m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]                          # dynamic-range int8 weights
    tflite = conv.convert()
    (DIST / "e5_small_encoder.tflite").write_bytes(tflite)
    print(f"encoder: {len(tflite) / 2**20:.1f} MB, table: {(DIST / 'e5_small_table.bin').stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
