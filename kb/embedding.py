"""Embedding models for the KB pack. The pack records which one it used, so
build_pack.py and every reader (evals, the app) embed the same way.

A model may need prefixes: e5 models expect "query: " for questions and
"passage: " for documents.

"e5-small-int8" is the on-device model: kb/export_embedder.py writes the int8
ONNX file and tokenizer to kb/dist/, and this module runs them exactly as the
phone does (EmbeddingModel.kt): tokenize, run, mean-pool over the attention
mask, L2-normalize. The other models go through fastembed (comparison only).
"""

from pathlib import Path

DIST = Path(__file__).resolve().parent / "dist"
MAX_TOKENS = 512

MODELS = {
    "e5-small-int8": {"query": "query: ", "passage": "passage: ",
                      "onnx": "e5_small_int8.onnx", "tokenizer": "e5_small_tokenizer.json"},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: ",
                                       "custom": {"dim": 384, "model_file": "onnx/model.onnx"}},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query": "", "passage": ""},
}


class OnnxEmbedder:
    def __init__(self, spec):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        missing = [f for f in (spec["onnx"], spec["tokenizer"]) if not (DIST / f).exists()]
        if missing:
            raise SystemExit(f"missing {missing} in kb/dist: run kb/export_embedder.py")
        self.tokenizer = Tokenizer.from_file(str(DIST / spec["tokenizer"]))
        self.session = ort.InferenceSession(str(DIST / spec["onnx"]), providers=["CPUExecutionProvider"])

    def ids(self, text):
        """<s> pieces </s>, cut to MAX_TOKENS keeping the closing </s> (id 2)."""
        ids = self.tokenizer.encode(text).ids
        return ids if len(ids) <= MAX_TOKENS else ids[:MAX_TOKENS - 1] + [2]

    def embed(self, texts, batch_size=1):
        import numpy as np
        for t in texts:
            ids = np.array([self.ids(t)], dtype=np.int64)
            mask = np.ones_like(ids)
            out = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
            yield out[0].mean(axis=0)  # mean over tokens (no padding: one text per run)


def load(name):
    spec = MODELS[name]
    if "onnx" in spec:
        return OnnxEmbedder(spec)
    from fastembed import TextEmbedding
    if "custom" in spec:
        from fastembed.common.model_description import ModelSource, PoolingType
        if name not in {m["model"] for m in TextEmbedding.list_supported_models()}:
            TextEmbedding.add_custom_model(model=name, pooling=PoolingType.MEAN, normalization=True,
                                           sources=ModelSource(hf=name), **spec["custom"])
    return TextEmbedding(name)


def embed(model, name, texts, kind):
    """Unit vectors for texts of one kind ("query" or "passage")."""
    prefix = MODELS[name][kind]
    out = []
    for v in model.embed([prefix + t for t in texts], batch_size=32):
        n = float(sum(float(x) * float(x) for x in v)) ** 0.5
        out.append([float(x) / n for x in v])
    return out
