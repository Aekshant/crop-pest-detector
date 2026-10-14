"""Embedding models for the KB pack. The pack records which one it used, so
build_pack.py and every reader (evals, the app) embed the same way.

A model may need prefixes: e5 models expect "query: " for questions and
"passage: " for documents.

"e5-small-litert" is the on-device model: kb/export_litert.py writes the LiteRT
encoder, the int8 word table and the tokenizer to kb/dist/, and LiteRtEmbedder runs
them exactly as the phone does (LiteRtEmbeddingModel.kt). The others go through
fastembed (comparison only). An ONNX Runtime int8 export was used before 2026-10-15:
cosine to fp32 >= 0.959, vs >= 0.999 for LiteRT (evals/kb/results_e5-small-int8.json).
"""

from pathlib import Path

DIST = Path(__file__).resolve().parent / "dist"
MAX_TOKENS = 512

MODELS = {
    "e5-small-litert": {"query": "query: ", "passage": "passage: ", "litert": "e5_small_encoder.tflite",
                        "table": "e5_small_table.bin", "tokenizer": "e5_small_tokenizer.json"},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: ",
                                       "custom": {"dim": 384, "model_file": "onnx/model.onnx"}},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query": "", "passage": ""},
}


class LiteRtEmbedder:
    """The phone's pipeline (LiteRtEmbeddingModel.kt): tokenize, look word vectors up in the
    int8 table (q * row scale), run the LiteRT encoder (mean-pooled, normalized inside)."""

    def __init__(self, spec):
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter
        from tokenizers import Tokenizer
        missing = [f for f in (spec["litert"], spec["table"], spec["tokenizer"]) if not (DIST / f).exists()]
        if missing:
            raise SystemExit(f"missing {missing} in kb/dist: run kb/export_litert.py")
        self.tokenizer = Tokenizer.from_file(str(DIST / spec["tokenizer"]))
        raw = (DIST / spec["table"]).read_bytes()
        assert raw[:4] == b"E5T1"
        rows, dim = np.frombuffer(raw[4:12], "<i4")
        self.scale = np.frombuffer(raw[12:12 + 4 * rows], "<f4")
        self.q = np.frombuffer(raw[12 + 4 * rows:], np.int8).reshape(rows, dim)
        self.interp = Interpreter(model_path=str(DIST / spec["litert"]))
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def ids(self, text):
        ids = self.tokenizer.encode(text).ids
        return ids if len(ids) <= MAX_TOKENS else ids[:MAX_TOKENS - 1] + [2]

    def embed(self, texts, batch_size=1):
        import numpy as np
        for t in texts:
            ids = np.array(self.ids(t))
            x = (self.q[ids].astype(np.float32) * self.scale[ids, None])[None]
            self.interp.resize_tensor_input(self.inp, x.shape)
            self.interp.allocate_tensors()
            self.interp.set_tensor(self.inp, x)
            self.interp.invoke()
            yield self.interp.get_tensor(self.out)[0]


def load(name):
    spec = MODELS[name]
    if "litert" in spec:
        return LiteRtEmbedder(spec)
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
