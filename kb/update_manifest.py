#!/usr/bin/env python3
"""Record the KB files in the model bundle manifest (ml/bundle/manifest_draft.json, key "kb").

  python kb/update_manifest.py      # after build_pack.py / export_embedder.py

The bundle downloads these next to the vision models; the app (KbBundle.kt) checks
every file's size and sha256 against this entry before opening it.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

KB = Path(__file__).resolve().parent
DIST = KB / "dist"
MANIFEST = KB.parent / "ml" / "bundle" / "manifest_draft.json"
FILES = {"pack": "kb_tomato_v1.sqlite", "embedding_model": "e5_small_int8.onnx", "vocab": "e5_small_vocab.tsv"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    meta = dict(sqlite3.connect(DIST / FILES["pack"]).execute("SELECT key, value FROM meta"))
    review = dict(sqlite3.connect(DIST / FILES["pack"]).execute(
        "SELECT review_status, count(*) FROM kb_treatment GROUP BY review_status"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["kb"] = {
        "kb_version": meta["kb_version"],
        "embedding_model": meta["embedding_model"],
        "embedding_dim": int(meta["embedding_dim"]),
        "query_prefix": "query: ",
        "onnxruntime": "1.31.0",
        "treatment_review": review,
        "files": {k: {"file": f, "bytes": (DIST / f).stat().st_size, "sha256": sha256(DIST / f)} for k, f in FILES.items()},
        "note": "Built by kb/build_pack.py and kb/export_embedder.py; the app opens them only after KbBundle.verify().",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in manifest["kb"]["files"].items():
        print(f"{k:16s} {v['bytes'] / 2**20:7.1f} MB  {v['sha256'][:16]}…  {v['file']}")


if __name__ == "__main__":
    main()
