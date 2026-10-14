#!/usr/bin/env python3
"""Cross-language fixture: a manifest signed here (Python, `cryptography`) that the app's
Kotlin verifier must accept (BundleManifestTest). The key is generated in memory and
discarded: only its public half and the signed envelope are written.

  uvx --with cryptography python ml/bundle/make_test_fixture.py
"""

import base64
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bundle import sign  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "android" / "app" / "src" / "test" / "resources" / "bundle_fixture.json"

key = Ed25519PrivateKey.generate()
manifest = {"bundle_version": "2026.01.01", "min_app_version": 1, "tool_schema_version": 1, "tiers": ["A", "B", "C"],
            "artifacts": [{"name": "labels", "file": "labels_v1.json", "sha256": "0" * 64, "bytes": 10},
                          {"name": "laya", "file": "laya.gguf", "sha256": "f" * 64, "bytes": 20, "tiers": ["A", "B"]}]}
envelope = sign(json.dumps(manifest, indent=1, ensure_ascii=False).encode(), key, "python-test")
public = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"note": "test only; private key discarded", "public_key": public, "envelope": envelope}, indent=1) + "\n")
print(OUT)
