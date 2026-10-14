#!/usr/bin/env python3
"""Build and sign a model bundle (design doc, LLD "Model bundle lifecycle").

  uvx --with cryptography python ml/bundle/build_bundle.py 2026.10.15 --key-id dev-2026-10 [--with-slm]

Writes dist/bundles/<version>/<artifact files> and dist/bundles/manifest.json:

  {"key_id": "...", "manifest": "<base64 of the manifest JSON bytes>", "signature": "ed25519:<base64>"}

The signature covers the exact manifest bytes (not re-serialized JSON), so Python and
Kotlin can't disagree about canonical form. Upload dist/bundles/ as is to the bundle host;
the app (BundleUpdater.kt) fetches <base>/manifest.json and <base>/<version>/<file>.
"""

import argparse
import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
REPO = ML.parent
OUT = REPO / "dist" / "bundles"
sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_APP_VERSION = 1          # versionCode of the first app that understands this bundle format
TOOL_SCHEMA_VERSION = 1
ARTIFACTS = [  # name, source path, tiers (None = all); vision files keep the app's asset names
    ("vision", "ml/runs/v3b_lite0_cpu/export/vision_int8.tflite", None),
    ("vision_gpu", "ml/runs/v3b_lite0_cpu/export/vision_fp16.tflite", ["A", "B"]),
    ("vision_config", "ml/bundle/manifest_draft.json", None, "manifest.json"),
    ("labels", "ml/labels_v1.json", None),
    ("tool_schema", "ml/bundle/tools_schema_v1.json", None),
    ("templates", "ml/bundle/laya_templates_v1.json", None),
    ("fewshot", "ml/bundle/laya_fewshot_v1.json", None),
    ("safety", "ml/bundle/laya_safety_v1.json", None),
    ("kb", "kb/dist/kb_tomato_v1.sqlite", None),
    ("kb_embedder", "kb/dist/e5_small_encoder.tflite", None),
    ("kb_table", "kb/dist/e5_small_table.bin", None),
    ("kb_vocab", "kb/dist/e5_small_vocab.tsv", None),
]
SLM = ("laya", "ml/slm/_models/Qwen3-1.7B-Q4_K_M.gguf", ["A", "B"])


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sign(manifest_bytes, private_key, key_id):
    return {"key_id": key_id, "manifest": base64.b64encode(manifest_bytes).decode(),
            "signature": "ed25519:" + base64.b64encode(private_key.sign(manifest_bytes)).decode()}


def build(version, private_key, key_id, with_slm=False, out=OUT, artifacts=None):
    artifacts = list(artifacts or ARTIFACTS) + ([SLM] if with_slm else [])
    target = out / version
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, src, tiers, *rename in artifacts:
        src = REPO / src if not Path(src).is_absolute() else Path(src)
        if not src.exists():
            raise SystemExit(f"missing artifact {name}: {src} (build it first)")
        dest = target / (rename[0] if rename else src.name)
        shutil.copyfile(src, dest)
        entry = {"name": name, "file": dest.name, "sha256": sha256(dest), "bytes": dest.stat().st_size}
        if tiers:
            entry["tiers"] = tiers
        entries.append(entry)
    manifest = {"bundle_version": version, "min_app_version": MIN_APP_VERSION, "tool_schema_version": TOOL_SCHEMA_VERSION,
                "tiers": ["A", "B", "C"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "artifacts": entries}
    manifest_bytes = json.dumps(manifest, indent=1).encode()
    (out / "manifest.json").write_text(json.dumps(sign(manifest_bytes, private_key, key_id), indent=1) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--key-id", required=True)
    ap.add_argument("--with-slm", action="store_true", help="include the edge SLM (tiers A/B, ~1.1 GB)")
    args = ap.parse_args()
    from bundle_key import load_private
    m = build(args.version, load_private(args.key_id), args.key_id, args.with_slm)
    total = sum(a["bytes"] for a in m["artifacts"])
    for a in m["artifacts"]:
        print(f"  {a['name']:14s} {a['bytes'] / 2**20:8.1f} MB  {a['sha256'][:12]}  {a['file']}")
    print(f"bundle {args.version}: {len(m['artifacts'])} artifacts, {total / 2**20:.0f} MB -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
