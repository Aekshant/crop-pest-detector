#!/usr/bin/env python3
"""Ed25519 key for signing model bundles (design doc: "verifies the Ed25519 signature
with a public key compiled into the app").

  uvx --with cryptography python ml/bundle/bundle_key.py gen <key-id>

Writes the PRIVATE key to ~/.config/farmlens/bundle_<key-id>.pem (mode 600, never in
the repo) and appends the PUBLIC key to ml/bundle/bundle_public_keys.json, which the
app compiles in. Rotating = add a new key id, ship an app that trusts both, then sign
with the new one.
"""

import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

KEYS = Path(__file__).resolve().parent / "bundle_public_keys.json"
PRIVATE_DIR = Path.home() / ".config" / "farmlens"


def private_path(key_id):
    return PRIVATE_DIR / f"bundle_{key_id}.pem"


def gen(key_id):
    path = private_path(key_id)
    if path.exists():
        raise SystemExit(f"{path} exists: refusing to overwrite a signing key")
    key = Ed25519PrivateKey.generate()
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    os.chmod(path, 0o600)
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    keys = json.loads(KEYS.read_text()) if KEYS.exists() else {"note": "Ed25519 public keys the app trusts for bundles, by key id (raw 32 bytes, base64).", "keys": {}}
    keys["keys"][key_id] = base64.b64encode(raw).decode()
    KEYS.write_text(json.dumps(keys, indent=2) + "\n")
    print(f"private key: {path} (keep it safe; back it up offline)\npublic key {key_id} -> {KEYS}")


def load_private(key_id):
    return serialization.load_pem_private_key(private_path(key_id).read_bytes(), password=None)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        raise SystemExit(__doc__)
