# Model bundle

Models, prompts, tool schema and the KB pack ship together as one **signed, versioned
bundle** (design doc, LLD "Model bundle lifecycle"). The APK carries the vision models as a
factory bundle. Everything else, and every later model, arrives as a downloaded bundle.

## Publishing

```bash
uvx --with cryptography python ml/bundle/bundle_key.py gen <key-id>                 # once per signing key
uvx --with cryptography python ml/bundle/build_bundle.py <version> --key-id <key-id> [--with-slm]
# upload dist/bundles/ as is: <host>/manifest.json and <host>/<version>/<files>
```

The app is built with the host URL:
`./gradlew assembleRelease -Pfarm.bundleBaseUrl=https://…`. If the URL is empty, the app
never updates (the current state: there's no host yet).

**Artifacts in `2026.10.15` (133 MB):**

| Artifact | Contents | Size |
|---|---|---|
| vision | INT8 model | 3.9 MB |
| vision_gpu | FP16 model (tiers A/B) | 6.6 MB |
| vision_config, labels | vision config and labels | small |
| tool_schema, templates, fewshot, safety | Laya's JSON files | small |
| kb | KB pack | 2.1 MB |
| kb_embedder | e5-small int8 | 112.6 MB |
| kb_vocab | vocabulary | 7.4 MB |
| laya (`--with-slm`) | Qwen3 1.7B GGUF (tiers A/B) | about 1.1 GB |

**Signature.** `manifest.json` is `{"key_id", "manifest": base64(manifest JSON bytes),
"signature": "ed25519:…"}`. The signature covers the exact manifest bytes, unlike the
design doc's embedded field over re-serialized JSON, so Python and Kotlin can never
disagree about canonical form. `make_test_fixture.py` writes a Python-signed fixture that
the Kotlin verifier must accept (`BundleManifestTest`).

**Keys:**

- `bundle_public_keys.json` holds the public keys the app trusts, compiled into
  `BuildConfig.BUNDLE_PUBLIC_KEYS`.
- Private keys live **outside the repo**, at `~/.config/farmlens/bundle_<key-id>.pem`
  (mode 600).
- `dev-2026-10` is a development key on the build machine. Production needs its own key,
  held by you, backed up offline, and never on a CI runner that builds untrusted code.
- To rotate: add a key, ship an app that trusts both, then sign with the new one.

## On the phone (`android/.../bundle/`)

`BundleWorker` runs every 12 h on an unmetered network, **or** while charging (two
WorkManager requests, since constraints can't express "or"; a lock serializes them).
`BundleUpdater` then follows the design doc's flow:

1. Fetch `manifest.json`.
2. Verify the Ed25519 signature (BouncyCastle; `java.security` has Ed25519 only from
   API 33). If it fails, discard the manifest.
3. Check `min_app_version` against `versionCode`, `tool_schema_version` against the
   supported versions, and the device tier against the manifest's tiers (A ≥ 7 GB RAM,
   B ≥ 5 GB, else C). If any check fails, skip and keep the current bundle.
4. Get **only changed artifacts**: same sha256 means copied from the current bundle.
   Downloads go to `staging/<version>/*.part`, **resume with HTTP Range** (or restart if
   the server ignores Range), and are checked for size and sha256. File names are
   allow-listed even though they are signed.
5. **Smoke test** (`BundleSmoke`):
   - all needed artifacts are present;
   - the tool schema, templates and safety lexicon parse, and the schema version matches;
   - **the app's own 5 golden images** (`assets/smoke`, expected labels by ID) get the
     right labels on CPU INT8 and, if shipped, GPU FP16;
   - the KB pack opens with rows and chunks;
   - the embedder returns a unit vector;
   - the GGUF header checks out.
   - Still to do: the "3 golden Laya prompts" step, which needs an on-device SLM runtime.
6. If it passes, the candidate moves into `versions/` and `bundle_state.json` is switched
   by an atomic rename. If it fails, the candidate is deleted and the failure reported.
7. The old bundle stays as `previous`, and anything older is deleted.

Release builds are HTTPS-only (Android's default). Debug builds allow cleartext to
`127.0.0.1` only, for the device test below.

## Tests

- **JVM** (`BundleManifestTest`, `BundleUpdaterTest`: 14 tests against a real local HTTP
  server). They cover:
  - install;
  - up-to-date;
  - changed-files-only (600 bytes fetched for a changed 600-byte file);
  - **resume after a dropped connection** (exactly the remaining bytes fetched);
  - a server that ignores Range;
  - a corrupt artifact;
  - incompatible app and schema versions;
  - tier filtering;
  - a failed smoke test;
  - a bad signature;
  - Python-signed fixture, tampered, unknown-key and wrong-key manifests;
  - unsafe file names.
- **Device** (`BundleDeviceTest`, POCO M6 Plus, 2026-10-15). The real bundle was served
  from the dev machine over `adb reverse`:
  - `2026.10.15`: signature verified, 139 MB fetched, smoke test passed (CPU and GPU
    goldens, KB, embedder), activated in 10.9 s, then up to date.
  - `2026.10.15-bad`, with the vision labels shuffled: **rejected**, "golden 00_healthy:
    got fruit_borer, expected healthy".
  - This test caught a real hole first: the smoke test looked expected labels up through
    the *candidate's* config, so the shuffled bundle was activated. Expected labels now
    come from the app's `golden.json`, by ID.

```bash
python3 -m http.server 8099 --bind 127.0.0.1 --directory dist/bundles &
adb reverse tcp:8099 tcp:8099
adb shell am instrument -w -e class com.farm.app.bundle.BundleDeviceTest -e bundleUrl http://127.0.0.1:8099 \
    com.farm.app.test/androidx.test.runner.AndroidJUnitRunner
```

**Not done yet:**

- the bundle host, plus a production signing key;
- making the app *use* the activated bundle: vision and Laya still read the factory
  assets, and the next step is to prefer `BundleStore.currentDir()` (`VisionFiles.dir`,
  `KbBundle`);
- the Laya smoke step;
- a tier rule that accounts for GPU delegate failures (DeviceProfiler).
