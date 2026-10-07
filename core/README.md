# core: portable C++ vision core

The same C++ runs on Android (through JNI) and on a Linux host (tests). Inference via
LiteRT is built only for Android.

| Part | What | Matches |
|---|---|---|
| `image` | Pillow-exact bilinear resize, box blur, RGB→luma; YUV_420_888→RGB (full-range BT.601) | Pillow, bit for bit |
| `quality` | Quality gate: blur effect at a 512 px short side + brightness, with reused buffers | `ml/quality_gate.py` |
| `preprocess` | Resize short side to 256 → center-crop 224 → rotate upright → NHWC float, written straight into the input tensor | `ml/export.py` `preprocess()` |
| `decision` | temperature, softmax, then retake / not recognised / uncertain / confident | manifest thresholds |
| `engine` (Android) | LiteRT 1.x C API: GPU delegate + FP16, else CPU (XNNPACK) + INT8; warm-up at creation; single-thread use | `ml/bundle/manifest_draft.json` |

```bash
cmake -S core -B build/core -DFARM_CORE_BUILD_TESTS=ON && cmake --build build/core && ctest --test-dir build/core
```

## Golden images: `tests/golden/`

There are 8 **validation** photos (never eval photos) with Python reference outputs,
written by `ml/golden.py`:

- `.rgb` is the input.
- `.pre` is the Python-preprocessed 224×224 tensor.
- `golden.tsv` / `golden.json` hold sharpness, brightness, the expected label and logits.

Host tests require **identical** preprocessing (0 differing pixels) and sharpness within
1e-5. The camera YUV path must stay within a mean difference of 4 of the RGB path. The
Android test `GoldenTest` runs the real engine on device and checks labels. A golden image
below the blur threshold must come back as RETAKE.

## Measured

| | Host (i5-1135G7) | Android emulator (x86_64, API 34) |
|---|---|---|
| Quality gate, 640×480 | 6.2 ms | 5.7 ms median |
| Preprocess (golden) | — | 0.2–2.8 ms |
| INT8 inference, 1 thread | 6.0 ms (Python LiteRT) | ~965 ms (*) |
| Engine warm-up | — | 1.1–2.0 s |

(*) The emulator's virtual CPU exposes no AVX2/FMA, and XNNPACK in LiteRT 1.4's x86_64 build
runs about 160× slower there than on the host. This says nothing about phones, which are
ARM. **Phone latency must be measured on real devices** (task S4) before checking the
budget (< 150 ms CPU INT8 on tier C).

## Notes

- **External weight buffers:** `ai-edge-quantizer` writes weights after the flatbuffer,
  referenced by offset, which LiteRT 1.x cannot read ("Input tensor N lacks data").
  `ml/export.py` rewrites both files with all buffers inline.
- **GPU:** the GPU delegate is never tried on emulators. Their GPU is a software renderer,
  and shader compilation there hung the emulator rather than failing cleanly. On phones,
  DeviceProfiler should validate the GPU path once with a timeout and remember the result.
