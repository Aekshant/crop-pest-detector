# core: portable C++ vision core

The same C++ runs on Android (through JNI) and on a Linux host (tests). Inference via
LiteRT is built only for Android.

| Part | What | Matches |
|---|---|---|
| `image` | Pillow-exact bilinear resize, box blur, RGB→luma; YUV_420_888→RGB (full-range BT.601) | Pillow, bit for bit |
| `quality` | Quality gate: blur effect at a 512 px short side + brightness, with reused buffers | `ml/quality_gate.py` |
| `preprocess` | Resize short side to 256 → center-crop 224 → rotate upright → NHWC float, written straight into the input tensor | `ml/export.py` `preprocess()` |
| `severity` | Lesion share of the leaf from HSV segmentation, for disease labels only, bucketed low < 10% ≤ medium < 30% ≤ high; "can't estimate" if under 15% of the crop is leaf | (A7) checked by eye on 10 sample leaves |
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

## Severity estimate (A7): how far to trust it

It runs on the 224×224 crop the model saw, takes about 4 ms (emulator), and is always shown
as "estimated". It is never derived from model confidence.

What it gets right, checked by eye on 10 lab and field leaves plus the golden images:

- Healthy leaves read 0–2%.
- Spot diseases (septoria, bacterial spot) on mostly green leaves read "low".
- Clear lesions on a close-up read medium or high, and the lesion masks sit on the lesions.

Known failure modes:

- **Grey-brown dead tissue on grey backgrounds** (PlantVillage lab photos) is colourless
  like the paper, so it is missed. A nearly dead leaf then reads "can't estimate" (too
  little leaf found) rather than falsely low.
- **Brown or yellow out-of-focus background** next to the leaf, and **yellow-green stems**,
  can be counted as lesion, which overestimates.
- **Whole-plant scenes** are rough. The estimate is meant for a close-up of one leaf, which
  is what the scan screen asks for.

Real calibration needs leaves with an expert-rated severity, which can come from own field
photos.

## Native safety (A8)

- **JNI never crashes the app.** Every entry point runs in `Guarded()`. It checks the
  handle against the set of live engines (a stale, zero or made-up handle is rejected, not
  dereferenced), validates sizes, strides and buffers, and catches every C++ exception.
  Each call returns an error code: `INVALID_HANDLE`, `INVALID_ARGUMENT`,
  `BUFFER_NOT_DIRECT`, `NATIVE_EXCEPTION`, `OUT_OF_MEMORY` or `INFERENCE_FAILED`. Kotlin
  gets `VisionOutcome.Failed(code)`: the camera skips the frame, and an inference failure
  on the GPU switches the engine to CPU for the session. Engine creation is the exception:
  it throws `IllegalStateException`, which is caught where the engine is created.
- `nativeForceError(kind)` throws on purpose. `ErrorHandlingTest` (on device) checks that
  forced exceptions, bad handles, bad arguments, non-direct buffers and corrupt models all
  come back as codes or exceptions, and that the engine still works afterwards.
- **AddressSanitizer:**
  - Host: `cmake -DFARM_CORE_SANITIZE=ON` (ASan + UBSan) runs all core tests. It is clean,
    and CI runs it.
  - Android: the `asan` build type sanitizes the core and JNI with `c++_shared`, and packages
    the NDK ASan runtime and the NDK's `wrap.sh`:
    `./gradlew assembleAsan assembleAsanAndroidTest -Pfarm.testBuildType=asan`. The error and
    golden tests pass under it on the emulator. Prebuilt LiteRT is not instrumented.

## Notes

- **External weight buffers:** `ai-edge-quantizer` writes weights after the flatbuffer,
  referenced by offset, which LiteRT 1.x cannot read ("Input tensor N lacks data").
  `ml/export.py` rewrites both files with all buffers inline.
- **GPU:** the GPU delegate is never tried on emulators. Their GPU is a software renderer,
  and shader compilation there hung the emulator rather than failing cleanly. On phones,
  DeviceProfiler should validate the GPU path once with a timeout and remember the result.
