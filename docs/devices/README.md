# Test devices (S4)

The plan calls for two devices: **your own phone**, and a **~4 GB RAM budget phone on
Firebase Test Lab**. Budget phones are the real users, and flagship numbers mislead.

Each device runs `GoldenTest` (expected labels on device) and `DeviceBenchmark`
(device facts, plus warm-up and gate / preprocess / inference timings on GPU and on CPU at
1, 2 and 4 threads). The report is checked against `docs/budget.md`: inference p50
< 80 ms on GPU (tier A/B) or < 150 ms on CPU INT8 (tier C), and gate < 10 ms.

| Device | How | Report |
|---|---|---|
| Your phone (USB): Xiaomi POCO M6 Plus 5G (SM4450, 8 GB, Android 16, tier A) | `scripts/bench_device.sh` | `24066PC95I.json` |
| 4 GB budget phone (Test Lab) | `MODEL=… VERSION=… scripts/bench_ftl.sh` | skipped for now (needs a Google Cloud login); script ready |
| Android emulator (reference only) | `scripts/bench_device.sh` | `sdk_gphone64_x86_64.json` |

```bash
python3 scripts/bench_summary.py docs/devices/*.json
```

**The emulator report is not a phone.** It proves the pipeline works and gets 7/7 golden
labels, but its virtual x86 CPU runs XNNPACK INT8 about 160× slower than real hardware (it
exposes no AVX2/FMA). Its "OVER" verdicts say nothing about phones. The GPU delegate is not
run on emulators.

## POCO M6 Plus 5G (S4, 2026-10-15)

The phone is a Snapdragon 4 Gen 2 (SM4450): 2 fast cores (A78, `cpu_capacity` 1024) and
6 slow ones (A55, 458), Adreno GPU, 8 GB RAM, Android 16. **Every configuration gets 7/7
golden labels.**

| | GPU FP16 (app setting) | CPU INT8 (app default: 2 threads) | Budget, tier A |
|---|---|---|---|
| inference p50 | 20 ms | 25–32 ms back-to-back, 45–54 ms camera-paced | < 80 ms GPU, < 150 ms CPU |
| quality gate p50 | 5.2–6.2 ms | 2.7 ms (4.7 ms camera-paced) | < 10 ms |

**What S4 found and fixed:**

1. **Thread count.** The old default (cores, at most 4) picked 4 threads, the slowest
   setting: 68 ms, against 24 ms for 2 threads. XNNPACK threads wait for each other at every
   layer, so threads that land on the slow cores pace everything.
   `VisionEngine.defaultThreads()` now counts only the fast cores (`cpu_capacity` ≥ 60% of
   the fastest, else max frequency ≥ 90%), at most 4. `ThreadsTest` covers this layout,
   prime + big, uniform cores, and the fallbacks.
2. **Quality gate in GPU mode: 30 ms instead of 6.** With camera-paced frames, the vision
   thread mostly waits for the GPU, so the CPU idles between frames and clocks drop.
   Pinning the thread to the fast cores (`nativePinThread`) plus an ADPF performance hint
   (`FrameHint`, Android 12+) brought it to 13.8 ms (median of 3 runs). A 10 ms hint
   target was no better than 25 ms.
3. **The gate itself, 2.2× faster on the phone (6.0 → 2.7 ms), with byte-identical
   results.** The changes:
   - the resize uses 32-bit accumulators and a row-order vertical pass;
   - the box blur slides a row of accumulators down the image;
   - the metric has no per-pixel modulo.

   `fast_path_test.cpp` compares 756 randomized cases with the Pillow-checked reference
   (`tests/ref_quality.h`), so the A4 calibration still holds. GPU mode with camera pacing
   is now 5–6 ms.

**Method:**

- Each figure is the median of at least 3 runs. Single runs on this phone vary a lot (the
  unpinned, unhinted paced gate measured 7–45 ms across runs).
- Runs were spaced 45 s apart. Thermal status stayed 0 and the battery at 35 °C, charging,
  with the screen on.
- The benchmark keeps the "plain" and "paced" variants so the effect of pinning and the
  hint stays visible.
