# Test devices (S4)

The plan calls for two devices: **your own phone**, and a **~4 GB RAM budget phone on
Firebase Test Lab**. Budget phones are the real users, and flagship numbers mislead.

Each device runs `GoldenTest` (expected labels on device) and `DeviceBenchmark`
(device facts, plus warm-up and gate / preprocess / inference timings on GPU and on CPU at
1, 2 and 4 threads). The report is checked against `docs/budget.md`: inference p50
< 80 ms on GPU (tier A/B) or < 150 ms on CPU INT8 (tier C), and gate < 10 ms.

| Device | How | Report |
|---|---|---|
| Your phone (USB) | `scripts/bench_device.sh` | pending |
| 4 GB budget phone (Test Lab) | `MODEL=… VERSION=… scripts/bench_ftl.sh` | pending |
| Android emulator (reference only) | `scripts/bench_device.sh` | `sdk_gphone64_x86_64.json` |

```bash
python3 scripts/bench_summary.py docs/devices/*.json
```

**The emulator report is not a phone.** It proves the pipeline works and gets 7/7 golden
labels, but its virtual x86 CPU runs XNNPACK INT8 about 160× slower than real hardware (it
exposes no AVX2/FMA). Its "OVER" verdicts say nothing about phones. The GPU delegate is not
run on emulators.
