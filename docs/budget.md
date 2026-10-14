# Performance budget

Source: *FarmLens — HLD & LLD (Hybrid Edge/Cloud)*, "Performance budget and observability"
and the LLD sections it summarises.

Every number here is a **pass/fail gate**. A change that pushes a measured value past its
budget fails, unless this file is updated in the same PR and the reason is written down.

These are **targets, not measurements.** The device-farm runs (build order step 8, Oct 26–28)
replace them with measured values. Until then, the "Measured" column stays `—`.

Tiers (from the design doc):

| Tier | Device                                  | Vision          | Laya (SLM)    |
|------|-----------------------------------------|-----------------|---------------|
| A    | 8 GB+ RAM, working GPU delegate         | Edge, GPU FP16  | Edge, 1.5–3B Q4 |
| B    | 6 GB RAM                                | Edge, GPU or CPU INT8 | Edge, 1.5B Q4 |
| C    | 4 GB RAM, or GPU delegate fails         | Edge, CPU INT8  | Cloud only    |

## 1. End-to-end latency

| ID   | Path                                       | Tier A/B        | Tier C          | Cloud path         | Measured |
|------|--------------------------------------------|-----------------|-----------------|--------------------|----------|
| L1   | Quality gate (blur + exposure, no ML)      | < 10 ms         | < 15 ms         | —                  | 5.2–6.2 ms GPU mode, 2.7 ms CPU mode (POCO M6 Plus, tier A, S4) |
| L2   | Vision inference, warm                     | < 80 ms (GPU)   | < 150 ms (CPU INT8) | < 1.2 s round trip on 4G | 20 ms GPU FP16; 25–45 ms CPU INT8, 2 threads (POCO M6 Plus, S4) |
| L3   | Shutter → result, warm model               | < 150 ms        | —               | —                  | —        |
| L4   | Photo → spoken diagnosis (template)        | < 2 s           | < 2.5 s         | —                  | —        |
| L5   | Laya time to first token                   | < 1.5 s         | — (cloud)       | < 1.5 s on 4G      | —        |
| L6   | Laya UNDERSTAND step (≤ 64 tokens)         | < 4 s           | —               | < 2 s              | —        |
| L7   | App cold start → camera ready              | < 2.5 s         | < 3.5 s         | —                  | 1.1 s to first frame, release build (POCO M6 Plus; camera-ready not timed separately) |

## 2. Memory

| ID   | Item                                       | Tier A/B        | Tier C          | Measured |
|------|--------------------------------------------|-----------------|-----------------|----------|
| M1   | Peak app RAM (PSS) per session             | < 2.5 GB        | < 1.2 GB        | —        |
| M2   | Vision model, resident (INT8)              | ≈ 5–15 MB       | ≈ 5–15 MB       | —        |
| M3   | Per-frame allocations in preprocess        | 0 (reused buffer) | 0             | —        |
| M4   | SLM resident while camera open             | A: allowed · B: must be unloaded first | not loaded (cloud) | — |

Lifecycle rules that the memory budget depends on:

- Vision model is released in `onTrimMemory` when the app is backgrounded.
- SLM loads on the first free-form request and is kept until backgrounding or memory pressure.
- OOM while loading the SLM → Laya marked edge-unavailable for the session (cloud or template).
- Two OOMs within 24 h of a bundle activation → automatic rollback.

## 3. Power and thermal

| ID   | Item                                       | Budget          | Measured |
|------|--------------------------------------------|-----------------|----------|
| P1   | Battery drain per 5-minute session         | < 3 % (all tiers) | —      |
| P2   | Live-scan sampling rate                    | 3–5 frames/s    | —        |
| P3   | Thermal status at which live scan pauses   | ≥ SEVERE        | —        |
| P4   | Battery level at which heavy edge tasks move off-device | < 15 %, not charging | — |

## 4. Edge runtime limits

| ID   | Item                                       | Limit           |
|------|--------------------------------------------|-----------------|
| R1   | Edge task timeout                          | 1.5 × profiled latency |
| R2   | Laya on edge requires throughput           | ≥ 4 tokens/s (else cloud) |
| R3   | Laya total generation per turn             | ≤ 8 s (watchdog → template) |
| R4   | Laya per turn                              | ≤ 3 tool calls, ≤ 2 clarifications |
| R5   | System prompt + tool schema                | < 600 tokens    |
| R6   | Output caps                                | UNDERSTAND ≤ 64 tokens, explanation ≤ 120 tokens |
| R7   | First-launch tier benchmark                | ≈ 10 s (1 vision inference + 32 SLM tokens) |

## 5. Network and cloud

| ID   | Item                                       | Limit           |
|------|--------------------------------------------|-----------------|
| N1   | Health check for ONLINE                    | < 800 ms (else CONSTRAINED) |
| N2   | Client cloud call timeout                  | 4 s (2 s for text on CONSTRAINED) |
| N3   | Upload image                               | 512 px JPEG, q80, EXIF stripped, ≈ 40–80 KB |
| N4   | Circuit breaker open window                | 30 s, then one probe |
| N5   | Per-device rate limit                      | 30 inference requests / min |

Gateway endpoint timeouts:

| Endpoint                     | Timeout          |
|------------------------------|------------------|
| `GET /v1/health`             | 500 ms           |
| `POST /v1/vision/diagnose`   | 3 s              |
| `POST /v1/laya/chat`         | 4 s to first token |
| `POST /v1/asr`               | 4 s              |
| `POST /v1/sync/events`       | 5 s (≤ 100 events) |
| `GET /v1/sync/changes`       | 5 s              |
| `GET /v1/bundles/latest`     | 2 s              |
| `POST /v1/pending`           | 10 s             |

## How each number is checked

| Budget          | Measured with                                         | Where |
|-----------------|-------------------------------------------------------|-------|
| L1–L3, L7       | On-device benchmark (warm, median of N runs; report p50 and p95) | device farm |
| L4–L6           | Instrumented end-to-end runs from the debug screen telemetry | device farm |
| M1              | `dumpsys meminfo` peak total PSS over a scripted 5-minute session | device farm |
| M2              | Model file size + resident delta after load           | CI + device farm |
| P1              | Battery delta over the same scripted 5-minute session | device farm |
| N1–N5, gateway  | Gateway Prometheus metrics (p50/p95/p99, 429 rate)    | server |

**Pass rule:** a latency budget passes when **p95 ≤ budget** on every reference device of
that tier. A memory or battery budget passes when the **worst run** is within budget.
