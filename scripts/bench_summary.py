#!/usr/bin/env python3
"""Summarize DeviceBenchmark reports (docs/devices/*.json) against the budget."""

import json
import sys
from pathlib import Path


def main():
    rows = []
    for p in sys.argv[1:]:
        try:
            r = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"{p}: unreadable ({e})")
            continue
        d = r["device"]
        name = f"{d['manufacturer']} {d['model']}" + (" (emulator)" if d.get("emulator") else "")
        for run in r["runs"]:
            if "error" in run:
                rows.append((name, d, run["config"], f"ERROR: {run['error']}"))
                continue
            limit = 80 if run.get("backend") == "GPU" else 150
            ok = "OK" if run["inference_ms_p50"] < limit else "OVER"
            rows.append((name, d, run["config"],
                         f"{run.get('backend')}: infer p50 {run['inference_ms_p50']:.1f} / p90 {run['inference_ms_p90']:.1f} ms "
                         f"[{ok} < {limit}] | gate p50 {run['gate_ms_p50']:.1f} ms | prep {run['preprocess_ms_p50']:.1f} ms | "
                         f"warm-up {run['warmup_ms']:.0f} ms | golden {run['golden_correct']}/{run['golden_gradable']}"))
    last = None
    for name, d, cfg, text in rows:
        if name != last:
            print(f"\n{name}  ({d['soc']}, {d['ram_gb']} GB RAM, {d['cores']} cores, Android {d['android']})")
            last = name
        print(f"  {cfg:13s} {text}")


if __name__ == "__main__":
    main()
