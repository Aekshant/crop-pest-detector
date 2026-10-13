#!/usr/bin/env python3
"""L10: time-to-first-token with and without reusing the cached system prompt.

  python ml/slm/bench_ttft.py qwen3-1.7b [--n 5]

Laya's UNDERSTAND prompt is a long fixed system prompt (instructions, tool
list, L5 few-shot examples) followed by a short per-turn message (app context
and the farmer's words). Without reuse every turn re-processes the whole
prompt. Three modes, same requests (n cases per language from
evals/laya/cases.jsonl, the L5 prompt and grammar):

  no_cache  cache_prompt=false: the whole prompt every turn ("before")
  warm      cache_prompt=true, consecutive turns in one session ("after"): the
            system prompt and the unchanged start of the app context are reused
  restored  a fresh server; before every request the system-prompt-only KV
            state is restored from a file (llama-server slot save/restore), so
            only the per-turn message is new. This is the first turn after the
            app process restarts, and the pure "system prefix cached" number

TTFT is measured client-side on a streamed request (request sent -> first
content chunk). Results: ml/slm/results/l10_ttft_<model>.json
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_laya_eval as R  # noqa: E402
from run_laya_eval import tool_schema  # noqa: E402

BASE = f"http://127.0.0.1:{R.PORT}"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ttft(model, system, messages, schema, cache):
    body = {"messages": [{"role": "system", "content": system}, *messages], "max_tokens": 8, "temperature": 0.0,
            "cache_prompt": cache, "id_slot": 0, "stream": True, "timings_per_token": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}}
    if model["id"].startswith("qwen3"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, timings = None, {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if first is None and any((c.get("delta") or {}).get("content") for c in chunk.get("choices", [])):
                first = time.perf_counter() - t0
            timings = chunk.get("timings") or timings
    return {"ttft_ms": round(1000 * first), "prompt_n": timings.get("prompt_n"), "cached_n": timings.get("cache_n"),
            "prompt_ms": round(timings.get("prompt_ms", 0))}


def summary(rows):
    t = [r["ttft_ms"] for r in rows]
    return {"n": len(t), "median_ms": round(statistics.median(t)), "mean_ms": round(statistics.mean(t)),
            "max_ms": max(t), "median_prompt_tokens_processed": statistics.median(r["prompt_n"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=5, help="cases per language")
    args = ap.parse_args()
    model = next(m for m in json.loads((R.SLM / "candidates.json").read_text())["models"] if m["id"] == args.model)

    schema = json.loads((R.REPO / "ml" / "bundle" / "tools_schema_v1.json").read_text())
    fewshot = json.loads((R.REPO / "ml" / "bundle" / "laya_fewshot_v1.json").read_text())
    default_ctx = json.loads((R.LAYA / "context_default.json").read_text())
    online = tool_schema.understand_schema(schema)
    system = R.understand_system(schema, fewshot)
    cases = R.load_jsonl(R.LAYA / "cases.jsonl")
    picked = [c for lang in ("en", "hi", "hinglish") for c in [c for c in cases if c["lang"] == lang
              and {**default_ctx, **c.get("context", {})}["network"] == "ONLINE"][:args.n]]
    reqs = [(c["id"], R.understand_messages(c, default_ctx)) for c in picked]

    slots = Path(tempfile.mkdtemp(prefix="laya_slots_"))
    bench_model = {**model, "extra_args": [*model.get("extra_args", []), "-np", "1", "--slot-save-path", str(slots)]}
    results = {}

    def run(mode, cache, before=None):
        rows = []
        for i, (cid, msgs) in enumerate(reqs, 1):
            if before:
                before()
            r = {"id": cid, **ttft(model, system, msgs, online, cache)}
            rows.append(r)
            print(f"  {mode:9s} {i:2d}/{len(reqs)} {cid}  TTFT {r['ttft_ms']:6d} ms  "
                  f"prompt tokens processed {r['prompt_n']}", flush=True)
        results[mode] = {"summary": summary(rows), "requests": rows}

    print(f"L10 TTFT, {model['id']}, {R.THREADS} threads, {len(reqs)} requests per mode", flush=True)
    server = R.Server(bench_model)
    try:
        run("no_cache", False)
        # A dummy turn (not a measured case) puts the system prompt in the cache.
        ttft(model, system, [{"role": "user", "content": "Farmer: namaste"}], online, True)
        t = time.perf_counter()
        saved = post("/slots/0?action=save", {"filename": "laya_system.bin"})
        save_ms = round(1000 * (time.perf_counter() - t))
        run("warm", True)
    finally:
        server.stop()

    server = R.Server(bench_model)
    try:
        restore_times = []

        def restore():
            t = time.perf_counter()
            post("/slots/0?action=restore", {"filename": "laya_system.bin"})
            restore_times.append(1000 * (time.perf_counter() - t))

        run("restored", True, before=restore)
        restore_ms = round(statistics.median(restore_times))
    finally:
        server.stop()

    file_mb = round((slots / "laya_system.bin").stat().st_size / 2**20, 1)
    report = {
        "model": model["id"], "threads": R.THREADS, "host": "CPU (x86_64), llama.cpp llama-server",
        "prompt": "L5 UNDERSTAND system prompt (instructions, tools, few-shot) + per-turn context and message; grammar on",
        "cached_prefix_tokens": saved.get("n_saved"), "slot_file_mb": file_mb, "slot_save_ms": save_ms,
        "slot_restore_ms": restore_ms,
        "speedup_median": {m: round(results["no_cache"]["summary"]["median_ms"] / results[m]["summary"]["median_ms"], 1)
                           for m in ("warm", "restored")},
        **results,
    }
    out = R.SLM / "results" / f"l10_ttft_{model['id']}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\n{'mode':10s} {'median TTFT':>12s} {'mean':>8s} {'max':>8s} {'prompt tokens processed':>24s}")
    for mode in ("no_cache", "warm", "restored"):
        s = results[mode]["summary"]
        print(f"{mode:10s} {s['median_ms']:9d} ms {s['mean_ms']:5d} ms {s['max_ms']:5d} ms {s['median_prompt_tokens_processed']:>24}")
    print(f"\ncached prefix {report['cached_prefix_tokens']} tokens (system prompt + dummy turn), slot file {file_mb} MB, "
          f"save {save_ms} ms, restore {restore_ms} ms (median)")
    print(f"median speedup vs no_cache: {report['speedup_median']}   -> {out.relative_to(R.REPO)}")


if __name__ == "__main__":
    main()
