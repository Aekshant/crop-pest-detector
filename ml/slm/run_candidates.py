#!/usr/bin/env python3
"""L1: run each candidate SLM through llama.cpp's server and check it answers.

  python ml/slm/run_candidates.py            # all candidates
  python ml/slm/run_candidates.py qwen3-1.7b # one

For each model: start llama-server (CPU, ctx 2048), wait for /health, send a
few Laya-style prompts (English, Hindi, Hinglish) through the OpenAI-compatible
API, record the answers, llama.cpp's own timings (prompt and generation
tokens/sec) and the server's resident memory, then stop it.
Writes ml/slm/results/l1_smoke.json.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SLM = Path(__file__).resolve().parent
SERVER = SLM / "_deps" / "llama.cpp" / "build" / "bin" / "llama-server"
PORT = 8091
THREADS = int(os.environ.get("SLM_THREADS", "4"))

SYSTEM = ("You are Laya, a farming assistant for tomato growers in India. Answer in the same language "
          "as the farmer (Hindi, English or Hinglish), in 1-2 short sentences. Do not invent pesticide "
          "names or doses.")
PROMPTS = [
    ("en", "My tomato leaves have brown spots with rings. What could it be?"),
    ("hi", "मेरे टमाटर के पत्ते पीले हो रहे हैं, क्या करूँ?"),
    ("hinglish", "kal baarish hogi toh spray karna theek hai kya?"),
]


def post(path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def rss_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def run(model):
    path = SLM / "_models" / model["file"]
    if not path.exists():
        return {"id": model["id"], "error": f"missing {path.name}"}
    cmd = [str(SERVER), "-m", str(path), "--port", str(PORT), "-c", "2048", "-t", str(THREADS),
           "--no-webui", "--jinja", *model.get("extra_args", [])]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    try:
        while True:
            if proc.poll() is not None:
                return {"id": model["id"], "error": "server exited: " + proc.stderr.read()[-500:]}
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > 180:
                return {"id": model["id"], "error": "server did not become healthy in 180 s"}
            time.sleep(0.5)
        load_s = time.time() - t0
        answers = []
        for lang, prompt in PROMPTS:
            t = time.time()
            r = post("/v1/chat/completions", {
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "max_tokens": 96, "temperature": 0.2,
            })
            timings = r.get("timings", {})
            answers.append({
                "lang": lang, "prompt": prompt,
                "answer": r["choices"][0]["message"]["content"].strip(),
                "prompt_tokens": timings.get("prompt_n"), "gen_tokens": timings.get("predicted_n"),
                "prompt_tok_s": round(timings.get("prompt_per_second", 0), 1),
                "gen_tok_s": round(timings.get("predicted_per_second", 0), 1),
                "wall_s": round(time.time() - t, 2),
            })
        return {"id": model["id"], "file": model["file"], "size_mb": round(path.stat().st_size / 1e6),
                "load_s": round(load_s, 1), "rss_mb": round(rss_mb(proc.pid) or 0), "threads": THREADS,
                "answers": answers}
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    models = json.loads((SLM / "candidates.json").read_text())["models"]
    if len(sys.argv) > 1:
        models = [m for m in models if m["id"] in sys.argv[1:]]
    out = SLM / "results" / "l1_smoke.json"
    out.parent.mkdir(exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    for m in models:
        print(f"== {m['id']}", flush=True)
        r = run(m)
        results[m["id"]] = r
        if "error" in r:
            print("   ERROR", r["error"], flush=True)
            continue
        print(f"   load {r['load_s']} s, RSS {r['rss_mb']} MB, file {r['size_mb']} MB", flush=True)
        for a in r["answers"]:
            print(f"   [{a['lang']}] {a['gen_tok_s']} tok/s gen, {a['prompt_tok_s']} tok/s prompt: {a['answer'][:160]!r}", flush=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
