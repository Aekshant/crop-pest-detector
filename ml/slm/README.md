# Laya SLM candidates (L1–L2)

The design doc's edge model is a 1.5–3B instruct model at Q4, served by llama.cpp. Its jobs:
turn free-form requests into tool calls (UNDERSTAND), and write a short explanation sentence.
Treatments come from KB templates, never from the model.

```bash
# llama.cpp (pinned, CPU) and the GGUFs go to gitignored _deps/ and _models/
git clone https://github.com/ggml-org/llama.cpp ml/slm/_deps/llama.cpp   # commit 18b5f8b
cmake -S ml/slm/_deps/llama.cpp -B ml/slm/_deps/llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build ml/slm/_deps/llama.cpp/build -j --target llama-server llama-cli llama-bench
# download the files listed in candidates.json into ml/slm/_models/, then:
python3 ml/slm/run_candidates.py
```

## L1: do they answer? (`results/l1_smoke.json`)

The setup is the laptop CPU (i5-1135G7, 4 threads), context 2048, Q4_K_M, and three Laya-style
prompts in English, Hindi and Hinglish.

| | Qwen3 1.7B | Llama 3.2 3B | Gemma 3 1B |
|---|---|---|---|
| GGUF size | 1.1 GB | 2.0 GB | 0.8 GB |
| Server RAM (RSS) | 2.0 GB | **3.5 GB** | **1.0 GB** |
| Generation, laptop | 25 tok/s | 13 tok/s | 28 tok/s |
| English prompt | leaked `</think>`, answered in Hindi | answered in Hinglish | ✅ English |
| Hindi prompt | ❌ garbled Hindi | ✅ coherent | ✅ coherent |
| Hinglish prompt | ❌ answered in Urdu script | ✅ | ✅ |

All three answer, so the L1 done-when is met. First impressions:

- **Gemma 3 1B** matches the language best, and is the smallest and fastest. Its farm advice
  is poor, but Laya doesn't give advice freely.
- **Llama 3.2 3B** has the best Hindi, but at 3.5 GB of RAM it is tight for tier B (6 GB phones).
- **Qwen3 1.7B** is weak at free-text Hindi and Hinglish. It may still be strongest at tool
  calling, which is what L2 measures. Its reasoning mode leaked `</think>` even with
  `--reasoning-budget 0`. L2 disables thinking through the chat template.

Phone speed will be far lower than the laptop's. The design routes Laya to the cloud below
4 tok/s on the device.
