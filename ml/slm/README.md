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

## L2: Laya eval baseline (`results/l2_*`)

```bash
python3 ml/slm/run_laya_eval.py <model-id>   # 150 cases + 30 safety cases, scored by evals/score_laya.py
```

This is a baseline on purpose. It uses instructions only: one fixed system prompt (tools,
labels, local names, language rule) and the case's app context in the user turn. There is
no JSON grammar (L4) and no examples (L5). Temperature is 0, and llama-server reuses the
cached system prompt. For safety cases, the model picks one of the 12 safety templates and
writes a reply, given only the KB rows a real turn would have. The raw output is scored as-is.

| | Gemma 3 1B | Qwen3 1.7B | Llama 3.2 3B |
|---|---|---|---|
| JSON valid | 38.7% | 72.7% | **100%** |
| Schema valid | 28.7% | 72.7% | **96.0%** |
| Action/tool accuracy | 10.0% | **23.3%** | **23.3%** |
| Arg accuracy (tool cases) | 8.1% | **11.7%** | 4.5% |
| Case pass | 4.7% | **22.0%** | 21.3% |
| Reply language: en / hi / hinglish | 94 / 62 / 0% | 95 / 38 / 24% | 30 / **100** / **98**% |
| Safety pass | 0% | 30% | **40%** |
| Generation (laptop, 4 threads) | 26.6 tok/s | 20.3 tok/s | 12.2 tok/s |
| Seconds per case (median) | 1.6 | 3.5 | 5.5 |
| Peak RAM (ctx 4096) | 1.2 GB | 2.5 GB | **4.0 GB** |

What the failures show:

- **Tools are under-used.** The top failure for Qwen and Llama is replying directly where a
  tool was expected (60–67 cases each). Their passes are almost all "respond" cases (95%),
  while most tool families pass at 0–20%. Examples (L5) address this.
- **Qwen3**: 27% of outputs are invalid JSON, which constrained decoding (L4) fixes. Hindi
  and Hinglish replies often come back in English.
- **Llama 3.2**: always valid JSON and near-perfect Hindi and Hinglish, but it answers
  *English* questions in Hinglish. 4 GB of RAM is too much for tier B (6 GB phones).
- **Gemma 3 1B**: can't keep the format (61% invalid JSON, every safety reply ignored the
  template, scripts drifted to Tamil and Marathi). **Dropped.**
- **Safety**: every model invented doses or named banned chemicals in some replies. Raw
  SLM text must never reach the farmer on pesticide topics. Templates (L8) and the safety
  validator (L9) are required.

**Decision:** Qwen3 1.7B and Llama 3.2 3B go on to L4 and L5. Pick after L5, possibly one per
tier: about 3B on 8 GB phones (tier A) and about 1.5B on 6 GB phones (tier B), as the design doc
suggests.

## L4: constrained decoding (`results/l4_*`)

```bash
python3 ml/slm/run_laya_eval.py <model-id> --grammar
```

The prompts are the same as L2. Each request also sends the UNDERSTAND JSON Schema from
`ml/bundle/tool_schema.py` as `response_format`, and llama.cpp compiles it to a grammar, so
only schema-valid output can be generated. When the case is OFFLINE, the schema has no
network tools (the router disables them). Safety replies get a schema too: a template id
from the catalogue, a capped reply and a certainty value.

| | Qwen3 1.7B (L2 → L4) | Llama 3.2 3B (L2 → L4) |
|---|---|---|
| JSON valid | 72.7% → **100%** | 100% → **100%** |
| Schema valid | 72.7% → **100%** | 96.0% → **100%** |
| Action/tool accuracy | 23.3% → 38.7% | 23.3% → 43.3% |
| Arg accuracy (tool cases) | 11.7% → 31.5% | 4.5% → 36.0% |
| Case pass | 22.0% → **36.7%** | 21.3% → **38.0%** |
| Reply language match | 53.2% → 58.7% | 76.3% → 75.3% (en 30%, hi 100%, hinglish 96%) |
| Safety pass | 30.0% → 30.0% | 40.0% → 36.7% |
| Seconds per case (laptop) | 4.3 | 7.0 |

L4 done-when: **JSON validity = 100% for both models.**

Getting from about 97% to 100% took three fixes to the schema. They live in
`ml/bundle/tool_schema.py`, so the app gets them too:

1. **Length caps** (`MAX_REPLY_CHARS` / `MAX_TEXT_ARG_CHARS` = 120). Without them, a looping
   reply ("बर्फ के बर्फ के …") ran into the token limit and the JSON was cut off. With a
   cap, the grammar forces the string, and the JSON, to close.
2. **A token budget that covers the worst case** (`MAX_OUTPUT_TOKENS` = 1024). Devanagari
   can take about 3 tokens per character, so a 120-character cap still overflowed 200 tokens.
3. **Free text limited to Devanagari, Latin letters, digits and common punctuation, with
   no `\u` escapes.** Llama looped on `\uXXXX` escapes, which count as one character but
   cost many tokens. The same rule stops replies drifting into Urdu, Tamil, Marathi or
   Gujarati script, as L1 and L2 showed.

What the grammar does not fix: it makes outputs well formed, not correct. The models still
pick a direct reply or the wrong tool in about 60% of cases. Llama still answers English in
Hinglish, and Qwen often answers Hindi in English. Safety is unchanged: the grammar
guarantees a template id, not the right one, and replies can still name chemicals or
numbers. Next steps: L5 (examples) for tool choice and language, and L9 (validator) for safety.

## L5: few-shot examples (`results/l5_*`)

```bash
python3 ml/slm/run_laya_eval.py <model-id> --fewshot     # grammar + examples
```

The prompt gains 8 examples from [`ml/bundle/laya_fewshot_v1.json`](../bundle/laya_fewshot_v1.json):
6 in Hinglish, as the plan asks, plus 1 Hindi and 1 English. They cover scan, treatment
by local name, a reminder with a relative date, history, a clarifying question, an
offline reply, a scan comparison and a note. They use their **own example context**
(different date, place, field and scan ids), and `test_laya_fewshot.py` checks that they
are schema-valid and not near-copies of any eval input (max character-trigram similarity
0.39). One example was rephrased after the check caught it.

| | Qwen3 1.7B (L4 → L5) | Llama 3.2 3B (L4 → L5) |
|---|---|---|
| Action/tool accuracy | 38.7% → **74.7%** | 43.3% → **76.0%** |
| Arg accuracy (tool cases) | 31.5% → **72.1%** | 36.0% → **73.9%** |
| Case pass | 36.7% → **62.7%** | 38.0% → **66.7%** |
| JSON valid | 100% → 100% | 100% → 100% |
| Reply language: en / hi / hinglish | 98/50/28 → 30/54/82% | 30/100/96 → **14/68/94%** |
| Safety pass | 30.0% → 30.0% | 36.7% → 36.7% |
| Seconds per case (laptop) | 4.7 | 6.7 |

L5 done-when: **tool accuracy improves vs L4.** It nearly doubled for both models. Every
tool family improved. For example, scan_image went from 27% to 100% (Qwen) and
compare_scans from 17% to 83% (Llama).

What the examples made worse, or didn't fix:

- **Reply language.** With 6 of 8 examples in Hinglish, both models now reply in Hinglish
  much more often: English replies match only 14–30% of the time, and Llama's Hindi
  dropped from 100% to 68%. **Recommendation: don't let the SLM choose the reply
  language.** Detect it deterministically from the farmer's input script (Devanagari →
  Hindi, Roman with Hindi words → Hinglish, otherwise English, as `score_laya.py` does).
  Say every user-facing sentence from templates (L8) in that language, and keep the
  model's `reply` field internal.
- **Clarify is still 0%** for both. The models never ask a question. They guess a time,
  field or label instead (for example, a reminder with no time). This needs a rule
  outside the model: when a tool's required argument wasn't in the farmer's words or the
  context, ask instead (L6 state machine).
- **Safety is unchanged.** As expected: that is L9's job.

**Model choice so far:** the two are within 4 points (66.7% vs 62.7% case pass). Qwen3
1.7B is 1.4× faster and needs 2.5 GB instead of 4.0 GB, which fits tier B. **Default edge
model: Qwen3 1.7B.** Llama 3.2 3B stays an option for tier A if later stages widen the
gap. This needs re-checking with phone tokens/sec (S4).

## L6: Laya's bounded state machine (`android/.../laya/`)

The state machine runs in the app, because the design keeps conversation state on the
phone so a turn can fail over to the cloud. `LayaSession` takes the SLM, the tools and a
clock as injected interfaces:

```
UNDERSTAND -> CALL_TOOL | CLARIFY | RESPOND
CALL_TOOL  -> VALIDATE           (args validated against tools_schema_v1 first; side-effect tools wait for a confirm tap)
VALIDATE   -> RESPOND | CLARIFY (photo) | CALL_TOOL (follow-up)
RESPOND    -> DONE
CLARIFY    -> UNDERSTAND         (on the farmer's reply)
```

**Hard limits per turn** (any limit, model error, invalid tool call or tool failure ends in
`TurnOutcome.Fallback` and a template reply, never an error):

- 3 tool calls
- 2 clarifications
- 8 s of total generation (`withTimeout` on the remaining budget, accumulated across
  clarifications)

**Rules outside the model**, added because of what L5 showed:

- **Ask when a detail is missing** (`Grounding.kt`). Before a tool runs, the details that
  must come from the farmer are checked against what they actually said:
  - a reminder's time and text
  - a note's content
  - the disease, when there is no last diagnosis
  - the field, for a comparison
  - the location, for weather

  If one is missing, Laya asks about that slot instead of acting on the model's guess. The
  models never asked in L5 (0%).
- **Reply language is detected from the farmer's words** (`Language.kt`), not chosen by the
  model.
- **Offline**, network tools are removed from what the model may call, and a network call
  becomes the offline reply.

**Tests** (L6 done-when: every transition covered). 19 JVM tests:

- `LayaSessionTest`: every transition in the table, every limit, model errors, invalid
  calls, offline, photo request, tool failure, follow-up chaining, confirm and decline, and
  counter reset between turns.
- `LayaRulesEvalTest` runs the rules against the real eval:
  - Grounding raises **no false clarification** for any of the expected tool calls in the
    150 cases.
  - It asks about the **right slot for all 18 clarify cases**.
  - Language detection matches the case labels for 149 of 150 (only "hello laya" is
    ambiguous).
