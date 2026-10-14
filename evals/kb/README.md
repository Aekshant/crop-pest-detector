# KB retrieval eval (K4)

```bash
uvx --with fastembed --with sqlite-vec python kb/build_pack.py          # build the pack first
uvx --with fastembed --with sqlite-vec python evals/kb/eval_retrieval.py
```

## Queries and scoring

[`retrieval_queries.jsonl`](retrieval_queries.jsonl) has **40 farmer questions**: 14
English, 14 Hindi and 12 Hinglish. They cover symptoms, weather and spread, non-chemical
control, traps, natural enemies and roguing for all 10 diseases and pests. The same
questions recur across the three languages, so language gaps show up directly. There are
no dose questions, because doses come from `kb_treatment`, not retrieval.

Each query has a relevance rule written together with the queries, before any run. A
retrieved chunk is relevant if it is tagged with the expected label **and** contains one
of the expected words. A query **hits** if any of the top 3 chunks is relevant.

Modes:

- `open` searches all chunks.
- `filtered` searches only chunks of the expected label. The app usually knows the last
  diagnosis, so it can filter this way.
- `+names` appends the English name when the question uses a local or Hindi name from
  `ml/labels_v1.json` ("agati jhulsa" → "Early blight"). Laya's grounding already knows
  these names.

## Results

```bash
uvx --with fastembed --with sqlite-vec --with onnxruntime==1.31.0 --with tokenizers python evals/kb/eval_retrieval.py [--pack ...]
```

Mode names:

- `hybrid` fuses the vector ranking with a BM25 keyword ranking by reciprocal rank
  fusion (k = 60).
- The app uses **`hybrid+names`**, and **`hybrid_filtered+names`** when the diagnosis is
  known. `KbEndToEndTest` reproduces both numbers on the phone's code path.

Top-3 hit rate over all 40 queries. Two of them, late blight "stop it spreading" in
English and Hindi, have no chunk that passes the relevance rule (see below).

| Pack / model | open | filtered | open+names | filtered+names | **hybrid+names** | **hybrid_filtered+names** |
|---|---|---|---|---|---|---|
| v1: MiniLM-L12, paragraph chunks with the label carry-over bug | 38% | 70% | | | | |
| v1, label bug fixed | 38% | 85% | 57% | 88% | | |
| v2 chunks (NIPHM structure), MiniLM-L12 | 45% | 80% | 65% | 90% | 78% | 88% |
| v2, multilingual-e5-small fp32 | 42% | 92% | 80% | 90% | 82% | 88% |
| **v2, e5-small int8 (shipped)** | 45% | 95% | 75% | 90% | **80%** | **88%** |

By language for the shipped configuration:

| Mode | en | hi | hinglish |
|---|---|---|---|
| hybrid+names | 86% | 86% | 67% |
| hybrid_filtered+names | 86% | 93% | 83% |

**What changed and why:**

- **Label carry-over bug.** In the NIPHM PDF, the last pest heading carried over to every
  paragraph after it. Weed and rodent text was tagged `mosaic_virus`, so mosaic scored 0%
  even with the filter. The fix was to cap the carry-over.
- **Structure-aware chunking (v2).** The NIPHM package is now chunked by its own
  structure: table rows, numbered description sections, and in-scope general sections.
  It is about 100 chunks smaller and has no rodent, weed or calibration text.
- **Hybrid search.** BM25 helps most without a diagnosis: MiniLM goes from 65% to 78%,
  and Hinglish from 50% to 75%. With a filter it changes about one query.
- **e5-small.** It is the same size as MiniLM-L12 (118M parameters, 384 dimensions) and
  better without a diagnosis (80% vs 65% for open+names). Quantizing it to int8 (113 MB)
  costs about one query.
- **Local names.** "agati jhulsa" gets "(Early blight)" appended, using
  `ml/labels_v1.json`.

**Known gaps:**

- **Late-blight prevention:** the only relevant chunk ("Field sanitation. Destruction of
  alternate hosts…") doesn't contain the rule's words ("destroy" doesn't match
  "Destruction"). I left the rule as written, so these 2 queries count as misses.
- **Hinglish without a diagnosis (67%)** is the weakest cell. The misses use everyday
  words ("dhabbe", "failta").

**Caveat:** the chunking fixes, the model choice (2 candidates) and hybrid search were all
decided on these 40 queries, so the numbers are optimistic. A held-out set of real farmer
questions is needed before release.

## Held-out check: real Kisan Call Centre questions

```bash
uvx --with pyarrow python -I evals/kb/make_heldout_kcc.py <kcc.parquet>   # once; writes heldout_kcc.jsonl
uvx --with fastembed --with sqlite-vec --with onnxruntime==1.31.0 --with tokenizers python evals/kb/eval_retrieval.py --queries evals/kb/heldout_kcc.jsonl
```

[`heldout_kcc.jsonl`](heldout_kcc.jsonl) has **60 real farmer questions**: 6 per label,
from 13 states, 2013–2023. They are tomato "Plant Protection" queries from the Kisan Call
Centre logs (Department of Agriculture & Farmers Welfare, Open Government Data platform),
read from the Hugging Face mirror `Omegaindebt/Kisan_Call_Centre_Transcripts`.

`make_heldout_kcc.py` fixed every rule before the first run:

- label keyword patterns;
- deduplication;
- the random seed;
- relevance: the chunk carries the label, and for "control / management" questions it
  must also contain management words.

The set was run **once**, after all the changes above, and nothing was tuned on it.

| Mode | open | filtered | open+names | filtered+names | hybrid+names (shipped, no diagnosis) | hybrid_filtered+names (shipped, diagnosis known) |
|---|---|---|---|---|---|---|
| e5-small int8 | 98% | 100% | 98% | 100% | **88%** | **98%** |

**Reading it:**

- **These questions are easier than the dev set.** The advisors type them in English and
  they name the pest ("TOMATO WHITE FLY MANAGEMENT"). Hindi and Hinglish speech is not
  covered here.
- **Without a diagnosis, hybrid search is worse than plain vector search here** (88% vs
  98%). BM25 latches onto generic words ("tomato", "control", "management") that are
  common in general chunks: mites and Tuta drop to 50%. On the dev set the opposite held
  (hybrid 80% vs vector 75%), because BM25 plus name expansion helps Hindi and Hinglish.
- **Not tuned.** Changing the fusion now would spend this held-out set. The next step is
  to try down-weighting BM25 or adding a stop-word list, and judge it on a **new** held-out
  sample: same script, new seed, excluding these 60.
