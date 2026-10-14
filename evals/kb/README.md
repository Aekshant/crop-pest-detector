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

The pack is `kb_tomato_v1`: 296 chunks, embedded with
`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Full output is in
`results_paraphrase-multilingual-MiniLM-L12-v2.json`.

| Mode (top-3 hit rate) | all 40 | en | hi | hinglish |
|---|---|---|---|---|
| first run: `open` (labels broken, see below) | 38% | 50% | 50% | 8% |
| first run: `filtered` | 70% | 79% | 79% | 50% |
| `open` | 38% | 50% | 50% | 8% |
| `filtered` | 85% | 93% | 86% | 75% |
| `open+names` (no diagnosis yet) | **57%** | 57% | 71% | 42% |
| `filtered+names` (diagnosis known) | **88%** | 93% | 86% | 83% |

**What the eval found:**

- **A labelling bug.** In the NIPHM PDF, the last pest heading carried over to every
  paragraph after it. Weed descriptions and rodent control were tagged `mosaic_virus` (35
  chunks), which is why mosaic scored 0% even with the filter. A label now applies to at
  most 3 paragraphs after its heading, and any other heading ends it. This was a bug fix,
  not tuning to the queries.
- **A content gap.** 2 of the 40 queries have no relevant chunk at all: "how do I stop
  late blight from spreading" (en and hi). Every late-blight management sentence in the
  sources was a fungicide dose, so all of them were removed (correctly). Non-chemical
  late-blight practice needs another source, for example a state agricultural university
  package of practices.
- **Hinglish is the weak spot.** The embedding model doesn't know romanized Hindi. Name
  expansion lifts Hinglish from 8% to 42% (open) and from 75% to 83% (filtered). Most of
  the remaining Hinglish misses use everyday words ("dhabbe", "failta") rather than
  disease names.

**Caveat:** the queries were seen while fixing the labelling bug, so these numbers are
not held out. The relevance rule and the name expansion were not tuned on them.

**Next steps:**

- Try `intfloat/multilingual-e5-small`: same size, usually stronger in Hindi, but it
  needs a custom ONNX export.
- Add a BM25 keyword index (sqlite FTS5, already in the pack's sqlite) and combine the
  two.
- Add Hindi translations of the TNAU chunks.
