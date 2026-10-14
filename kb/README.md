# kb

Knowledge base content used by the app and server. Laya's treatment answers come only
from here: the products, doses and waiting periods in KB rows (L9), never from the model.

## `sources.csv`: where the KB comes from (K1)

There are 44 items, one per label and source, for all 10 diseases and pests in
`ml/labels_v1.json`. `unknown` and `healthy` need no treatment sources. Every label has
three kinds of source:

| Kind | Source | Used for |
|---|---|---|
| `advisory` | [TNAU Agritech Portal](https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_tomato.html) page for that disease or pest; ICAR-NBAIR note for Tuta | symptoms and management practices |
| `ipm_package` | [AESA based IPM package: Tomato](https://www.niphm.gov.in/IPMPackages/Tomato.pdf) (DPPQS / NIPHM / ICAR-NCIPM, 2014) and the [IPM Schedule for Vegetables](https://agritech.tnau.ac.in/horticulture/pdf/tech_bulletin/national/IPM-Schedule-for-vegetables.pdf) (NHM, 2012) | IPM practices, scouting, cultural and biological control |
| `registration_list` | CIB&RC [Major Uses of Pesticides](https://ppqs.gov.in/divisions/cib-rc/major-uses-of-pesticides), fungicides, insecticides and bio-pesticides, **as on 30.09.2026** | which products are registered for tomato and that problem, the dose and the waiting period |

Columns: `id, label, kind, topic, publisher, title, url, doc_date, accessed, sha256, notes`.
`sha256` is the hash of the file as fetched on `accessed`.

**Rules for K2 (treatment rows):**

- Doses and waiting periods come **only** from the CIB&RC lists, which are current. The
  IPM packages are 12–14 years old: use them for practices, not for doses.
- `registration_status` is `registered` only if the CIB&RC list has that product for
  **tomato** and **that problem**.
- **Viruses (leaf curl, mosaic):** no product is registered against the virus itself.
  Answers are vector (whitefly) control and removing infected plants, never a "virus
  medicine".
- **Bacterial spot:** has very few tomato registrations, so many answers will be the
  consult-KVK template.

```bash
python kb/fetch_sources.py            # download all 17 documents to kb/_cache/ (gitignored), report changes
python kb/fetch_sources.py --strict   # exit 1 if a URL fails or a document changed
python -m unittest discover -s kb -p "test_*.py"   # offline checks (CI)
```

The documents are not committed: they are publishers' files and add up to about 15 MB.
`fetch_sources.py` gets the same versions back and flags any update.

**Gaps:**

- No tomato-specific bulletin from ICAR-IIHR or ICAR-IIVR is published online as a
  downloadable file. ICAR is represented through NCIPM (co-author of the NIPHM package)
  and NBAIR. Requesting the institutes' tomato extension folders would strengthen the
  Hindi content.
- Only TNAU (Tamil Nadu) is covered among the state agricultural universities. Adding
  MPKV (Maharashtra) or PAU (Punjab) would cover regional practices.

## `kb_treatment.csv`: treatment rows (K2)

```bash
python kb/fetch_sources.py
uvx --with pdfplumber python kb/build_kb.py
```

`build_kb.py` reads the table cells of the four CIB&RC lists and keeps every **tomato** row:

- [`cibrc_tomato.csv`](cibrc_tomato.csv) has all 198 rows, as printed (the audit trail).
- [`kb_treatment.csv`](kb_treatment.csv) has **129 rows**, one per (label, product), with
  the design doc's columns: `label, product, dose, phi_days, registration_status,
  source_url (with #page=), kb_version`. Each row also keeps `pest_as_listed`.

| Label | Products | | Label | Products |
|---|---|---|---|---|
| early_blight | 40 | | whitefly | 16 |
| late_blight | 28 | | fruit_borer | 31 |
| septoria_leaf_spot | 3 | | spider_mites | 9 |
| bacterial_spot | 1 | | tuta_absoluta | 1 |
| yellow_leaf_curl_virus | 0 (vector control) | | mosaic_virus | 0 (roguing, vector control) |

**Rules (from L9: numbers are never generated):**

- **Doses are copied, not converted.** A row reads, for example, "500 ml per hectare in
  500 litres of water". There is no per-litre or per-pump arithmetic. The only inferred
  part is the unit for a bare number: ml for liquid formulations (EC, SC, SL, OD…), g for
  solid ones (WP, WG, SP…).
- **Waiting period.** A range such as "3-5" keeps its longest printed value. A row with
  "-" (none printed) is not shown as a treatment.
- **Skipped (46).** Rows were dropped when Laya couldn't state them safely in the spray
  template:
  - doses on another basis: per 1,000 seedlings, per kg of seed, nursery, soil granules;
  - concentrations ("0.2%", "0.8 ml/L");
  - a per-hectare dose under 25 g or ml, which is a per-litre figure in the wrong column
    ("3 ml");
  - a spray volume under 200 L per hectare.
- **Pest mapping.** Text like "Early & Late blight" maps to both labels. Serpentine leaf
  miner (*Liriomyza*) is **not** mapped to `tuta_absoluta`; only rows naming Tuta or
  pinworm are.
- **Registration.** `registered` means the product appears in a CIB&RC list for tomato
  against that problem. Nothing else can make a row registered.

`test_kb_treatment.py` checks that:

- every number in a row is printed in its source row;
- the viruses have no products;
- leaf miner rows are not Tuta;
- the L9 safety lexicon names no registered product. This found imidacloprid, which is
  registered for tomato whitefly and was removed from the lexicon;
- every active ingredient in the KB is in the lexicon's `known_actives`. `build_kb.py`
  writes this list (67 names), so the L9 validator also blocks a real product named in a
  turn whose KB row is a different product.

**Before release:** an agronomist should review these rows. Choosing which of the 40
early-blight products to show (for example, by FRAC group rotation, availability or
price) is a product decision. The responder currently shows the first row.

## KB pack (K3): `kb/dist/kb_tomato_v1.sqlite`

```bash
python kb/fetch_sources.py
uvx --with pdfplumber python kb/build_kb.py      # K2 rows + known_actives (only when the lists change)
uvx --python 3.11 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with "torch==2.4.1+cpu" --with tensorflow-cpu==2.17.0 --with tf-keras==2.17.0 --with transformers==4.44.2 python kb/export_litert.py   # the on-device model (once)
uvx --python 3.11 --with fastembed --with sqlite-vec --with pdfplumber --with tokenizers --with ai-edge-litert python kb/build_pack.py
```

`kb/dist/` holds the build outputs and is gitignored:

| File | Size | What |
|---|---|---|
| `kb_tomato_v1.sqlite` | 2.1 MB | the pack |
| `e5_small_encoder.tflite` | 21.4 MB | multilingual-e5-small's network on LiteRT: word vectors in, mean-pooled unit vector out; int8 weights |
| `e5_small_table.bin` | 92.5 MB | its 250k × 384 word table, int8 with a scale per row (memory-mapped by the app) |
| `e5_small_vocab.tsv` | 7.4 MB | its vocabulary, for the phone's tokenizer |

A single sqlite file is what the bundle ships (design doc: read-only, replaced by KB pack
updates). It contains:

| Table | Contents |
|---|---|
| `meta` | `kb_version`, `embedding_model` (`e5-small-litert`), `embedding_dim` (384), `built_at` |
| `kb_source` | `sources.csv` |
| `kb_treatment` | the 129 K2 rows, indexed by label: the only place a product, dose or waiting period lives |
| `kb_chunk` | 197 advisory chunks (`label`, `text`, `source_id`, `source_url`) plus `embedding`, a float32 unit vector |
| `vec_chunk` | the same vectors in a sqlite-vec `vec0` table (cosine distance), for the server and evals |

The phone doesn't use `vec_chunk`. Android's SQLite can't load extensions, so `KbPack.kt`
reads `kb_chunk.embedding`, and `KbRetriever.kt` searches in memory (brute force over
about 200 × 384 floats).

**Chunks** are built from the structure of each source:

- **TNAU pages:** one chunk per section (symptoms, identification, management,
  favourable conditions), prefixed with the pest name.
- **NIPHM/NCIPM tomato IPM package:**
  - its tables become one chunk per pest row: natural enemies, resistant varieties, and
    crop-stage IPM ("Alternaria blight and late blight (tomato, vegetative stage):
    Cultural control. Field sanitation…");
  - its "Description of insect pests / diseases" sections are split at their numbered
    headings;
  - general IPM and pesticide-safety sections are kept as general chunks;
  - weeds, rodents, calibration, references and front matter are left out as outside
    Laya's scope.
- **ICAR-NBAIR Tuta note:** chunked as paragraphs, labelled Tuta.

**No doses in chunks:** every sentence with a dose ("@", "200 ml", "0.2%", "per acre") is
dropped (89 sentences). Doses come from `kb_treatment` only.

**Embedding:** e5-small is about 96M parameters of word table plus a 21M-parameter
network, so `kb/export_litert.py` ships it in two parts:

- **the word table**, as int8 with a scale per row (max error 0.005), which the app
  memory-maps and reads directly;
- **the network**, converted to LiteRT with int8 weights. Mean pooling and L2
  normalisation are inside the model.

`kb/embedding.py` (`e5-small-litert`) runs exactly what the phone runs: tokenize, look up
the table, run the encoder, with `passage: ` / `query: ` prefixes.

Its vectors are **closer to the original fp32 model** than the earlier ONNX Runtime int8
export, which also quantized activations:

| vs fp32, 237 KB texts | min cosine | median |
|---|---|---|
| **LiteRT (shipped)** | **0.9992** | 0.9996 |
| ONNX int8 (until 2026-10-15) | 0.9586 | 0.9883 |

**On the phone** (`android/.../kb/`):

- **`UnigramTokenizer`:** a Kotlin port of the XLM-R SentencePiece Unigram tokenizer (NFKC,
  Metaspace, Viterbi). It gives the same ids as the reference on all 291 test texts
  (`TokenizerTest`, on the JVM).
- **`LiteRtEmbeddingModel`:** the memory-mapped table plus the LiteRT interpreter the app
  already ships for vision.
- **`KbRetriever`:** hybrid search, BM25 plus vectors fused by RRF, with local-name
  expansion and an optional label filter. It ranks the same as the Python eval on all 40
  queries for fixed vectors (`KbRetrieverTest`, on the JVM).
- **`KbDeviceTest`** runs on the phone (LiteRT has no desktop JVM build). It loads the KB
  from the installed bundle through `KbRepository`. On the POCO M6 Plus:
  - embeddings match the Python reference with a worst cosine of 0.9995 (ARM vs x86
    kernels), at 21.7 ms per embedding;
  - **K4 end to end**: 32/40 without a diagnosis and 35/40 with one. Python on x86 gets
    33/40 and 35/40. The one difference is a near-tie at rank 3 (`kbq-040`, "fal chhedak
    ke liye kaunse trap lagaye?"), and the test allows at most one such flip per mode
    and prints it;
  - 25 ms per search;
  - a Laya treatment turn through `LayaSession`.
- **The release build** (R8) was checked on the phone through the Status screen's
  knowledge-base check (tap the row). It loads the KB in 828 ms and searches in 31 ms.

**Tests:** `test_pack.py` checks that:

- every table is filled;
- every label has chunks;
- no chunk contains a dose;
- a vector query returns the nearest chunk first.

## Retrieval eval (K4)

See [`evals/kb/README.md`](../evals/kb/README.md). Top-3 hit rate on 40 English, Hindi and
Hinglish questions, using the app's pipeline (hybrid search with local names, int8
e5-small):

- **88%** when the diagnosis is known (filtered by label);
- **80%** without a diagnosis.

Hinglish is the weakest language.

## Agronomist review: which rows Laya may name, and in what order

```bash
uvx --with openpyxl python kb/make_review_sheet.py            # -> kb/review/treatment_review.xlsx (send to the agronomist)
uvx --with openpyxl python kb/make_review_sheet.py --import   # the filled sheet -> kb/review_decisions.csv
uvx --with pdfplumber python kb/build_kb.py                   # applies it: review_status, display_rank
```

**The sheet.** It has one row per (label, product), with columns for decision
(`approve` / `reject`, as a dropdown), `rank` (1 = the product Laya names first),
`reviewer` and `notes`.

- Helper columns: `type` (biological or chemical), `actives`, the waiting period, and
  the CIB&RC page.
- `suggested_rank` is a transparent default only: biological first, then single active
  ingredient, then shortest waiting period. It is not advice.
- Decisions are keyed by (label, product), so they survive rebuilds when the CIB&RC
  lists are updated.

**In the app:**

- `KbRow.usable` = registered **and** approved.
- The safety responder and `get_treatment_info` name the best-ranked usable row.
- A problem with registered but unapproved rows gets "ask your KVK" (`T_NO_KB_CONSULT`).
- `KbPack.open(includeUnreviewed = true)` lets debug builds use pending rows. Rejected
  rows are never used.

Until the review is done, every row is `pending`.

## Laya's KB answers (`android/.../kb/KbAnswers.kt`)

- **`KbTools`:** the `get_treatment_info` tool. It returns the label's approved rows,
  best rank first, plus the top 3 prevention passages for the label.
- **`KbAnswers.treatment()`:** `T_KB_TREATMENT` for the best row (or the consult
  template), followed by "To prevent it:" and the best passage, quoted verbatim with its
  source ("TNAU Agritech Portal ke anusaar (English mein): …").
- **`KbAnswers.explain()`** answers "why?" and "what do I do?", following the design doc:
  1. retrieve passages for the question, filtered by label;
  2. `ExplanationModel` (the edge or cloud SLM) answers in the farmer's language from those
     passages only;
  3. the `SafetyValidator` removes numbers and product names;
  4. a language check confirms the reply is in the farmer's language.

  If the model fails, writes in the wrong language, or nothing survives the validator,
  the answer falls back to the label's template description plus an attributed passage.
- **Only clean passages are quoted.** A passage sentence with any number (for example
  "24-29 degree C") is never quoted, because the validator allows no numbers outside the
  KB rows.

`KbEndToEndTest` runs a whole turn on the real pack: "pachheti jhulsa ka ilaj batao" →
`LayaSession` → `KbTools` → a Hinglish treatment answer with a late-blight prevention
passage.

**Two validator bugs** came up on real KB data, with no effect on the fixture eval:

- A combination product ("Ametoctradin 27% + Dimethomorph 20.27% SC") was blocked,
  because "Dimethomorph" is another row's first word.
- Sentences were split at the decimal point in "20.27".

Both are fixed and covered by `SafetyTest`.

## Shipping it: model bundle and app size

```bash
python kb/update_manifest.py   # after build_pack.py / export_litert.py: writes the "kb" entry of ml/bundle/manifest_draft.json
```

The KB files travel in the **downloadable model bundle**, not in the APK. That is about
123 MB: the pack (2.1 MB), the encoder (21.4 MB), the word table (92.5 MB) and the
vocabulary (7.4 MB).

The manifest's `kb` entry records:

- `kb_version` and the embedding model;
- the query prefix and the runtime;
- the review counts;
- for each file, its size in bytes and its sha256.

On the device, `KbBundle.verify()` checks each file's size and sha256 once, after
download and before activation. `KbBundle.open()` then loads the pack, the embedder and
the retriever. The downloader is described in `ml/bundle/README.md`.

**App size:**

| Build | ABIs | APK | Download (gzip estimate) |
|---|---|---|---|
| debug | arm64-v8a + x86_64 (emulator) | 43.2 MB | n/a |
| release with ONNX Runtime (until 2026-10-15) | arm64-v8a | 54 MB | about 26 MB |
| **release** (R8 and resource shrinking, LiteRT only) | arm64-v8a only | **21.6 MB** | **about 15.8 MB** |

- Dropping ONNX Runtime removed 34 MB of native code, along with its telemetry provider.
- The embedder now runs on LiteRT, which the app already ships for vision.
- The release build is unsigned until a release keystore exists. It has been run on the
  POCO M6 Plus.
