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
uvx --with onnx --with onnxruntime==1.31.0 --with huggingface_hub python kb/export_embedder.py   # the on-device model
uvx --with fastembed --with sqlite-vec --with pdfplumber --with onnxruntime==1.31.0 --with tokenizers python kb/build_pack.py
```

`kb/dist/` holds the build outputs and is gitignored:

| File | Size | What |
|---|---|---|
| `kb_tomato_v1.sqlite` | 2.1 MB | the pack |
| `e5_small_int8.onnx` | 113 MB | multilingual-e5-small, dynamic int8 (448 MB at fp32) |
| `e5_small_vocab.tsv` | 7.8 MB | its 250k-piece vocabulary, for the phone's tokenizer |

A single sqlite file is what the bundle ships (design doc: read-only, replaced by KB pack
updates). It contains:

| Table | Contents |
|---|---|
| `meta` | `kb_version`, `embedding_model` (`e5-small-int8`), `embedding_dim` (384), `built_at` |
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

**Embedding:** `kb/embedding.py` runs the same int8 ONNX file the phone runs:

1. tokenize;
2. mean-pool over tokens;
3. L2-normalize;
4. prefix `passage: ` for chunks and `query: ` for questions.

**On the phone** (`android/.../kb/`):

- **`UnigramTokenizer`:** a Kotlin port of the XLM-R SentencePiece Unigram tokenizer (NFKC,
  Metaspace, Viterbi). It gives the same ids as the reference on all 291 test texts.
- **`EmbeddingModel`:** runs the model on ONNX Runtime 1.31.0 (the same version as the
  Python build). Its vectors match the reference with cosine ≥ 0.9999999.
- **`KbRetriever`:** hybrid search, BM25 plus vectors fused by RRF, with local-name
  expansion and an optional label filter. It ranks the same as the Python eval on all 40
  queries.
- **`KbEndToEndTest`** runs question → tokenizer → model → retriever and reproduces the K4
  hit rates exactly: 32/40 without a diagnosis and 35/40 with one. A search takes about
  14 ms on a desktop JVM.

These tests skip when `kb/dist/` hasn't been built, which is the case in CI. The ONNX
Runtime Android library adds about 34 MB (`arm64-v8a`, uncompressed) to the app. An
arm64-only split or ORT's reduced mobile build would cut that.

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
