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
  registered for tomato whitefly and was removed from the lexicon.

**Before release:** an agronomist should review these rows. Choosing which of the 40
early-blight products to show (for example, by FRAC group rotation, availability or
price) is a product decision. The responder currently shows the first row.
