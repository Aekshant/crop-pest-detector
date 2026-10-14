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
