# Vision eval sets

## `field/`: field-condition test set (E1)

These are real-world tomato leaf photos: cluttered backgrounds, mixed lighting, hands in
frame. Lab sets like PlantVillage overstate accuracy, and this set is meant to measure
the drop farmers will actually see. **Never train on it.**

`field/labels.csv` columns:

| Column         | Meaning |
|----------------|---------|
| `file`         | Image file in `field/` |
| `label`        | Class name. Scored through [`ml/labels_v1.json`](../../ml/labels_v1.json): `leaf_mold` is out of scope and maps to `unknown`, so those 12 photos must be rejected. |
| `source`       | `plantdoc`, `inaturalist` or `own` |
| `source_class` | Original PlantDoc folder (empty for `own`) |
| `source_split` | PlantDoc split the image came from (empty for `own`) |
| `source_path`  | Path inside PlantDoc, iNaturalist observation URL, or a short note for `own` |
| `sha256`       | Content hash, used for dedup and the training exclusion list |
| `license`      | Image license (with photographer attribution for web images) |
| `verified`     | `1` once an agronomist or someone who knows tomato diseases has confirmed the label |

### Current contents

144 images in 12 classes, 12 per class.

**Diseases and healthy (96, PlantDoc).** These come from
[PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) at commit `5467f60`, licensed
CC BY 4.0 (Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection*,
CODS-COMAD 2020). Classes: healthy, early_blight, late_blight, septoria_leaf_spot,
bacterial_spot, leaf_mold, mosaic_virus, yellow_leaf_curl_virus.

- All usable images from PlantDoc's **test** split are included (57). The rest come from
  the **train** split (39). Those train images are listed in
  `field_exclude_from_training.txt`, and `ml/` must skip them when training on PlantDoc.
- PlantDoc is scraped from the web and is noisy. Every image was screened by eye, and 41
  were rejected (`field_rejects.csv`, with reasons): slides, collages, white-background
  stock photos, non-tomato plants, and clear mislabels such as powdery mildew filed as
  leaf mold or canker filed as bacterial spot.

**Pests (48, iNaturalist).** Classes: tuta_absoluta, whitefly, fruit_borer (Helicoverpa
armigera larvae) and spider_mites. These are the main tomato pests in India and are
provisional until `labels_v1.json` (V2). The 48 images were chosen by hand from 144
candidate observations and are listed in `field_web.csv`. They show damage or pests the way
a farmer would photograph them: leaf mines, bored fruit, whitefly on leaf undersides, and
mite webbing. Microscope shots, adult moths, and larvae on hands, flowers or corn were left
out.

- **Licenses vary.** Most images are CC BY-NC, and some are CC BY-NC-SA, CC BY, CC BY-SA or
  CC0. Each row has its license and photographer. NC images are fine for evaluation, but
  they must not be bundled in a commercial app or used to train a commercial model without
  checking the license.
- **Host plant isn't confirmed for every image.** Some whitefly and spider-mite photos may
  be on plants other than tomato. The pest looks the same, but the leaf doesn't.

The screening caught non-photos and obvious mistakes. It is **not** a pathologist's check,
so `verified` is `0` everywhere.

### Known gaps

- **No own photos yet.** Real phone shots from Indian fields are still the closest match to
  production and should replace web images where possible.
- Dedup is exact-hash only. PlantDoc has some resized copies across splits that this
  doesn't catch.

### Adding own photos

1. Copy the photo into `field/` as `own_<label>_<short-id>.jpg`. Leave EXIF stripped and
   take it at normal phone distance.
2. Add a row with `source=own` and `license=own`. Fill in `sha256` (`sha256sum <file>`),
   and put place and date in `source_path`.
3. Rows that aren't from PlantDoc are kept when either script is re-run.

### Rebuilding

```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git /tmp/pd
git -C /tmp/pd checkout 5467f6012d78d1c446145d5f582da6096f852ae8
python evals/vision/build_field_set.py --plantdoc /tmp/pd
python evals/vision/fetch_web.py   # web images from field_web.csv
```

Selection is deterministic: test split first, then train images in hash order, skipping
anything in `field_rejects.csv`. To reject an image, add its hash there and re-run.

## `ood/`: out-of-domain set (E2)

These are 30 images the app must **not** diagnose. They test that it says "not recognised"
instead of guessing a tomato label.

| Category     | Count | Expected | What |
|--------------|-------|----------|------|
| `other_crop` | 14    | `unknown` | Leaves of potato, brinjal, chilli (2 each; tomato's own family, so the hardest cases), okra (2), cotton, rice, maize, cucumber, cabbage, groundnut |
| `soil`       | 5     | `unknown` | Bare or ploughed soil, close-ups and a soil profile |
| `hand`       | 5     | `unknown` | Open palms, and a hand holding soil |
| `blurry`     | 6     | `retake`  | Tomato plants blurred on purpose: 3 Gaussian, 3 motion (camera shake) |

`expected` tells the scorer what passes. `unknown` means the model rejected the image or
gave it the `unknown` class. `retake` means the quality gate caught it before any model
ran. Getting `unknown` for a blurry image is acceptable, but a disease label is a failure
in every category.

`ood/labels.csv` columns are `file`, `category`, `subject`, `expected`, `source_path` (the
source page), `transform` (the blur applied, if any), `sha256` and `license` (with
photographer attribution).

**Sources.** Crop and tomato photos come from iNaturalist and soil and hand photos from
Wikimedia Commons, all hand-picked. Every row has its license: a mix of CC BY-NC, CC BY,
CC BY-SA, CC0 and public domain (the USDA NRCS soil photos). The blur sources are
iNaturalist tomato plants that aren't in `field/`.

**Rebuilding.** `ood_sources.csv` lists each source URL and its transform:

```bash
pip install pillow
python evals/vision/build_ood_set.py
```

**Gaps.** Real blurry phone shots are better than synthetic ones, and the quality gate's
blur threshold should be checked against both. There are also no indoor scenes, screens,
or tomato fruit on a plate yet.

## Scoring (E5)

```bash
python evals/score_vision.py preds.jsonl                          # report
python evals/score_vision.py preds.jsonl --baseline current.jsonl # + better/worse verdict
```

`preds.jsonl` has one line for each image in `field/` and `ood/`, keyed by its path
relative to this folder:

```json
{"file": "field/plantdoc_early_blight_0a1b2c3d4e5f.jpg", "probs": {"early_blight": 0.81, "unknown": 0.02, "...": 0.17}, "quality": "ok"}
```

`probs` are calibrated probabilities. Raw `logits` can be given instead, together with
`--temperature T`. `quality` is the quality-gate result, either `ok` or `retake`. An image
counts as **rejected** when the gate says retake, the top label is `unknown`, or the top
probability is below `--reject`. `--reject` and `--accept` default to 0.5 and 0.8; use the
values from the model manifest.

| Metric | Meaning |
|---|---|
| accuracy | Field images whose top-1 label is right and not rejected |
| top-3 | Truth is in the top 3 labels |
| false reject rate | Good field photos the app would turn away |
| accuracy when answered | Accuracy on the images it didn't reject |
| confident answers | Share of images with prob ≥ `--accept`, and how often those are right. This is what a farmer sees labelled "confident". |
| per-class recall, worst class | Catches a model that gives up on one disease |
| accuracy by source | PlantDoc vs iNaturalist. Watch for domain shift. |
| calibration (ECE) | Expected calibration error, 15 bins, over top-1 probability |
| OOD rejected | Overall and per category, with a list of every wrongly accepted image |

With `--baseline`, each headline metric is compared, and the run **exits 1** if any
metric is worse beyond its tolerance (`REGRESSION` in the script: 2 points for accuracy,
5 for the worst class, 3 for OOD rejection, and so on). Missing predictions exit 2.
`--json out.json` saves the full report.

Tests: `python -m unittest discover -s evals/tests`. CI runs them in the `evals` job.
