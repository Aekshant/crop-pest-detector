# Training data (V1)

Tomato images for training the vision model. They come from a lab dataset (PlantVillage),
a field dataset (PlantDoc) and a pest dataset (IP102). Mixing lab and field photos is
the main defence against domain shift.

The images are **not in git**. `prepare.py` downloads them to `_downloads/` (gitignored)
and writes [`manifest.csv`](manifest.csv), one row per usable image.

```bash
pip install pillow
python ml/data/prepare.py            # fetch everything (resumable), then build the manifest
python ml/data/prepare.py --no-fetch # rebuild the manifest only
```

The download is about 3.6 GB, almost all of it IP102. Only one IP102 class is kept; the
rest of the archive can be deleted after extraction.

## Manifest

| Column | Meaning |
|---|---|
| `path` | Image path relative to `ml/data/` |
| `label` | FarmLens label (provisional until `labels_v1.json`, V2) |
| `source` | `plantvillage`, `plantdoc` or `ip102` |
| `source_label` | Original class name |
| `split` | Source split: `train`/`test` (PlantVillage, grouped by leaf), `train` (PlantDoc), `train`/`val`/`test` (IP102) |
| `sha256` | Content hash |
| `license` | License of the source dataset |

### Current counts

| Label | PlantVillage (lab) | PlantDoc (field) | iNaturalist (field) | Commons | Total |
|---|---|---|---|---|---|
| unknown | 4036 (leaf mold, target spot, 28 other-crop classes × 60) | 73 (leaf mold) | 879 (other crops, weeds) | 125 (soil, hands) | 5113 |
| healthy | 1585 | 39 | — | — | 1624 |
| early_blight | 1000 | 73 | — | — | 1073 |
| late_blight | 1901 | 97 | — | — | 1998 |
| septoria_leaf_spot | 1771 | 135 | — | — | 1906 |
| bacterial_spot | 2127 | 89 | — | — | 2216 |
| yellow_leaf_curl_virus | 5357 | 59 | — | — | 5416 |
| mosaic_virus | 373 | 36 | — | — | 409 |
| tuta_absoluta | — | — | 73 | — | 73 |
| whitefly | — | — | 299 | — | 299 |
| fruit_borer | — | — | 300 | — | 300 |
| spider_mites | 1676 | — | 294 | — | 1970 |
| **total** | | | | | **22397** |

Web photos come from `fetch_web_train.py`, which is run by `prepare.py` or on its own. API
answers are cached, so it is resumable. IP102 is optional; its download stalls on Google
Drive quota. Dropped while building: 41 exact copies of eval images, 5 near-duplicates of
eval images, 30 rejected or eval-used PlantDoc images, and 16 duplicates within the sources.

## Kept out of training

Every image in the vision eval sets (`evals/vision/field`, `evals/vision/ood`) is kept out:

- PlantDoc's whole **test** split, because it supplies most of the field eval set.
- PlantDoc **train** images that went into the field set
  (`evals/vision/field_exclude_from_training.txt`) or were rejected during screening
  (`evals/vision/field_rejects.csv`: slides, collages, wrong crop, wrong label).
- Any image whose 64-bit dHash is within 6 bits of an eval image, which catches resized
  or re-encoded copies, as well as exact duplicates.

## Extra field datasets (opt-in)

`fetch_field_datasets.py` downloads four more field sets. `prepare.py` adds them to the
manifest, but `train.py` leaves them out by default (see `ml/README.md`, experiments
v3c/v3d).

| Source | What | License |
|---|---|---|
| `pakistan` | [Mendeley 3mbnb82mxd](https://data.mendeley.com/datasets/3mbnb82mxd): in-field photos. Raw set only; the augmented copies would leak across splits. **About a third are copies of PlantDoc images**, and `prepare.py` drops them (27 were copies of eval photos). | CC BY 4.0 |
| `bangladesh` | [Mendeley jttrv2w27r](https://data.mendeley.com/datasets/jttrv2w27r): leaves on paper, including 117 bacterial spot | CC BY 4.0 |
| `fieldfruit` | [Mendeley swtt5wy2pr](https://data.mendeley.com/datasets/swtt5wy2pr): leaves and fruit on paper (resized set) | CC BY 4.0 |
| `tomatovillage` | [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village): leaves from Rajasthan fields on paper. The leaf-miner class is not used (it shows Liriomyza, not Tuta). | **No license stated.** Ask the authors before any commercial use. |

## Licenses

Web photos (iNaturalist, Commons) carry a per-row license and attribution in the
manifest. **Most iNaturalist photos are CC BY-NC.** That is fine for a hackathon or
research model, but filter `license` to CC0, CC BY and CC BY-SA before training anything
commercial.


| Source | License | Use |
|---|---|---|
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (commit `7f7ecc7`; splits from [Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)) | **CC BY-SA 3.0** | Training is fine. Credit Hughes & Salathé (2015) and Mohanty et al. (2016). ShareAlike applies to redistributed images and derived datasets; whether it reaches trained weights is legally unsettled, so keep the attribution in the model card. |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (commit `5467f60`) | **CC BY 4.0** | Training is fine. Credit Singh et al., CODS-COMAD 2020. The images were scraped from the web, so individual photos may carry their own rights. |
| [IP102](https://github.com/xpwu95/IP102) v1.1 | **Academic use only.** "For other purposes, please contact Xiaoping Wu (xpwu95@163.com)." | ⚠️ A hackathon entry or product is not clearly academic. Either get the author's permission, or train without IP102 (drop rows with `source=ip102`). It contributes whitefly images only. |

## Gaps

- **Pests are thin.** PlantVillage has spider mites (lab photos). IP102 has only greenhouse
  whitefly; it has no Tuta absoluta and no Helicoverpa armigera, because its pests are
  rice, wheat, maize, citrus and mango pests. `tuta_absoluta` and `fruit_borer` have **no
  training images yet**. Candidates: CC-licensed iNaturalist observations (as used for the
  eval set, minus those images), or own photos.
- **target_spot** is in PlantVillage but not in the v1 label set. V2 decides whether to
  add it, merge it, or drop it.
- PlantVillage dominates the counts and is all lab photos on plain backgrounds. Training
  should rebalance by source, not just by class.
