# ml

Model training, conversion and export (e.g. to on-device format consumed by `core/`).

## Label map: [`labels_v1.json`](labels_v1.json) (V2)

There is one label map for everything: the model's output layer, the training manifest,
the vision scorer and Laya's tools. Model output index `i` is `labels[i]`.

| Index | id | Kind | Hindi |
|---|---|---|---|
| 0 | unknown | unknown | पहचान नहीं हुई |
| 1 | healthy | healthy | स्वस्थ |
| 2 | early_blight | disease | अगेती झुलसा |
| 3 | late_blight | disease | पछेती झुलसा |
| 4 | septoria_leaf_spot | disease | सेप्टोरिया पत्ती धब्बा |
| 5 | bacterial_spot | disease | जीवाणु धब्बा रोग |
| 6 | yellow_leaf_curl_virus | disease | पत्ती मरोड़ रोग |
| 7 | mosaic_virus | disease | मोज़ेक रोग |
| 8 | tuta_absoluta | pest | टूटा कीट (पिनवर्म) |
| 9 | whitefly | pest | सफ़ेद मक्खी |
| 10 | fruit_borer | pest | फल छेदक |
| 11 | spider_mites | pest | लाल मकड़ी (माइट) |

This is 6 diseases and 4 pests, matching the design doc's v1 scope of "4–6 diseases,
3–4 pests, healthy, unknown".

- **Out of scope → `unknown`:** leaf mold, which is mostly a polyhouse problem, and target
  spot. Their images train the unknown class, and the 12 leaf-mold field photos test it:
  the app should say "not recognised" rather than misdiagnose a real disease.
- **`yellow_leaf_curl_virus`** is shown as "leaf curl virus". In India the cause is mostly
  ToLCV, which looks the same as TYLCV.
- **Pests without training data yet:** tuta_absoluta, fruit_borer, and whitefly (IP102 is
  pending). They are in the map so that output indices don't change when data arrives.
- **`aliases`** are local names (Hindi and Hinglish) that Laya and the KB match on.
- **`source_map`** maps each dataset's class names to v1 ids. `ml/data/prepare.py` and
  `evals/score_vision.py` read it, and nothing else holds its own copy.

**Rules.** Within v1, never rename or reorder an id. Adding or removing a class means
`labels_v2.json` and a new model bundle. The scorer rejects any prediction label that
isn't in the map, and `evals/laya/check_cases.py` fails if Laya's label list drifts
from it.

**Gap for the unknown class.** Today it trains only on leaf mold and target spot.
Non-tomato negatives would help with OOD rejection: PlantVillage's pepper and potato
lab photos are an easy addition. They don't overlap the OOD eval set, which uses
iNaturalist field photos.
