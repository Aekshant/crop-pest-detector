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

## Training (V3)

```bash
uv venv -p 3.11 ml/.venv
uv pip install -p ml/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
uv pip install -p ml/.venv/bin/python -r ml/requirements.txt
ml/.venv/bin/python ml/train.py --out ml/runs/<run_name>
```

[`train.py`](train.py) fine-tunes **EfficientNet-Lite0** (timm `tf_efficientnet_lite0`,
ImageNet weights, 3.4M params). It was chosen over MobileNetV3 because the Lite variants
drop squeeze-excite and swish, so INT8 quantization for the tier-C CPU path loses less
accuracy. Expected INT8 size is about 4 MB, inside the 5–15 MB budget in `docs/budget.md`.

Transfer learning runs in two stages:

1. **Head:** the backbone is frozen. Features are computed once and cached, and only the
   classifier trains.
2. **Fine-tune:** the last `--unfreeze-blocks` blocks and the head train with augmentation
   (crop, flips, rotation, colour jitter, light blur). `0` unfreezes the whole network.

Sampling is balanced per class, and field photos get at least 30% of each class's
samples. They are 3% of the data, but they're what the app actually sees.

Validation uses PlantVillage's leaf-grouped test split (capped per class) and a
hash-stable 15% slice of PlantDoc. The best epoch is chosen by validation macro recall,
then a temperature is fitted on validation logits. Finally, the script predicts the vision
eval sets and runs `evals/score_vision.py` on them.

Each run writes to `ml/runs/<name>/`:

| File | Contents |
|---|---|
| `model.pt` | Best weights plus everything needed to use them: arch, input size, mean/std, `labels_v1` order, temperature, val metrics, manifest hash, git commit (Git LFS) |
| `train_log.txt`, `train_log.jsonl` | Human-readable and machine-readable logs, per epoch |
| `eval_preds.jsonl`, `eval_report.txt`, `eval_report.json` | Field and OOD eval predictions and the scorer's report |

**On a rented GPU** (Nebius or similar, any CUDA box): install torch from the CUDA index
instead of the CPU one, run `ml/data/prepare.py`, then fine-tune the whole network over
the full data:

```bash
python ml/train.py --out ml/runs/v3_lite0_gpu --unfreeze-blocks 0 --samples-per-epoch 0 \
    --epochs-ft 15 --batch 128 --workers 8 --val-max 0
```

Commit the run folder, and compare runs with
`python evals/score_vision.py new/eval_preds.jsonl --baseline old/eval_preds.jsonl`.

### Run `runs/v3_lite0_cpu` (first baseline, CPU)

This was trained on CPU with the default settings, because no GPU was available. Stage 1 ran
30 head epochs on cached features. Stage 2 fine-tuned the last 2 blocks with 4,000 samples
per epoch. The session was interrupted during fine-tune epoch 5. The best checkpoint
(epoch 4) was finished with `--finalize-only`, so the run took about 22 minutes of
training plus finalization.

| | Result |
|---|---|
| Validation (mostly PlantVillage lab photos), macro recall | **0.890** (head only: 0.874) |
| Field eval, PlantDoc photos | **52%** accuracy |
| Field eval, overall | 35% (0% on the 4 pest classes; 3 of them have no training data) |
| Field calibration error (ECE) | 0.36. The temperature (0.72) was fitted on lab validation and makes field answers overconfident. |
| OOD rejected | 43%. Soil 0%, other crops 57%, blurry 17%. Green non-tomato leaves mostly come out as "healthy". |

What this baseline shows:

1. **Domain shift is real.** Accuracy drops from 89% on lab validation to 52% on field
   photos. The field eval set catches what a lab test set hides.
2. **The unknown class needs negatives.** It only learned leaf mold and target spot (75%
   of out-of-scope tomato diseases rejected), not "not a tomato leaf at all". Next step:
   add non-tomato images (PlantVillage pepper and potato, other crops, soil) to `unknown`.
3. **Calibrate on field-like data,** not lab validation. Otherwise the confident
   threshold means little in the field.
4. **Pests need data:** Tuta, whitefly and fruit borer have none. Spider mites come only
   from lab photos and score 0% on field photos.
5. **A GPU run** that fine-tunes the whole network over all the data should beat this
   baseline. Compare against it with `--baseline ml/runs/v3_lite0_cpu/eval_preds.jsonl`.
