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

### Run `runs/v3b_lite0_cpu`: data fixes (current best)

Same model and CPU recipe as the baseline, with three fixes for the problems it showed:

1. **Negatives for `unknown`:** 60 PlantVillage leaves from each of 28 other-crop classes,
   880 iNaturalist field photos of 22 other crops and weeds, and 125 soil and hand photos
   from Wikimedia Commons (`ml/data/fetch_web_train.py`).
2. **Pest data:** iNaturalist photos of Tuta (73: only leaf mines, larvae and photos on
   tomato, because the rest are adult moths on walls), whitefly (299), fruit borer larvae
   (300) and spider mites (294). They are not hand-screened, and a 30-image spot check per
   class looked acceptable.
3. **Field-aware training:** model selection uses the mean of lab and field validation
   macro recall. The temperature is fitted on field validation photos. The **reject
   threshold** is chosen on field validation (at most 20% of good photos turned away) and
   saved with the model (`thresholds.json`, `model.pt["thresholds"]`). It is never tuned
   on the eval sets.

Each model is compared at its own thresholds:

```bash
python evals/score_vision.py ml/runs/v3b_lite0_cpu/eval_preds.jsonl --reject 0.35 \
    --baseline ml/runs/v3_lite0_cpu/eval_preds.jsonl --baseline-reject 0.5
```

| Field and OOD eval | v3 baseline (reject < 0.5) | v3b (reject < 0.35) |
|---|---|---|
| Field accuracy | 34.7% | **52.8%** |
| Confident answers correct | 46.6% | **84.8%** |
| Good photos turned away | 23.5% | **15.2%** |
| Calibration error (ECE) | 0.36 | **0.15** |
| OOD rejected | 43% | **67%** |
| Fruit borer / whitefly / spider mites recall | 0 / 0 / 0% | 83 / 83 / 58% |

At a fixed 0.5 threshold the scorer called v3b "worse" on false rejects. That was a
threshold artefact: the old model's temperature (0.72) sharpened its probabilities. This
is why thresholds now travel with the model.

**Still weak:** Tuta (8%, only 59 training photos), mosaic virus (8%), bacterial spot
(17%) and healthy (25%) on field photos. Hands and soil are sometimes taken as pests,
because many pest photos have a hand or soil in frame. Blurred photos need the quality
gate, which isn't built yet. More own field photos for these classes would help most.

## Current best: `runs/v3b_lite0_cpu` + quality gate

| | Without gate | With gate (`eval_report_gated.txt`) |
|---|---|---|
| Field accuracy | 52.8% | 50.0% |
| Good field photos turned away | 15.2% | 18.9% (4 sent to retake) |
| Confident answers correct | 84.8% | 84.1% |
| OOD rejected | 67% | **80%** (blurry 100%, soil 80%, other crops 79%, hands 60%) |

Ship it with the gate. A retake prompt on a borderline photo costs the farmer one more
photo, while a blurry photo without the gate gets a confident wrong diagnosis.

## Quality gate: [`quality_gate.py`](quality_gate.py), [`quality_gate.json`](quality_gate.json)

This is step 2 of the vision pipeline, and it runs before the model:

- **Blur:** the no-reference *blur effect* (Crété-Roffet et al., 2007) at a 512 px short
  side. The photo is compared with a re-blurred copy of itself.
- **Exposure:** mean brightness.

Thresholds are calibrated on 1,500 **training** field photos (never the eval sets). At most
5% of sharp photos are sent to a retake, and 79% of blurred copies are caught. Laplacian
variance, the textbook choice, caught only 42% at the same 5%: it depends too much on how
busy the scene is.

```bash
ml/.venv/bin/python ml/quality_gate.py --calibrate      # refit thresholds
ml/.venv/bin/python ml/quality_gate.py photo.jpg        # check photos
```

`train.py` applies the gate when it writes eval predictions. The C++ core must implement
the same metric and thresholds.

## Experiments that did not help (kept as logs, no checkpoints)

**`runs/v3c_lite0_cpu`, `runs/v3d_lite0_cpu`.** These runs added about 9,000 field images
(Pakistan, Bangladesh, field leaf/fruit, Tomato-Village), extra hand and soil negatives,
and a paper-to-field background swap (`augment.py`).

| | v3b | v3c | v3d |
|---|---|---|---|
| Field accuracy | **52.8%** | 36.1% | 39.6% |
| PlantDoc photos | **45.8%** | 31.2% | 33.3% |
| Confident answers correct | 84.8% | 71.1% | **89.3%** |
| Hands rejected | **60%** | 40% | 0% |

What these runs taught:

1. **Validate on photos like the app's.** In v3c, selection and calibration used all field
   validation, and most of it was leaves on paper. Validation improved while the eval got
   worse. Now only *in-situ* photos (`IN_SITU_SOURCES`) judge a model; paper photos only
   train it.
2. **More sources diluted PlantDoc-style photos.** The field share per class went from
   PlantDoc alone to mostly paper and Pakistan photos, and the eval is mostly PlantDoc-style
   web photos. Next time, balance sampling **per source**, not just per class.
3. **The CPU recipe** (classifier + last 2 blocks) can't absorb this much variety.

The data and code stay. They are opt-in: `train.py --exclude-sources ''`, and the default
leaves them out. Next attempt: a full fine-tune on a GPU with per-source balancing:

```bash
python ml/train.py --out ml/runs/v4_lite0_gpu --exclude-sources '' --unfreeze-blocks 0 \
    --samples-per-epoch 0 --epochs-ft 15 --batch 128 --workers 8 --val-max 0
```

## Still open

- **Tuta absoluta:** there is no public field data of real Tuta damage. Tomato-Village and
  the Bangladesh set label *Liriomyza* (serpentine) leaf miner, not Tuta. The NM-AIST
  (Tanzania) Tuta set isn't public, and TomatoEbola (Nigeria) needs a login. Own photos,
  or asking NM-AIST, are the only route.
- **Serpentine leaf miner** is common in Rajasthan (Tomato-Village) and has no v1 class.
  Consider it for `labels_v2`.
- **Mosaic virus:** no public field photos found.

## On-device export (V6–V9): [`export.py`](export.py), [`bundle/manifest_draft.json`](bundle/manifest_draft.json)

```bash
ml/.venv-export/bin/python ml/export.py --run ml/runs/v3b_lite0_cpu   # setup: see the export.py docstring
```

The model is exported with `litert-torch` and quantized with `ai-edge-quantizer`. Its input
is NHWC RGB 0–255 at 224×224 (normalization is inside the model), and its output is the 12
logits in `labels_v1` order. Each variant gets **its own temperature and reject
threshold**, refit on in-situ field validation photos (V8), because quantization shifts
probabilities. Scores below include the quality gate. Eval: v3b, field and OOD sets.
| Variant | Size | Top-1 agreement with PyTorch | Field accuracy | Confident answers correct | ECE | OOD rejected | Desktop CPU p50 / p90 |
|---|---|---|---|---|---|---|---|
| FP32 | 13.59 MB | 100.0% | 49.3% | 84.4% | 0.162 | 80% | 4.9 / 9.5 ms |
| FP16 | 6.93 MB | 99.4% | 50.0% | 84.4% | 0.159 | 83% | 4.6 / 7.0 ms |
| INT8 | 4.12 MB | 92.0% | 50.7% | 85.1% | 0.117 | 80% | 3.6 / 3.8 ms |

- **Ship INT8 for the CPU path (XNNPACK) and FP16 for the GPU delegate.** INT8 is 3.3×
  smaller than FP32 and fits the 5–15 MB budget, with no loss on the field eval.
- **INT8 quantizes only CONV_2D and DEPTHWISE_CONV_2D,** with per-channel weights and
  min/max ranges from 192 training images. Quantizing every op dropped top-1 agreement to
  40%: the pooling `SUM`, `GATHER_ND` and classifier don't survive INT8. 16-bit
  activations were worse still (3%), and MSE ranges reached 78%. Conv-only reaches 92%
  and is the same size, because the convolutions hold almost all the weights. The 8% of
  disagreements are borderline images that fall both ways, so accuracy holds.
- **Latency is on this desktop CPU (4 threads),** not a phone. The budget's phone numbers
  (< 80 ms GPU on tier A/B, < 150 ms CPU INT8 on tier C) are measured on devices in the
  Android tasks.
- The `.tflite` files are in Git LFS. `manifest_draft.json` records their sha256 and
  per-file calibration.
