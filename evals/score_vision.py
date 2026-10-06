#!/usr/bin/env python3
"""Score vision predictions on the field and OOD eval sets.

  python evals/score_vision.py preds.jsonl
  python evals/score_vision.py preds.jsonl --baseline current.jsonl   # exit 1 if worse

preds.jsonl has one line per image, keyed by path relative to evals/vision/:

  {"file": "field/plantdoc_early_blight_0a1b2c3d4e5f.jpg",
   "probs": {"early_blight": 0.81, "septoria_leaf_spot": 0.12, ...},
   "quality": "ok"}

- "probs" are calibrated probabilities (temperature already applied). Instead,
  "logits" plus --temperature can be given and softmax is applied here.
- "quality" is the quality-gate result: "ok" (default) or "retake".
- Label names must be ids from ml/labels_v1.json; anything else is an error.

An image counts as rejected when the gate says retake, the top label is
"unknown", or the top probability is below --reject. Rejecting a field image
is a miss; rejecting an OOD image is the correct outcome.
"""

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parent / "vision"
LABELS_V1 = Path(__file__).resolve().parents[1] / "ml" / "labels_v1.json"
UNKNOWN = "unknown"
ECE_BINS = 15

# Allowed change versus the baseline before a metric counts as a regression.
# Same idea as the design doc's shadow evaluation: publish only if nothing
# regresses beyond its threshold.
REGRESSION = {
    # metric: (direction, tolerance) - direction +1 means higher is better
    "field.accuracy": (+1, 0.02),
    "field.macro_recall": (+1, 0.02),
    "field.min_recall": (+1, 0.05),
    "field.confident_accuracy": (+1, 0.02),
    "field.false_reject_rate": (-1, 0.03),
    "field.ece": (-1, 0.02),
    "ood.reject_rate": (+1, 0.03),
}


def load_label_map():
    """v1 label ids, plus the map from older eval label names to v1 ids."""
    m = json.loads(LABELS_V1.read_text())
    ids = [l["id"] for l in m["labels"]]
    aliases = {k: v for k, v in m["source_map"]["eval"].items() if k != "note"}
    return ids, aliases


LABEL_IDS, EVAL_ALIASES = load_label_map()


def v1_label(name):
    label = EVAL_ALIASES.get(name, name)
    if label not in LABEL_IDS:
        raise SystemExit(f"eval label {name!r} is not in labels_v1.json")
    return label


def read_labels(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def softmax(logits, temperature):
    m = max(logits.values())
    exps = {k: math.exp((v - m) / temperature) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def read_preds(path, temperature):
    preds = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = json.loads(line)
        if "probs" in p:
            probs = p["probs"]
        elif "logits" in p:
            probs = softmax(p["logits"], temperature)
        else:
            raise SystemExit(f"{path}:{n}: needs 'probs' or 'logits'")
        bad = set(probs) - set(LABEL_IDS)
        if bad:
            raise SystemExit(f"{path}:{n}: labels not in labels_v1.json: {sorted(bad)}")
        preds[p["file"]] = {"probs": probs, "quality": p.get("quality", "ok")}
    return preds


def decide(pred, reject):
    """Return (top_label, top_prob, ranked_labels, rejected)."""
    ranked = sorted(pred["probs"].items(), key=lambda kv: -kv[1])
    top, prob = ranked[0]
    rejected = pred["quality"] == "retake" or top == UNKNOWN or prob < reject
    return top, prob, [k for k, _ in ranked], rejected


def ece(confidences, correct):
    bins = [[0, 0.0, 0.0] for _ in range(ECE_BINS)]  # count, sum conf, sum correct
    for c, ok in zip(confidences, correct):
        b = bins[min(int(c * ECE_BINS), ECE_BINS - 1)]
        b[0] += 1
        b[1] += c
        b[2] += ok
    n = len(confidences)
    return sum(abs(s_c - s_ok) for cnt, s_c, s_ok in bins if cnt) / n if n else 0.0


def score_field(rows, preds, reject, accept):
    per_class = collections.defaultdict(lambda: [0, 0])  # label -> [n, hits]
    per_source = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    hits = top3 = rejected_n = answered = answered_hits = confident_n = confident_hits = 0
    in_scope = 0
    confs, correct = [], []
    missing = 0
    for r in rows:
        key = f"field/{r['file']}"
        # Out-of-scope tomato problems (e.g. leaf mold) have truth "unknown":
        # rejecting them is the right answer.
        truth = v1_label(r["label"])
        in_scope += truth != UNKNOWN
        per_class[truth][0] += 1
        per_source[r["source"]][0] += 1
        pred = preds.get(key)
        if pred is None:
            missing += 1
            continue
        top, prob, ranked, rejected = decide(pred, reject)
        ok = rejected if truth == UNKNOWN else (not rejected and top == truth)
        hits += ok
        per_class[truth][1] += ok
        per_source[r["source"]][1] += ok
        top3 += truth in ranked[:3]
        rejected_n += rejected and truth != UNKNOWN
        if not rejected:
            answered += 1
            answered_hits += top == truth
        if not rejected and top != truth:
            confusions[(truth, top)] += 1
        if not rejected and prob >= accept:
            confident_n += 1
            confident_hits += top == truth
        # Calibration over the model's own top-1, before any rejection rule.
        confs.append(prob)
        correct.append(top == truth)

    n = len(rows)
    recalls = {k: h / c for k, (c, h) in per_class.items()}
    return {
        "n": n,
        "missing": missing,
        "accuracy": hits / n,
        "top3_accuracy": top3 / n,
        "false_reject_rate": rejected_n / in_scope if in_scope else 0.0,
        "selective_accuracy": answered_hits / answered if answered else 0.0,
        "confident_coverage": confident_n / n,
        "confident_accuracy": confident_hits / confident_n if confident_n else 0.0,
        "macro_recall": sum(recalls.values()) / len(recalls),
        "min_recall": min(recalls.values()),
        "ece": ece(confs, correct),
        "per_class_recall": dict(sorted(recalls.items())),
        "per_class_n": {k: c for k, (c, _) in sorted(per_class.items())},
        "per_source_accuracy": {k: h / c for k, (c, h) in sorted(per_source.items())},
        "top_confusions": [[t, p, k] for (t, p), k in confusions.most_common(5)],
    }


def score_ood(rows, preds, reject):
    per_cat = collections.defaultdict(lambda: [0, 0])  # category -> [n, rejected]
    false_accepts, missing, rejected_total = [], 0, 0
    for r in rows:
        key = f"ood/{r['file']}"
        per_cat[r["category"]][0] += 1
        pred = preds.get(key)
        if pred is None:
            missing += 1
            continue
        top, prob, _, rejected = decide(pred, reject)
        rejected_total += rejected
        per_cat[r["category"]][1] += rejected
        if not rejected:
            false_accepts.append([r["file"], r["subject"], top, round(prob, 3)])
    n = len(rows)
    return {
        "n": n,
        "missing": missing,
        "reject_rate": rejected_total / n,
        "per_category_reject_rate": {k: rj / c for k, (c, rj) in sorted(per_cat.items())},
        "false_accepts": false_accepts,
    }


def score(preds_path, args):
    preds = read_preds(preds_path, args.temperature)
    field = score_field(read_labels(VISION / "field" / "labels.csv"), preds, args.reject, args.accept)
    ood = score_ood(read_labels(VISION / "ood" / "labels.csv"), preds, args.reject)
    return {"preds": str(preds_path), "reject": args.reject, "accept": args.accept,
            "field": field, "ood": ood}


def pct(x):
    return f"{100 * x:5.1f}%"


def print_report(r):
    f, o = r["field"], r["ood"]
    print(f"Vision eval: {r['preds']}   (reject < {r['reject']}, confident >= {r['accept']})")
    print()
    print(f"Field set ({f['n']} images" + (f", {f['missing']} MISSING" if f["missing"] else "") + ")")
    print(f"  accuracy            {pct(f['accuracy'])}   top-3 {pct(f['top3_accuracy'])}")
    print(f"  false reject rate   {pct(f['false_reject_rate'])}   accuracy when answered {pct(f['selective_accuracy'])}")
    print(f"  confident answers   {pct(f['confident_coverage'])} of images, {pct(f['confident_accuracy'])} correct")
    print(f"  macro recall        {pct(f['macro_recall'])}   worst class {pct(f['min_recall'])}")
    print(f"  calibration (ECE)   {f['ece']:.3f}")
    print("  per-class recall")
    for label, rec in f["per_class_recall"].items():
        print(f"    {label:24s} {pct(rec)}  (n={f['per_class_n'][label]})")
    print("  accuracy by source")
    for src, acc in f["per_source_accuracy"].items():
        print(f"    {src:24s} {pct(acc)}")
    if f["top_confusions"]:
        print("  top confusions (true -> predicted)")
        for t, p, k in f["top_confusions"]:
            print(f"    {t} -> {p}: {k}")
    print()
    print(f"OOD set ({o['n']} images" + (f", {o['missing']} MISSING" if o["missing"] else "") + ")")
    print(f"  rejected            {pct(o['reject_rate'])}")
    for cat, rate in o["per_category_reject_rate"].items():
        print(f"    {cat:24s} {pct(rate)}")
    if o["false_accepts"]:
        print("  wrongly accepted")
        for file, subject, label, prob in o["false_accepts"]:
            print(f"    {file} ({subject}) -> {label} {prob}")


def get(r, dotted):
    section, key = dotted.split(".")
    return r[section][key]


def compare(cur, base):
    """Print metric deltas; return list of regressed metric names."""
    print()
    print(f"Versus baseline: {base['preds']}")
    print(f"  {'metric':26s} {'baseline':>9s} {'this':>9s} {'delta':>8s}")
    worse = []
    for name, (direction, tol) in REGRESSION.items():
        b, c = get(base, name), get(cur, name)
        delta = c - b
        if direction * delta < -tol:
            verdict = "WORSE"
            worse.append(name)
        elif direction * delta > tol:
            verdict = "better"
        else:
            verdict = ""
        print(f"  {name:26s} {b:9.3f} {c:9.3f} {delta:+8.3f}  {verdict}")
    print()
    print("Verdict: " + (f"WORSE ({', '.join(worse)})" if worse else "not worse than baseline"))
    return worse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", type=Path)
    ap.add_argument("--baseline", type=Path, help="predictions of the current model to compare against")
    ap.add_argument("--reject", type=float, default=0.5, help="reject threshold (manifest value)")
    ap.add_argument("--accept", type=float, default=0.8, help="confident threshold (manifest value)")
    ap.add_argument("--temperature", type=float, default=1.0, help="applied to 'logits' inputs")
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = ap.parse_args()

    cur = score(args.preds, args)
    print_report(cur)
    worse = []
    if args.baseline:
        base = score(args.baseline, args)
        cur["baseline"] = base
        worse = compare(cur, base)
        cur["regressions"] = worse
    if args.json:
        args.json.write_text(json.dumps(cur, indent=2))
    if cur["field"]["missing"] or cur["ood"]["missing"]:
        print("\nERROR: predictions missing for some images", file=sys.stderr)
        return 2
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
