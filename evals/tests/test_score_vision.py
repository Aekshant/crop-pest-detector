"""Tests for evals/score_vision.py. Run: python -m unittest discover evals/tests"""

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS))
import score_vision as sv  # noqa: E402

FIELD = sv.read_labels(sv.VISION / "field" / "labels.csv")
OOD = sv.read_labels(sv.VISION / "ood" / "labels.csv")


def oracle(prob=0.95):
    """Correct label on field, unknown on OOD."""
    preds = {}
    for r in FIELD:
        truth = sv.v1_label(r["label"])
        other = "healthy" if truth == "unknown" else "unknown"
        preds[f"field/{r['file']}"] = {"probs": {truth: prob, other: 1 - prob}, "quality": "ok"}
    for r in OOD:
        preds[f"ood/{r['file']}"] = {"probs": {"unknown": prob, "healthy": 1 - prob}, "quality": "ok"}
    return preds


def write(preds, path):
    path.write_text("\n".join(json.dumps({"file": k, **v}) for k, v in preds.items()))


class ScoreVisionTest(unittest.TestCase):
    def test_oracle_is_perfect(self):
        f = sv.score_field(FIELD, oracle(), reject=0.5, accept=0.8)
        o = sv.score_ood(OOD, oracle(), reject=0.5)
        self.assertEqual(f["accuracy"], 1.0)
        self.assertEqual(f["min_recall"], 1.0)
        self.assertEqual(f["false_reject_rate"], 0.0)
        self.assertEqual(o["reject_rate"], 1.0)
        self.assertAlmostEqual(f["ece"], 0.05)  # always 95% sure, always right

    def test_low_confidence_counts_as_reject(self):
        preds = oracle()
        first = f"field/{FIELD[0]['file']}"
        preds[first] = {"probs": {FIELD[0]["label"]: 0.4, "healthy": 0.35, "unknown": 0.25}, "quality": "ok"}
        f = sv.score_field(FIELD, preds, reject=0.5, accept=0.8)
        in_scope = sum(sv.v1_label(r["label"]) != "unknown" for r in FIELD)
        self.assertAlmostEqual(f["false_reject_rate"], 1 / in_scope)
        self.assertAlmostEqual(f["accuracy"], 1 - 1 / len(FIELD))
        self.assertEqual(f["top3_accuracy"], 1.0)

    def test_retake_rejects_ood(self):
        preds = oracle()
        blurry = next(r for r in OOD if r["category"] == "blurry")
        preds[f"ood/{blurry['file']}"] = {"probs": {"healthy": 0.99}, "quality": "retake"}
        o = sv.score_ood(OOD, preds, reject=0.5)
        self.assertEqual(o["reject_rate"], 1.0)

    def test_false_accept_listed(self):
        preds = oracle()
        crop = next(r for r in OOD if r["category"] == "other_crop")
        preds[f"ood/{crop['file']}"] = {"probs": {"late_blight": 0.9, "unknown": 0.1}, "quality": "ok"}
        o = sv.score_ood(OOD, preds, reject=0.5)
        self.assertEqual(o["false_accepts"], [[crop["file"], crop["subject"], "late_blight", 0.9]])

    def test_ece_two_bins(self):
        # 0.9 confident and right, 0.6 confident and wrong: (|0.9-1| + |0.6-0|) / 2
        self.assertAlmostEqual(sv.ece([0.9, 0.6], [True, False]), 0.35)

    def test_softmax_temperature(self):
        p = sv.softmax({"a": 2.0, "b": 0.0}, temperature=2.0)
        self.assertAlmostEqual(p["a"], 1 / (1 + math.exp(-1)))

    def test_cli_regression_exit_code(self):
        good, bad = oracle(), oracle()
        for r in [r for r in FIELD if sv.v1_label(r["label"]) != "unknown"][:20]:  # 20 wrong answers
            bad[f"field/{r['file']}"] = {"probs": {"unknown": 0.1, "healthy" if r["label"] != "healthy" else "late_blight": 0.9}, "quality": "ok"}
        with tempfile.TemporaryDirectory() as d:
            g, b = Path(d) / "good.jsonl", Path(d) / "bad.jsonl"
            write(good, g)
            write(bad, b)
            run = lambda *a: subprocess.run([sys.executable, str(EVALS / "score_vision.py"), *a],
                                            capture_output=True, text=True)
            self.assertEqual(run(str(g), "--baseline", str(g)).returncode, 0)
            r = run(str(b), "--baseline", str(g))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("WORSE", r.stdout)
            self.assertEqual(run(str(g), "--baseline", str(b)).returncode, 0)

    def test_out_of_scope_disease_must_be_rejected(self):
        mold = next(r for r in FIELD if r["label"] == "leaf_mold")
        self.assertEqual(sv.v1_label("leaf_mold"), "unknown")
        preds = oracle()
        f = sv.score_field(FIELD, preds, reject=0.5, accept=0.8)
        self.assertEqual(f["accuracy"], 1.0)
        preds[f"field/{mold['file']}"] = {"probs": {"early_blight": 0.9, "unknown": 0.1}, "quality": "ok"}
        f = sv.score_field(FIELD, preds, reject=0.5, accept=0.8)
        self.assertLess(f["per_class_recall"]["unknown"], 1.0)
        self.assertEqual(f["false_reject_rate"], 0.0)

    def test_label_outside_map_is_an_error(self):
        preds = oracle()
        preds[f"field/{FIELD[0]['file']}"] = {"probs": {"leaf_mold": 0.9, "unknown": 0.1}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.jsonl"
            write(preds, p)
            r = subprocess.run([sys.executable, str(EVALS / "score_vision.py"), str(p)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not in labels_v1.json", r.stderr)

    def test_missing_predictions_fail(self):
        preds = oracle()
        del preds[f"field/{FIELD[0]['file']}"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.jsonl"
            write(preds, p)
            r = subprocess.run([sys.executable, str(EVALS / "score_vision.py"), str(p)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
