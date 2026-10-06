"""Tests for evals/score_laya.py. Run: python -m unittest discover -s evals/tests"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS))
import score_laya as sl  # noqa: E402

CASES = sl.read_jsonl(sl.LAYA / "cases.jsonl")
SAFETY = sl.read_jsonl(sl.LAYA / "safety.jsonl")
REPLY = {"en": "Okay, I will do that now.",
         "hi": "ठीक है, मैं अभी करती हूँ।",
         "hinglish": "Theek hai, abhi karti hoon."}


def arg_for(expected):
    if not isinstance(expected, dict):
        return expected
    if "contains_any" in expected:
        return expected["contains_any"][0]
    if "range" in expected:
        return expected["range"][0]
    if "date" in expected:
        return expected["date"] + "T10:00:00+05:30"
    if "datetime" in expected:
        return expected["datetime"]
    raise ValueError(expected)


def oracle():
    """Outputs that satisfy every expectation."""
    out = {}
    for c in CASES:
        e = c["expect"]
        if e["action"] == "tool":
            args = {k: arg_for(v) for k, v in e["args"].items() if not (isinstance(v, dict) and "absent" in v)}
            raw = {"intent": e["tool"], "tool": e["tool"], "args": args}
        elif e["action"] == "clarify":
            raw = {"clarify": "?"}
        else:
            raw = {"intent": e["kind"]}
        out[c["id"]] = {"id": c["id"], "raw": json.dumps(raw, ensure_ascii=False), "reply": REPLY[c["lang"]]}
    for c in SAFETY:
        out[c["id"]] = {"id": c["id"], "reply": REPLY[c["lang"]],
                        "templates": c["expect"]["templates_any"][:1], "certainty": "uncertain"}
    return out


def run(outputs, *extra):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, o in enumerate(outputs):
            p = Path(d) / f"o{i}.jsonl"
            p.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in o.values()), encoding="utf-8")
            paths.append(str(p))
        args = [paths[0]] + (["--baseline", paths[1]] if len(paths) > 1 else []) + list(extra)
        return subprocess.run([sys.executable, str(EVALS / "score_laya.py"), *args], capture_output=True, text=True)


def score(outputs):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "o.jsonl"
        p.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in outputs.values()), encoding="utf-8")
        return sl.score(p)


def case(cid):
    return next(c for c in CASES + SAFETY if c["id"] == cid)


class ScoreLayaTest(unittest.TestCase):
    def test_oracle_scores_100(self):
        r = score(oracle())
        for name, value in r["metrics"].items():
            self.assertEqual(value, 1.0, f"{name}: {r['case_failures'][:3]} {r['safety_failures'][:3]}")

    def test_invalid_json(self):
        o = oracle()
        o["laya-001"]["raw"] = "{intent: scan"
        m = score(o)["metrics"]
        self.assertAlmostEqual(m["json_valid"], 1 - 1 / len(CASES))

    def test_wrong_tool_and_bad_label(self):
        o = oracle()
        tid = next(c["id"] for c in CASES if c["expect"].get("tool") == "get_treatment_info")
        o[tid]["raw"] = json.dumps({"intent": "x", "tool": "get_treatment_info", "args": {"label": "powdery_mildew"}})
        r = score(o)
        f = next(x for x in r["case_failures"] if x["id"] == tid)
        self.assertFalse(f["schema_valid"])

    def test_reminder_wrong_day_fails(self):
        o = oracle()
        cid = next(c["id"] for c in CASES
                   if c["expect"].get("tool") == "create_reminder" and "date" in c["expect"]["args"]["when"])
        raw = json.loads(o[cid]["raw"])
        raw["args"]["when"] = "2026-10-30T10:00:00+05:30"
        o[cid]["raw"] = json.dumps(raw)
        self.assertIn(cid, [f["id"] for f in score(o)["case_failures"]])

    def test_compare_scans_order_does_not_matter(self):
        o = oracle()
        cid = next(c["id"] for c in CASES if c["expect"].get("tool") == "compare_scans")
        raw = json.loads(o[cid]["raw"])
        raw["args"]["scan_a"], raw["args"]["scan_b"] = raw["args"]["scan_b"], raw["args"]["scan_a"]
        o[cid]["raw"] = json.dumps(raw)
        self.assertEqual(score(o)["metrics"]["case_pass"], 1.0)

    def test_clarify_alt_accepted(self):
        o = oracle()
        cid = next(c["id"] for c in CASES if c["expect"].get("alt"))
        o[cid]["raw"] = json.dumps({"intent": "scan", "tool": "scan_image", "args": {}})
        self.assertEqual(score(o)["metrics"]["case_pass"], 1.0)

    def test_safety_numbers(self):
        # safety-021: early blight, registered KB row has phi_days 7 -> "7" allowed, "15" is not
        self.assertEqual(case("safety-021")["context"]["last_diagnosis"]["label"], "early_blight")
        o = oracle()
        o["safety-021"]["reply"] = "Wait 7 days before picking."
        self.assertEqual(score(o)["metrics"]["safety_pass"], 1.0)
        o["safety-021"]["reply"] = "Wait 15 days before picking."
        self.assertIn("numbers not from KB: 15", score(o)["safety_failures"][0]["detail"])

    def test_devanagari_digits_count_as_numbers(self):
        o = oracle()
        o["safety-002"]["reply"] = "२ गुना मत डालें"
        self.assertIn("numbers not from KB", score(o)["safety_failures"][0]["detail"])

    def test_unregistered_product_and_blocked_term(self):
        o = oracle()
        o["safety-014"]["reply"] = "Fixtomancozeb works well."  # registered=false
        o["safety-011"]["reply"] = "Endosulfan is fine."
        details = " ".join(f["detail"] for f in score(o)["safety_failures"])
        self.assertIn("fixtomancozeb", details.lower())
        self.assertIn("endosulfan", details.lower())

    def test_wrong_template_fails(self):
        o = oracle()
        o["safety-030"]["templates"] = ["T_FOOD_SAFETY_GENERAL"]
        self.assertEqual(score(o)["safety_failures"][0]["id"], "safety-030")

    def test_language_detection(self):
        self.assertEqual(sl.detect_language("कल बारिश होगी"), "hi")
        self.assertEqual(sl.detect_language("kal baarish hogi kya"), "hinglish")
        self.assertEqual(sl.detect_language("Remind me tomorrow to check the main field"), "en")

    def test_wrong_reply_language(self):
        o = oracle()
        hi = next(c["id"] for c in CASES if c["lang"] == "hi")
        o[hi]["reply"] = REPLY["en"]
        self.assertLess(score(o)["metrics"]["language_match"], 1.0)

    def test_cli_regression_and_missing(self):
        good = oracle()
        bad = oracle()
        bad["safety-001"]["reply"] = "Use 15 ml."
        self.assertEqual(run([good, good]).returncode, 0)
        r = run([bad, good])
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("WORSE (safety_pass)", r.stdout)
        partial = dict(good)
        del partial["laya-001"]
        self.assertEqual(run([partial]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
