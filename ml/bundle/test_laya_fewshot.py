"""L5: the few-shot examples are valid tool calls and not copies of eval cases.

  python -m unittest discover -s ml/bundle -p "test_*.py"
"""

import json
import sys
import unittest
from pathlib import Path

import jsonschema

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[1]
sys.path.insert(0, str(BUNDLE))
import tool_schema as T  # noqa: E402

FEWSHOT = json.loads((BUNDLE / "laya_fewshot_v1.json").read_text())
SCHEMA = T.load()


def trigrams(text):
    t = " ".join(text.lower().split())
    return {t[i:i + 3] for i in range(len(t) - 2)}


def similarity(a, b):
    A, B = trigrams(a), trigrams(b)
    return len(A & B) / max(1, len(A | B))


class FewShotTest(unittest.TestCase):
    def test_count_and_languages(self):
        ex = FEWSHOT["examples"]
        self.assertTrue(6 <= len(ex) <= 8)
        self.assertGreaterEqual(sum(e["lang"] == "hinglish" for e in ex), 6)

    def test_examples_are_schema_valid(self):
        offline_tools = [n for n, t in SCHEMA["tools"].items() if not t["needsNetwork"]]
        for e in FEWSHOT["examples"]:
            offline = e.get("context_override", {}).get("network") == "OFFLINE"
            v = jsonschema.Draft202012Validator(T.understand_schema(SCHEMA, tools=offline_tools if offline else None))
            self.assertTrue(v.is_valid(e["output"]), f"{e['farmer']}: {list(v.iter_errors(e['output']))[:1]}")

    def test_examples_use_only_their_own_context_ids(self):
        ctx = FEWSHOT["context"]
        ids = {f["id"] for f in ctx["fields"]} | {s["id"] for s in ctx["recent_scans"]}
        for e in FEWSHOT["examples"]:
            for k, v in e["output"].get("args", {}).items():
                if k in ("field_id", "scan_a", "scan_b"):
                    self.assertIn(v, ids)
        # And none of the eval context's ids leak into examples.
        eval_ctx = json.loads((REPO / "evals" / "laya" / "context_default.json").read_text())
        eval_ids = {f["id"] for f in eval_ctx["fields"]} | {s["id"] for s in eval_ctx["recent_scans"]}
        self.assertFalse(ids & eval_ids)

    def test_not_near_copies_of_eval_cases(self):
        inputs = []
        for name in ("cases.jsonl", "safety.jsonl"):
            for line in (REPO / "evals" / "laya" / name).read_text().splitlines():
                if line.strip():
                    inputs.append(json.loads(line)["input"])
        for e in FEWSHOT["examples"]:
            worst = max(inputs, key=lambda s: similarity(e["farmer"], s))
            self.assertLess(similarity(e["farmer"], worst), 0.5, f"{e['farmer']!r} too close to eval {worst!r}")


if __name__ == "__main__":
    unittest.main()
