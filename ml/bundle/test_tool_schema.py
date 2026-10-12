"""L3 validation test for the Laya tool schema.

  pip install jsonschema
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

SCHEMA = T.load()
UNDERSTAND = T.understand_schema(SCHEMA)
VALIDATOR = jsonschema.Draft202012Validator(UNDERSTAND)


def valid(obj, validator=VALIDATOR):
    return validator.is_valid(obj)


def expected_output(expect):
    """A correct UNDERSTAND output for an eval case's expectation."""
    if expect["action"] == "clarify":
        return {"clarify": "Kab yaad dilaun?"}
    if expect["action"] == "respond":
        return {"intent": expect["kind"], "reply": "ok"}
    args = {}
    for name, v in expect["args"].items():
        if isinstance(v, dict):
            if "absent" in v:
                continue
            if "contains_any" in v:
                v = v["contains_any"][0]
            elif "range" in v:
                v = v["range"][0]
            elif "date" in v:
                v = v["date"] + "T09:00:00+05:30"
            elif "datetime" in v:
                v = v["datetime"]
        args[name] = v
    return {"intent": expect["tool"], "tool": expect["tool"], "args": args, "reply": "ok"}


class ToolSchemaTest(unittest.TestCase):
    def test_schema_file_is_consistent(self):
        self.assertEqual(T.check(SCHEMA), [])
        self.assertEqual(SCHEMA["schema_version"], 1)
        jsonschema.Draft202012Validator.check_schema(UNDERSTAND)

    def test_manifest_declares_this_schema_version(self):
        manifest = json.loads((BUNDLE / "manifest_draft.json").read_text())
        self.assertEqual(manifest["laya"]["tool_schema_version"], SCHEMA["schema_version"])

    def test_every_expected_eval_output_is_valid(self):
        cases = [json.loads(l) for l in (REPO / "evals" / "laya" / "cases.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(cases), 150)
        for c in cases:
            out = expected_output(c["expect"])
            self.assertTrue(valid(out), f"{c['id']}: {out} -> {list(VALIDATOR.iter_errors(out))[:1]}")

    def test_invalid_outputs_are_rejected(self):
        ok = {"intent": "reminder", "tool": "create_reminder",
              "args": {"when": "2026-10-21T08:00:00+05:30", "text": "spray"}, "reply": "ok"}
        self.assertTrue(valid(ok))
        bad = {
            "unknown tool": {**ok, "tool": "buy_pesticide"},
            "missing required arg": {**ok, "args": {"text": "spray"}},
            "extra arg": {**ok, "args": {**ok["args"], "urgent": True}},
            "date not ISO 8601": {**ok, "args": {**ok["args"], "when": "kal subah"}},
            "args null (seen in L2)": {**ok, "args": None},
            "missing reply": {k: v for k, v in ok.items() if k != "reply"},
            "label outside the set": {"intent": "t", "tool": "get_treatment_info", "args": {"label": "powdery_mildew"}, "reply": "x"},
            "days as text": {"intent": "h", "tool": "get_scan_history", "args": {"days": "7"}, "reply": "x"},
            "days out of range": {"intent": "h", "tool": "get_scan_history", "args": {"days": 0}, "reply": "x"},
            "empty clarify": {"clarify": ""},
            "unknown respond kind": {"intent": "chitchat", "reply": "hi"},
            "runaway reply (looping)": {"intent": "greeting", "reply": "बर्फ के " * 60},
            "reply in Urdu script (seen in L1)": {"intent": "greeting", "reply": "بالطبع! کال بااریش"},
            "reply in Tamil script (seen in L2)": {"intent": "greeting", "reply": "இதைப் பச்சைப்"},
            "escaped characters in a note (seen in L4)": {"intent": "n", "tool": "record_observation",
                                                           "args": {"text": "क्य\\u0628"}, "reply": "ok"},
        }
        for why, obj in bad.items():
            self.assertFalse(valid(obj), why)

    def test_normal_replies_in_all_three_languages_pass(self):
        for reply in ("Opening the camera now.", "कैमरा खोल रही हूँ, पत्ती दिखाइए।", "Theek hai, kal subah 8 baje yaad dilaungi!",
                      "आज शाम 5 बजे पानी देने की याद दिलाऊँगी।", "Price: ₹120 (approx.)"):
            self.assertTrue(valid({"intent": "greeting", "reply": reply}), reply)

    def test_tool_and_clarify_cannot_mix(self):
        # L2 outputs like this ("ask_clarification" + a tool) must not validate.
        self.assertFalse(valid({"clarify": "Which field?", "tool": "scan_image", "args": {}}))

    def test_offline_schema_drops_network_tools(self):
        offline_tools = [n for n, t in SCHEMA["tools"].items() if not t["needsNetwork"]]
        offline = jsonschema.Draft202012Validator(T.understand_schema(SCHEMA, tools=offline_tools))
        weather = {"intent": "weather", "tool": "get_weather", "args": {"location": "Nashik"}, "reply": "x"}
        self.assertTrue(valid(weather))
        self.assertFalse(valid(weather, offline))
        self.assertTrue(valid({"intent": "offline", "reply": "No internet now."}, offline))


if __name__ == "__main__":
    unittest.main()
