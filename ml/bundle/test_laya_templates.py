"""L8: Laya's templates cover every label, slot, reply and safety case in three
languages, use the right script, and never contain numbers of their own.

  python -m unittest discover -s ml/bundle -p "test_*.py"
"""

import json
import re
import unittest
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[1]
T = json.loads((BUNDLE / "laya_templates_v1.json").read_text())
LANGS = T["languages"]


def texts(node, path=""):
    """Yield (path, lang, text) for every template string."""
    if isinstance(node, dict) and set(node) == set(LANGS):
        for lang in LANGS:
            yield path, lang, node[lang]
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from texts(v, f"{path}.{k}" if path else k)


def all_texts():
    for section in ("diagnosis", "severity_levels", "clarify", "respond", "tools", "fallback", "safety", "kb"):
        yield from texts(T[section], section)
    for label, by_lang in T["labels"].items():
        for lang in LANGS:
            for field in ("name", "about"):
                yield f"labels.{label}.{field}", lang, by_lang[lang][field]


class TemplatesTest(unittest.TestCase):
    def test_every_label_in_every_language(self):
        labels = [l["id"] for l in json.loads((REPO / "ml" / "labels_v1.json").read_text())["labels"]]
        self.assertEqual(sorted(T["labels"]), sorted(labels))
        for label in labels:
            for lang in LANGS:
                self.assertTrue(T["labels"][label][lang]["name"] and T["labels"][label][lang]["about"], (label, lang))

    def test_every_section_entry_in_every_language(self):
        for path, lang, text in all_texts():
            self.assertTrue(text.strip(), (path, lang))

    def test_covers_state_machine_and_safety(self):
        self.assertEqual(set(T["clarify"]), {"when", "text", "label", "field", "location", "photo"})
        kinds = set(json.loads((BUNDLE / "tools_schema_v1.json").read_text())["respond_kinds"])
        self.assertTrue(kinds <= set(T["respond"]), kinds - set(T["respond"]))
        self.assertEqual(set(T["fallback_reasons"]),
                         {"TOOL_LIMIT", "CLARIFY_LIMIT", "TIME_BUDGET", "MODEL_ERROR", "INVALID_TOOL_CALL", "TOOL_FAILED"})
        self.assertTrue(set(T["fallback_reasons"].values()) <= set(T["fallback"]))
        safety = json.loads((REPO / "evals" / "laya" / "safety_templates.json").read_text())["templates"]
        self.assertEqual(set(T["safety"]), set(safety))

    def test_no_numbers_except_through_placeholders(self):
        for path, lang, text in all_texts():
            without = re.sub(r"\{[a-z_]+\}", "", text)
            self.assertIsNone(re.search(r"[0-9०-९]", without), f"{path} [{lang}] has a digit: {text}")

    def test_placeholders_are_known(self):
        for path, lang, text in all_texts():
            for p in re.findall(r"\{([^}]*)\}", text):
                self.assertIn(p, T["placeholders"], f"{path} [{lang}]")
        # Same placeholders in every language of an entry.
        groups = {}
        for path, lang, text in all_texts():
            groups.setdefault(path, {})[lang] = set(re.findall(r"\{([a-z_]+)\}", text))
        for path, by_lang in groups.items():
            self.assertEqual(len({frozenset(v) for v in by_lang.values()}), 1, f"{path}: {by_lang}")

    def test_scripts(self):
        for path, lang, text in all_texts():
            plain = re.sub(r"\{[a-z_]+\}", "", text)
            dev = len(re.findall(r"[ऀ-ॿ]", plain))
            latin = len(re.findall(r"[A-Za-z]", plain))
            if dev + latin == 0:
                continue  # placeholders only, e.g. "{name}: {about}"
            if lang == "hi":
                self.assertGreater(dev, latin, f"{path} [hi] should be Devanagari: {text}")
            else:
                self.assertEqual(dev, 0, f"{path} [{lang}] should be Latin script: {text}")


if __name__ == "__main__":
    unittest.main()
