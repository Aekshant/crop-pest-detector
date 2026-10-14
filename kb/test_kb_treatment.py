"""K2: kb_treatment.csv holds only what the CIB&RC lists print.

  python -m unittest discover -s kb -p "test_*.py"
"""

import csv
import json
import re
import unittest
from pathlib import Path

KB = Path(__file__).resolve().parent
ROWS = list(csv.DictReader((KB / "kb_treatment.csv").open(encoding="utf-8")))
RAW = list(csv.DictReader((KB / "cibrc_tomato.csv").open(encoding="utf-8")))
LABELS = {l["id"] for l in json.loads((KB.parent / "ml" / "labels_v1.json").read_text())["labels"]}
NUM = re.compile(r"\d+(?:\.\d+)?")


class KbTreatmentTest(unittest.TestCase):
    def test_rows_are_well_formed(self):
        self.assertGreater(len(ROWS), 100)
        for r in ROWS:
            self.assertIn(r["label"], LABELS - {"unknown", "healthy"})
            self.assertEqual(r["registration_status"], "registered")
            self.assertRegex(r["phi_days"], r"^\d+$")
            self.assertRegex(r["dose"], r"(?i)\d\s*(ml|g|gm|gms|kg|l|lit\w*)\b.* per hectare", r["product"])
            self.assertRegex(r["source_url"], r"^https://ppqs\.gov\.in/sites/default/files/.+\.pdf#page=\d+$")
        self.assertEqual(len({(r["label"], r["product"]) for r in ROWS}), len(ROWS))

    def test_every_number_is_printed_in_the_source_row(self):
        raw = {}
        for r in RAW:
            raw.setdefault(r["product"], []).append(NUM.findall(r["values"]))
        for r in ROWS:
            printed = [set(v) for v in raw[r["product"]]]
            used = set(NUM.findall(r["dose"].split(" per hectare")[0]) + NUM.findall(r["dose"].split(" in ")[-1]))
            used = {n for n in used if n in NUM.findall(r["dose"])}
            used.add(r["phi_days"])
            self.assertTrue(any(used <= {n.lstrip("0") or "0" for n in p} | p for p in printed),
                            f"{r['label']} {r['product']}: {used} not all in {printed}")

    def test_viruses_have_no_products(self):
        # Nothing is registered against leaf curl or mosaic: answers are vector control and roguing.
        for label in ("yellow_leaf_curl_virus", "mosaic_virus"):
            self.assertFalse([r for r in ROWS if r["label"] == label])

    def test_serpentine_leaf_miner_is_not_tuta(self):
        for r in ROWS:
            if r["label"] == "tuta_absoluta":
                self.assertRegex(r["pest_as_listed"].lower(), r"tuta|pinworm|pin worm")

    def test_safety_lexicon_does_not_ban_registered_products(self):
        lexicon = json.loads((KB.parent / "ml" / "bundle" / "laya_safety_v1.json").read_text())["chemicals_not_in_kb"]
        for chem in lexicon:
            hits = [r["product"] for r in ROWS if chem.lower() in r["product"].lower()]
            self.assertFalse(hits, f"{chem} is in the safety lexicon but registered: {hits[:2]}")


if __name__ == "__main__":
    unittest.main()
