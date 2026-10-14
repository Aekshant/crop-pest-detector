"""K1: kb/sources.csv covers every diagnosable label with authoritative sources.

  python -m unittest discover -s kb -p "test_*.py"
"""

import csv
import json
import re
import unittest
from pathlib import Path

KB = Path(__file__).resolve().parent
ROWS = list(csv.DictReader((KB / "sources.csv").open(encoding="utf-8")))
LABELS = [l["id"] for l in json.loads((KB.parent / "ml" / "labels_v1.json").read_text())["labels"]]
# Government, ICAR and state agricultural university domains only.
AUTHORITATIVE = ("ppqs.gov.in", "niphm.gov.in", "agritech.tnau.ac.in", "nbair.res.in", "icar.org.in", "iihr.res.in", "iivr.org.in")


class SourcesTest(unittest.TestCase):
    def test_every_problem_label_has_advice_ipm_and_registration_sources(self):
        for label in LABELS:
            if label in ("unknown", "healthy"):
                continue
            kinds = {r["kind"] for r in ROWS if r["label"] == label}
            self.assertIn("advisory", kinds, label)
            self.assertIn("ipm_package", kinds, label)
            self.assertIn("registration_list", kinds, label)

    def test_rows_are_complete_and_authoritative(self):
        ids = [r["id"] for r in ROWS]
        self.assertEqual(len(ids), len(set(ids)))
        for r in ROWS:
            self.assertIn(r["label"], LABELS, r["id"])
            self.assertTrue(r["url"].startswith("https://"), r["id"])
            host = re.sub(r"^https://(www\.)?", "", r["url"]).split("/")[0]
            self.assertTrue(any(host == d or host.endswith("." + d) for d in AUTHORITATIVE), f"{r['id']}: {host}")
            self.assertRegex(r["sha256"], r"^[0-9a-f]{64}$", r["id"])
            self.assertRegex(r["accessed"], r"^\d{4}-\d{2}-\d{2}$", r["id"])
            for field in ("publisher", "title", "topic"):
                self.assertTrue(r[field].strip(), f"{r['id']} {field}")

    def test_registration_lists_are_current_cibrc(self):
        for r in ROWS:
            if r["kind"] == "registration_list":
                self.assertTrue(r["url"].startswith("https://ppqs.gov.in/"), r["id"])
                self.assertGreaterEqual(r["doc_date"], "2026-01-01", r["id"])

    def test_same_url_same_hash(self):
        seen = {}
        for r in ROWS:
            self.assertEqual(seen.setdefault(r["url"], r["sha256"]), r["sha256"], r["id"])


if __name__ == "__main__":
    unittest.main()
