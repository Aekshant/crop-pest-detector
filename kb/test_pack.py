"""K3: the built KB pack (kb/dist/kb_tomato_v1.sqlite). Skipped if it hasn't been built.

  uvx --with fastembed --with sqlite-vec python kb/build_pack.py
  uvx --with sqlite-vec python -m unittest discover -s kb -p "test_*.py"
"""

import csv
import sqlite3
import unittest
from pathlib import Path

KB = Path(__file__).resolve().parent
PACK = KB / "dist" / "kb_tomato_v1.sqlite"

try:
    import sqlite_vec
except ImportError:  # plain CI python: only the offline CSV tests run
    sqlite_vec = None


@unittest.skipUnless(PACK.exists() and sqlite_vec, "pack not built or sqlite-vec missing")
class PackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(PACK)
        cls.db.enable_load_extension(True)
        sqlite_vec.load(cls.db)

    def test_tables_and_meta(self):
        meta = dict(self.db.execute("SELECT key, value FROM meta"))
        self.assertEqual(meta["embedding_dim"], "384")
        n_csv = sum(1 for _ in csv.DictReader((KB / "kb_treatment.csv").open(encoding="utf-8")))
        self.assertEqual(self.db.execute("SELECT count(*) FROM kb_treatment").fetchone()[0], n_csv)
        chunks = self.db.execute("SELECT count(*) FROM kb_chunk").fetchone()[0]
        self.assertGreater(chunks, 100)
        self.assertEqual(self.db.execute("SELECT count(*) FROM vec_chunk").fetchone()[0], chunks)

    def test_every_problem_label_has_chunks(self):
        for label in ("early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot", "yellow_leaf_curl_virus",
                      "mosaic_virus", "tuta_absoluta", "whitefly", "fruit_borer", "spider_mites"):
            n = self.db.execute("SELECT count(*) FROM kb_chunk WHERE ';' || label || ';' LIKE ?", (f"%;{label};%",)).fetchone()[0]
            self.assertGreater(n, 0, label)

    def test_chunks_carry_no_doses(self):
        import sys
        sys.path.insert(0, str(KB))
        import build_pack  # noqa: PLC0415
        for (text,) in self.db.execute("SELECT text FROM kb_chunk"):
            self.assertIsNone(build_pack.DOSE.search(text), text)

    def test_vector_query_returns_nearest_first(self):
        rowid, blob = self.db.execute("SELECT rowid, embedding FROM vec_chunk LIMIT 1 OFFSET 5").fetchone()
        hits = self.db.execute("SELECT rowid, distance FROM vec_chunk WHERE embedding MATCH ? AND k = 3", (blob,)).fetchall()
        self.assertEqual(hits[0][0], rowid)
        self.assertAlmostEqual(hits[0][1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
