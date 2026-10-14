#!/usr/bin/env python3
"""Reference token ids for the Kotlin tokenizer parity test (EmbeddingModelTest).

  uvx --with tokenizers python kb/tokenizer_fixture.py

Every text the pack embeds ("passage: " + chunk), every eval question ("query: "
+ question, plain and name-expanded) and some edge cases, with the ids the
reference tokenizer (HF tokenizers, kb/dist/e5_small_tokenizer.json) produces.
"""

import json
import sqlite3
from pathlib import Path

from tokenizers import Tokenizer

KB = Path(__file__).resolve().parent
DIST = KB / "dist"
EDGE = ["", "  double  spaces  ", "Tab\\tand\\nnewline", "ＦＵＬＬ-width ｔｏｍａｔｏ", "Café naïve", "सफ़ेद मक्खी", "सफ़ेद मक्खी",
        "क़ ख़ ग़ ज़ ड़ ढ़ फ़", "emoji 🍅 tomato", "Ⅻ ½ ﬁ", "UPPER lower 123 4.5%", "a" * 40, "ज़्यादा।", "'quotes' “smart”"]


def main():
    tok = Tokenizer.from_file(str(DIST / "e5_small_tokenizer.json"))
    texts = ["passage: " + t for (t,) in sqlite3.connect(DIST / "kb_tomato_v1.sqlite").execute("SELECT text FROM kb_chunk ORDER BY id")]
    fixture = json.loads((DIST / "retrieval_fixture.json").read_text())
    for q in fixture["queries"]:
        texts += ["query: " + q["query"], "query: " + q["expanded"]]
    texts += EDGE
    out = [{"text": t, "ids": tok.encode(t).ids} for t in texts]
    (DIST / "tokenizer_fixture.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(out)} texts -> {DIST / 'tokenizer_fixture.json'}")


if __name__ == "__main__":
    main()
