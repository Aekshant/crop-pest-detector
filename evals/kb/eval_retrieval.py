#!/usr/bin/env python3
"""K4: top-3 hit rate of the KB pack on evals/kb/retrieval_queries.jsonl.

  uvx --with fastembed --with sqlite-vec python evals/kb/eval_retrieval.py [--pack kb/dist/kb_tomato_v1.sqlite]

Each mode also runs with local-name expansion ("+names"): when the query uses a
name from ml/labels_v1.json (Hindi name or alias, e.g. "agati jhulsa", "लाल मकड़ी"),
the English name is appended, as Laya's grounding already knows these names.

A query hits when one of the 3 nearest chunks is tagged with the expected label
AND contains one of the expected words (the relevance rule was written with the
queries, before any run). Two modes:

  open      search all chunks
  filtered  search only chunks of the expected label (the app usually knows the
            last diagnosis, so it can filter)

Writes evals/kb/results_<model>.json.
"""

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
K = 3


def norm(s):
    return unicodedata.normalize("NFD", s).lower().replace("\u093c", "")


def name_expander():
    labels = json.loads((REPO / "ml" / "labels_v1.json").read_text(encoding="utf-8"))["labels"]
    names = [(norm(n.split("(")[0].strip()), l["name_en"].split("(")[0].strip())
             for l in labels if l["id"] not in ("unknown", "healthy") for n in [l["name_hi"], *l["aliases"]]]

    def expand(q):
        found = {en for n, en in names if n and n in norm(q)}
        return q + (" (" + ", ".join(sorted(found)) + ")" if found else "")
    return expand


def relevant(chunk_label, text, expect):
    labels = (chunk_label or "").split(";")
    return expect["label"] in labels and any(w.lower() in text.lower() for w in expect["contains_any"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=REPO / "kb" / "dist" / "kb_tomato_v1.sqlite")
    args = ap.parse_args()
    import sqlite_vec
    from fastembed import TextEmbedding

    db = sqlite3.connect(args.pack)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    meta = dict(db.execute("SELECT key, value FROM meta"))
    model = TextEmbedding(meta["embedding_model"])
    queries = [json.loads(l) for l in (HERE / "retrieval_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = {i: (label, text) for i, label, text in db.execute("SELECT id, label, text FROM kb_chunk")}
    answerable = sum(any(relevant(l, t, q["expect"]) for l, t in chunks.values()) for q in queries)

    expand = name_expander()
    texts = [q["query"] for q in queries]
    vectors = {"": list(model.embed(texts)), "+names": list(model.embed([expand(t) for t in texts]))}
    results = []
    for i, q in enumerate(queries):
        row = {"id": q["id"], "lang": q["lang"], "label": q["expect"]["label"], "query": q["query"], "expanded": expand(q["query"])}
        for suffix, vs in vectors.items():
            v = vs[i] / (sum(x * x for x in vs[i]) ** 0.5)
            blob = sqlite_vec.serialize_float32([float(x) for x in v])
            ranked = [r for r, _ in db.execute("SELECT rowid, distance FROM vec_chunk WHERE embedding MATCH ? AND k = ?",
                                                (blob, len(chunks)))]
            for mode, ids in (("open", ranked[:K]),
                              ("filtered", [r for r in ranked if q["expect"]["label"] in (chunks[r][0] or "").split(";")][:K])):
                row[mode + suffix] = any(relevant(*chunks[c], q["expect"]) for c in ids)
                row[mode + suffix + "_top"] = chunks[ids[0]][1][:90] if ids else ""
        results.append(row)

    def rate(rows, mode):
        return sum(r[mode] for r in rows) / len(rows)

    report = {"model": meta["embedding_model"], "kb_version": meta["kb_version"], "k": K, "queries": len(queries),
              "answerable": answerable, "chunks": len(chunks)}
    MODES = ("open", "filtered", "open+names", "filtered+names")
    for mode in MODES:
        by_lang, by_label = defaultdict(list), defaultdict(list)
        for r in results:
            by_lang[r["lang"]].append(r)
            by_label[r["label"]].append(r)
        report[mode] = {"hit_rate": round(rate(results, mode), 3),
                        "by_lang": {k: round(rate(v, mode), 3) for k, v in sorted(by_lang.items())},
                        "by_label": {k: round(rate(v, mode), 3) for k, v in sorted(by_label.items())}}
    report["results"] = results
    out = HERE / f"results_{meta['embedding_model'].split('/')[-1]}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(f"K4 retrieval, {meta['embedding_model']}, top-{K}, {len(queries)} queries ({answerable} answerable from the pack)")
    print(f"{'':15s} {'all':>6s} {'en':>6s} {'hi':>6s} {'hinglish':>9s}")
    for mode in MODES:
        m = report[mode]
        print(f"{mode:15s} {m['hit_rate']:6.0%} " + " ".join(f"{m['by_lang'].get(l, 0):6.0%}" for l in ("en", "hi")) +
              f" {m['by_lang'].get('hinglish', 0):9.0%}")
    print("\nby label: " + " / ".join(MODES))
    for label in report["open"]["by_label"]:
        print(f"  {label:24s} " + " ".join(f"{report[m]['by_label'][label]:6.0%}" for m in MODES))
    print("\nmisses (filtered+names):")
    for r in results:
        if not r["filtered+names"]:
            print(f"  {r['id']} [{r['lang']}] {r['expanded'][:70]}\n      top: {r['filtered+names_top']}")
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
