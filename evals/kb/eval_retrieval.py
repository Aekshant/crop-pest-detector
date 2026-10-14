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
  hybrid    vector ranking fused with BM25 keyword ranking (reciprocal rank fusion)

Writes evals/kb/results_<model>.json.
"""

import argparse
import json
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
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


TOKEN = re.compile(r"[\w\u0900-\u097f]+")


class BM25:
    """Okapi BM25 over the chunk texts (k1=1.2, b=0.75)."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.ids = list(docs)
        self.tf = [Counter(TOKEN.findall(norm(docs[i]))) for i in self.ids]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.ids)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query):
        q = set(TOKEN.findall(norm(query)))
        scores = []
        for i, tf, ln in zip(self.ids, self.tf, self.len):
            s = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    for w in q if w in tf)
            scores.append((s, i))
        return [i for s, i in sorted(scores, key=lambda x: -x[0]) if s > 0]


def rrf(*rankings, k=60):
    """Reciprocal rank fusion."""
    score = Counter()
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += 1 / (k + r + 1)
    return [i for i, _ in score.most_common()]


def relevant(chunk_label, text, expect):
    labels = (chunk_label or "").split(";")
    return expect["label"] in labels and any(w.lower() in text.lower() for w in expect["contains_any"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=REPO / "kb" / "dist" / "kb_tomato_v1.sqlite")
    ap.add_argument("--fixture", type=Path, default=None,
                    help="also write query vectors and hybrid top-3 ids (the Android KbRetriever parity test reads it)")
    args = ap.parse_args()
    import sqlite_vec
    sys.path.insert(0, str(REPO / "kb"))
    import embedding

    db = sqlite3.connect(args.pack)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    meta = dict(db.execute("SELECT key, value FROM meta"))
    name = meta["embedding_model"]
    model = embedding.load(name)
    queries = [json.loads(l) for l in (HERE / "retrieval_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = {i: (label, text) for i, label, text in db.execute("SELECT id, label, text FROM kb_chunk")}
    answerable = sum(any(relevant(l, t, q["expect"]) for l, t in chunks.values()) for q in queries)

    expand = name_expander()
    texts = [q["query"] for q in queries]
    vectors = {"": embedding.embed(model, name, texts, "query"),
               "+names": embedding.embed(model, name, [expand(t) for t in texts], "query")}
    bm25 = BM25({i: t for i, (_, t) in chunks.items()})
    results = []
    for i, q in enumerate(queries):
        row = {"id": q["id"], "lang": q["lang"], "label": q["expect"]["label"], "query": q["query"], "expanded": expand(q["query"])}
        for suffix, vs in vectors.items():
            blob = sqlite_vec.serialize_float32(vs[i])
            ranked = [r for r, _ in db.execute("SELECT rowid, distance FROM vec_chunk WHERE embedding MATCH ? AND k = ?",
                                                (blob, len(chunks)))]
            hybrid = rrf(ranked, bm25.rank(expand(q["query"]) if suffix else q["query"]))
            in_label = lambda ids: [r for r in ids if q["expect"]["label"] in (chunks[r][0] or "").split(";")]
            if suffix:
                row["vector"] = vs[i]
                row["hybrid_ids"], row["hybrid_filtered_ids"] = hybrid[:K], in_label(hybrid)[:K]
            for mode, ids in (("open", ranked[:K]), ("filtered", in_label(ranked)[:K]),
                              ("hybrid", hybrid[:K]), ("hybrid_filtered", in_label(hybrid)[:K])):
                row[mode + suffix] = any(relevant(*chunks[c], q["expect"]) for c in ids)
                row[mode + suffix + "_top"] = chunks[ids[0]][1][:90] if ids else ""
        results.append(row)

    def rate(rows, mode):
        return sum(r[mode] for r in rows) / len(rows)

    report = {"model": meta["embedding_model"], "kb_version": meta["kb_version"], "k": K, "queries": len(queries),
              "answerable": answerable, "chunks": len(chunks)}
    MODES = ("open", "filtered", "open+names", "filtered+names", "hybrid+names", "hybrid_filtered+names")
    for mode in MODES:
        by_lang, by_label = defaultdict(list), defaultdict(list)
        for r in results:
            by_lang[r["lang"]].append(r)
            by_label[r["label"]].append(r)
        report[mode] = {"hit_rate": round(rate(results, mode), 3),
                        "by_lang": {k: round(rate(v, mode), 3) for k, v in sorted(by_lang.items())},
                        "by_label": {k: round(rate(v, mode), 3) for k, v in sorted(by_label.items())}}
    if args.fixture:
        args.fixture.write_text(json.dumps({"pack_built_at": meta["built_at"], "k": K, "queries": [
            {k: r[k] for k in ("id", "query", "label", "expanded", "vector", "hybrid_ids", "hybrid_filtered_ids")}
            for r in results]}) + "\n")
    for r in results:
        for k in ("vector", "hybrid_ids", "hybrid_filtered_ids"):
            r.pop(k, None)
    report["results"] = results
    out = HERE / f"results_{meta['embedding_model'].split('/')[-1]}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(f"K4 retrieval, {meta['embedding_model']}, top-{K}, {len(queries)} queries ({answerable} answerable from the pack)")
    print(f"{'':22s} {'all':>6s} {'en':>6s} {'hi':>6s} {'hinglish':>9s}")
    for mode in MODES:
        m = report[mode]
        print(f"{mode:22s} {m['hit_rate']:6.0%} " + " ".join(f"{m['by_lang'].get(l, 0):6.0%}" for l in ("en", "hi")) +
              f" {m['by_lang'].get('hinglish', 0):9.0%}")
    print("\nby label: " + " / ".join(MODES))
    for label in report["open"]["by_label"]:
        print(f"  {label:24s} " + " ".join(f"{report[m]['by_label'][label]:6.0%}" for m in MODES))
    print("\nmisses (hybrid_filtered+names):")
    for r in results:
        if not r["hybrid_filtered+names"]:
            print(f"  {r['id']} [{r['lang']}] {r['expanded'][:70]}\n      top: {r['hybrid_filtered+names_top']}")
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
