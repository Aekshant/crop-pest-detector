#!/usr/bin/env python3
"""Held-out retrieval questions from real Kisan Call Centre (KCC) logs.

  uvx --with pyarrow python -I evals/kb/make_heldout_kcc.py <kcc.parquet>

Source: KCC query logs (Department of Agriculture & Farmers Welfare, published on the
Open Government Data platform, data.gov.in), via the Hugging Face mirror
Omegaindebt/Kisan_Call_Centre_Transcripts. Only the question text, state and year of
the sampled rows are kept.

Every rule here was fixed before any retrieval run on these questions:
  - tomato, QueryType "Plant Protection"; duplicates (after normalizing) removed
  - label from fixed keyword patterns; questions matching no label or several are dropped
  - up to PER_LABEL questions per label, random with a fixed seed
  - relevance: a top-3 chunk carries the label; for control/prevention questions it must
    also contain management words (CONTROL_WORDS)
"""

import json
import random
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "heldout_kcc.jsonl"
PER_LABEL, SEED = 6, 20261014
LABELS = {
    "early_blight": r"early\s*blight|alternaria|अगेती",
    "late_blight": r"late\s*blight|phytophthora|पछेती",
    "septoria_leaf_spot": r"septoria",
    "bacterial_spot": r"bacterial\s*(leaf\s*)?spot",
    "yellow_leaf_curl_virus": r"leaf\s*curl|curling|पत्ती\s*मरोड|मरोडिया|leaf\s*roll",
    "mosaic_virus": r"mosaic|मोजेक",
    "tuta_absoluta": r"tuta|pin\s*worm",
    "whitefly": r"white\s*fl(y|ies)|सफेद\s*मक्खी",
    "fruit_borer": r"fruit\s*borer|helicoverpa|फल\s*छेदक|fruit\s*bor",
    "spider_mites": r"\bmites?\b|माइट|मकडी",
}
CONTROL = r"control|manage|prevent|treat|spray|medicine|remed|niyantran|रोकथाम|नियंत्रण|उपचार|दवा|बचाव"
CONTROL_WORDS = ["control", "remov", "destroy", "destruction", "rogu", "uproot", "sanitation", "rotation", "resistant",
                 "tolerant", "trap", "natural enem", "parasit", "predator", "neem", "spray", "avoid", "seed treatment",
                 "barrier", "border crop", "mulch", "intercrop", "marigold", "collect"]


def main():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(sys.argv[1], columns=["StateName", "Crop", "QueryType", "QueryText", "year"])
    t = t.filter(pc.and_(pc.match_substring(pc.utf8_lower(t.column("Crop")), "tomato"),
                         pc.match_substring(pc.utf8_lower(t.column("QueryType")), "protection")))
    seen, by_label = set(), {l: [] for l in LABELS}
    for row in t.to_pylist():
        q = re.sub(r"\s+", " ", row["QueryText"] or "").strip()
        key = re.sub(r"[^\wऀ-ॿ]+", " ", q.lower()).strip()
        if len(key) < 8 or key in seen:
            continue
        seen.add(key)
        labels = [l for l, p in LABELS.items() if re.search(p, q.lower())]
        if len(labels) == 1:
            by_label[labels[0]].append({"q": q, "state": row["StateName"], "year": row["year"]})
    rng = random.Random(SEED)
    out = []
    for label, items in by_label.items():
        for it in rng.sample(items, min(PER_LABEL, len(items))):
            control = bool(re.search(CONTROL, it["q"].lower()))
            out.append({"id": f"kcc-{len(out) + 1:03d}", "lang": "hi" if re.search(r"[ऀ-ॿ]", it["q"]) else "en",
                        "query": it["q"], "source": f"KCC {it['state']} {it['year']}",
                        "expect": {"label": label, "contains_any": CONTROL_WORDS if control else [""]}})
    OUT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print(f"{len(out)} questions -> {OUT}")
    print({l: f"{min(PER_LABEL, len(v))}/{len(v)}" for l, v in by_label.items()})


if __name__ == "__main__":
    main()
