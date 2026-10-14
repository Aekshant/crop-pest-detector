#!/usr/bin/env python3
"""K3: build the on-device KB pack (sqlite + sqlite-vec).

  python kb/fetch_sources.py
  uvx --with fastembed --with sqlite-vec --with pdfplumber python kb/build_pack.py [--model ...]

Pack (kb/dist/kb_tomato_v1.sqlite, design doc "kb_chunk, kb_treatment; read-only,
replaced by KB pack updates"):

  meta          kb_version, embedding model, dimensions, build time
  kb_source     kb/sources.csv
  kb_treatment  kb/kb_treatment.csv (K2): the only source of products, doses, waiting periods
  kb_chunk      advisory text chunks (TNAU pages, NIPHM/NCIPM IPM package, ICAR-NBAIR note)
  vec_chunk     sqlite-vec vec0 table, rowid = kb_chunk.id, normalized embeddings
                (kb_chunk.embedding holds the same vectors as plain blobs: Android's
                SQLite can't load extensions, and brute force over ~200 chunks is <1 ms)

Chunks never carry doses: any sentence with a dose ("@ 2 g/l", "200 ml",
"0.2%") is dropped at build time, so retrieval can't put an advisory's
(old, possibly unregistered) dose in front of Laya. Doses come from
kb_treatment only (L9).

Needs pdftotext (poppler-utils) for the PDFs.
"""

import argparse
import csv
import html
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

KB = Path(__file__).resolve().parent
sys.path.insert(0, str(KB))
from fetch_sources import CACHE, cache_name  # noqa: E402

KB_VERSION = "2026-10-14"
DEFAULT_MODEL = "e5-small-int8"  # K4: e5-small beats MiniLM-L12 without a diagnosis; int8 = the on-device file
OUT = KB / "dist" / "kb_tomato_v1.sqlite"

LABEL_PATTERNS = {
    "early_blight": r"early\s*blight|alternaria",
    "late_blight": r"late\s*blight|phytoph?thora",
    "septoria_leaf_spot": r"septoria",
    "bacterial_spot": r"bacterial\s*(leaf\s*|fruits?\s*and\s*leaf\s*)?spot|xanthomonas",
    "yellow_leaf_curl_virus": r"leaf\s*curl",
    "mosaic_virus": r"mosaic",
    "tuta_absoluta": r"tuta|pin\s*worm",
    "whitefly": r"white\s*fl(y|ies)|bemisia",
    "fruit_borer": r"fruit\s*borer|helicoverpa|gram\s*pod\s*borer",
    "spider_mites": r"spider\s*mite|red\s*mite|tetranychus|\bmites?\b",
}
DOSE = re.compile(r"@|\d+(?:\.\d+)?\s*(?:%|ml|g|gm|gms|kg|l|lit|litre|liter|ppm|ec|sc|wp|wg|sl)\b|/\s*(?:l|lit|litre|acre|ha)\b"
                  r"|\bper\s+(?:litre|liter|acre|hectare)\b", re.I)
JUNK = re.compile(r"https?://|www\.|blogspot|\.com\b|^\s*\d+\s*$|^(fig|figure|table|plate)\b|crop production guide|"
                  r"photo cou|courtesy|professor|\bdr\.|department of|horticulture \d{4}|^source", re.I)
MAX_CHARS, MIN_CHARS = 600, 80
CARRY_PARAGRAPHS = 3  # paragraphs a pest heading's label applies to (PDF sections run on without end markers)


def labels_in(text):
    t = text.lower()
    return [l for l, p in LABEL_PATTERNS.items() if re.search(p, t)]


def sentences(text):
    text = re.sub(r"[•●–]\s*", ". ", text)
    return [s.strip(" .;:") for s in re.split(r"(?<=[.!?])\s+|\n(?=[A-Z])|\.\s", text) if s.strip(" .;:")]


def clean_sentences(text, stats):
    kept = []
    for s in sentences(re.sub(r"\s+", " ", text)):
        if JUNK.search(s):
            continue
        if DOSE.search(s):
            stats["dose_sentences_dropped"] += 1
            continue
        kept.append(s)
    return kept


def pack_sentences(sents, max_chars=MAX_CHARS, min_chars=30):
    chunk, out = "", []
    for s in sents:
        s = s if s.endswith((".", "!", "?")) else s + "."
        if chunk and len(chunk) + len(s) + 1 > max_chars:
            out.append(chunk)
            chunk = ""
        chunk = f"{chunk} {s}".strip()
    if chunk:
        out.append(chunk)
    return [c for c in out if len(c) >= min_chars]


HEADINGS = re.compile(r"^(symptoms?( of damage)?|field diagnostic symptoms|identification( of pest)?|management|"
                      r"favourable conditions|mode of spread.*|survival and spread|pathogen|etiology|epidemiology)\s*:?$", re.I)


def tnau_chunks(path, label, stats):
    """One TNAU page: line by line, one chunk per section (Symptoms, Management...)."""
    t = path.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"(?s)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</(p|li|tr|h\d|div|td)>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", l).strip() for l in t.splitlines()]
    lines = [l for l in lines if l]
    try:
        start = next(i for i, l in enumerate(lines) if re.search(r"^[A-Z][\w ]+\s*:\s*[A-Z][a-z]+ [a-z]", l))
    except StopIteration:
        return []
    name = lines[start].split("(")[0].strip()
    sections, heading = {}, "About"
    for l in lines[start + 1:]:
        if l.startswith("Updated on") or l.startswith("Home |"):
            break
        if HEADINGS.match(l):
            heading = l.rstrip(":").strip().capitalize()
            continue
        if len(l.split()) < 4:  # image captions: "Larvae", "Adult", "Pinhole in fruit"
            continue
        sections.setdefault(heading, []).append(l)
    out = []
    for heading, ls in sections.items():
        for c in pack_sentences(clean_sentences(". ".join(ls), stats)):
            out.append({"label": label, "text": f"{name}, {heading.lower()}: {c}"})
    return out


def pdf_chunks(path, stats, default_label=None):
    """A PDF as paragraphs; labels from the text itself or the last pest/disease heading."""
    text = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, check=True).stdout
    out, current, carry = [], default_label, 0
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        found = labels_in(p)
        if len(p) < 60:
            # A heading: "Early blight" sets the label for the next few paragraphs; any other
            # heading ("V. Rodent pest management", "Major kharif weeds") ends it.
            current, carry = (found[0], CARRY_PARAGRAPHS) if found else (default_label, 0)
            if found:
                continue
        sents = clean_sentences(p, stats)
        chunks = pack_sentences(sents, min_chars=MIN_CHARS)
        for c in chunks:
            labels = labels_in(c) or ([current] if current and (carry > 0 or current == default_label) else [])
            out.append({"label": ";".join(labels), "text": c})
        if chunks and carry > 0:
            carry -= 1
    return out


def prefixed(prefix, text, stats, label=None):
    """Chunks of one labelled passage, each starting with its heading so it reads on its own."""
    out = []
    for c in pack_sentences(clean_sentences(text, stats), max_chars=MAX_CHARS - len(prefix)):
        labels = labels_in(prefix) or labels_in(c) or ([label] if label else [])
        out.append({"label": ";".join(dict.fromkeys(labels)), "text": f"{prefix}: {c}"})
    return out


# NIPHM package sections: kept as general advice, or dropped as outside Laya's scope.
NIPHM_GENERAL = r"AESA based IPM|Ecological engineering|Insecticide resistance|Nutritional deficienc|Safety measures|" \
                r"Do.s and Don.ts|Safety parameters|Pesticide application techniques"
NIPHM_DESCRIPTIONS = r"Description of insect pests|Description of diseases"


def niphm_chunks(path, stats):
    """The NIPHM/NCIPM tomato IPM package, by its structure:
    tables (natural enemies, resistant varieties, crop stage-wise IPM) one row per chunk,
    'Description of ...' sections split at their numbered headings ("4) Early blight:"),
    general IPM and safety sections as paragraphs. Weeds, rodents, calibration,
    references and front matter are left out (outside Laya's scope)."""
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
        first = next(i for i, t in enumerate(texts) if "Flowering plants that attract natural enemies" in t)
        last = next(i for i, t in enumerate(texts) if i > first and "Rodent pest management" in t)
        stage = mgmt = None
        rows = []  # [stage, management, activity]
        for page in pdf.pages[first:last + 1]:
            for table in page.extract_tables():
                for raw in table:
                    cells = [re.sub(r"\s+", " ", c or "").strip() for c in raw]
                    if len(cells) == 4 and cells[1] and cells[1] != "Insect":  # natural enemies table
                        out += prefixed(f"Natural enemies of {cells[1]}", f"{cells[2]}. {cells[3]}", stats)
                    elif len(cells) == 2 and cells[0] and cells[0] != "Pest":  # resistant varieties
                        out += prefixed(f"Resistant or tolerant tomato varieties for {cells[0]}", cells[1], stats)
                    elif len(cells) == 3 and cells[2] and cells[0] != "Crop stage":
                        if cells[1]:
                            stage = cells[0] or stage
                            rows.append([stage, cells[1], cells[2]])
                        elif rows:  # continues the previous row (often across a page break)
                            rows[-1][2] += " " + cells[2]
        for stage, mgmt, activity in rows:
            stage_name = re.sub(r"\s*(stage)?\*?$", "", stage or "").strip()
            prefix = f"{mgmt} (tomato, {stage_name.lower()} stage)" if stage_name else mgmt
            out += prefixed(prefix, activity.replace("\uf0b7", ". "), stats)

    text = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, check=True).stdout
    # "*For management refer to page number----4) Early blight:" glues the next heading on.
    text = re.sub(r"\*?For management refer to page number-*\s*", "\n", text)
    heads = [(m.start(), m.group(2)) for m in re.finditer(r"(?m)^\s*([IVX]+)\.\s+(.+)$", text)]
    for (pos, title), nxt in zip(heads, heads[1:] + [(len(text), "")]):
        body = text[pos:nxt[0]]
        if len(body) < 300:  # table of contents entry
            continue
        if re.search(NIPHM_DESCRIPTIONS, title, re.I):
            items = re.split(r"(?m)^\s*\d+\)\s*([A-Z][^:\n]{2,60}):", body)
            for name, item in zip(items[1::2], items[2::2]):
                item = re.sub(r"\*For management refer.*", "", item)
                out += prefixed(name.strip(), item.replace("\uf0b7", ". "), stats)
        elif re.search(NIPHM_GENERAL, title, re.I):
            for para in re.split(r"\n\s*\n", body):
                out += [c for c in prefixed(title.strip().rstrip(":"), para.replace("\uf0b7", ". "), stats)
                        if len(c["text"]) - len(title) > MIN_CHARS]
    return out


def build_chunks(stats):
    sources = list(csv.DictReader((KB / "sources.csv").open(encoding="utf-8")))
    chunks, seen = [], set()
    for s in sources:
        if s["kind"] == "registration_list" or s["url"] in seen and s["kind"] != "advisory":
            continue
        path = CACHE / cache_name(s["url"])
        if not path.exists():
            raise SystemExit(f"missing {path}: run kb/fetch_sources.py first")
        if s["url"].endswith(".html"):
            new = tnau_chunks(path, s["label"], stats)
        elif s["url"] in seen:
            continue
        elif "IPM-Schedule-for-vegetables" in s["url"]:
            continue  # covers many crops with 2012 doses; the NIPHM tomato package supersedes it
        elif "niphm.gov.in" in s["url"]:
            new = niphm_chunks(path, stats)
        else:
            new = pdf_chunks(path, stats, s["label"] if "nbair" in s["url"] else None)
        seen.add(s["url"])
        for c in new:
            c.update(source_id=s["id"], url=s["url"], publisher=s["publisher"])
        chunks += new
        print(f"  {len(new):4d} chunks  {s['publisher'][:40]:40s} {s['title'][:50]}", flush=True)
    # Drop exact duplicates (repeated boilerplate across pages).
    uniq = {}
    for c in chunks:
        uniq.setdefault(c["text"], c)
    return list(uniq.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(__import__("embedding").MODELS))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out = args.out.resolve()

    import sqlite_vec

    import embedding

    stats = {"dose_sentences_dropped": 0}
    print("chunking sources", flush=True)
    chunks = build_chunks(stats)
    print(f"{len(chunks)} chunks; dropped {stats['dose_sentences_dropped']} sentences with doses", flush=True)

    print(f"embedding with {args.model}", flush=True)
    model = embedding.load(args.model)
    t = time.time()
    vectors = embedding.embed(model, args.model, [c["text"] for c in chunks], "passage")
    dim = len(vectors[0])
    print(f"  {len(vectors)} x {dim} in {time.time() - t:.0f} s", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    db = sqlite3.connect(args.out)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.executescript(f"""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE kb_source (id TEXT PRIMARY KEY, label TEXT, kind TEXT, publisher TEXT, title TEXT,
                                url TEXT, doc_date TEXT, accessed TEXT, sha256 TEXT);
        CREATE TABLE kb_treatment (id INTEGER PRIMARY KEY, label TEXT NOT NULL, product TEXT NOT NULL, dose TEXT NOT NULL,
                                   phi_days INTEGER NOT NULL, registration_status TEXT NOT NULL, source_url TEXT NOT NULL,
                                   kb_version TEXT NOT NULL);
        CREATE INDEX kb_treatment_label ON kb_treatment(label);
        CREATE TABLE kb_chunk (id INTEGER PRIMARY KEY, label TEXT, text TEXT NOT NULL, source_id TEXT NOT NULL,
                               source_url TEXT NOT NULL, kb_version TEXT NOT NULL,
                               embedding BLOB NOT NULL);  -- float32 little-endian unit vector, for readers without sqlite-vec
        CREATE VIRTUAL TABLE vec_chunk USING vec0(embedding float[{dim}] distance_metric=cosine);
    """)
    meta = {"kb_version": KB_VERSION, "embedding_model": args.model, "embedding_dim": str(dim),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "chunks": str(len(chunks)),
            "note": "Doses only in kb_treatment; chunks have dose sentences removed."}
    db.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    for s in csv.DictReader((KB / "sources.csv").open(encoding="utf-8")):
        db.execute("INSERT INTO kb_source VALUES (?,?,?,?,?,?,?,?,?)",
                   [s[k] for k in ("id", "label", "kind", "publisher", "title", "url", "doc_date", "accessed", "sha256")])
    for r in csv.DictReader((KB / "kb_treatment.csv").open(encoding="utf-8")):
        db.execute("INSERT INTO kb_treatment (label, product, dose, phi_days, registration_status, source_url, kb_version) "
                   "VALUES (?,?,?,?,?,?,?)", [r["label"], r["product"], r["dose"], int(r["phi_days"]),
                                              r["registration_status"], r["source_url"], r["kb_version"]])
    for i, (c, v) in enumerate(zip(chunks, vectors), 1):
        blob = sqlite_vec.serialize_float32(v)
        db.execute("INSERT INTO kb_chunk VALUES (?,?,?,?,?,?,?)", (i, c["label"], c["text"], c["source_id"], c["url"], KB_VERSION, blob))
        db.execute("INSERT INTO vec_chunk(rowid, embedding) VALUES (?, ?)", (i, blob))
    db.commit()
    db.execute("VACUUM")
    db.close()

    by_label = {}
    for c in chunks:
        for l in (c["label"] or "general").split(";"):
            by_label[l] = by_label.get(l, 0) + 1
    print(f"\n{args.out.relative_to(KB.parent)}: {args.out.stat().st_size / 2**20:.1f} MB")
    print("chunks per label: " + json.dumps(dict(sorted(by_label.items()))))


if __name__ == "__main__":
    main()
