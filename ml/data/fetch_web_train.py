#!/usr/bin/env python3
"""Fetch CC-licensed web photos for training (pests and 'unknown' negatives).

  python ml/data/fetch_web_train.py      # resumable; then run prepare.py

Writes _downloads/web/<label>/<source>_<photo_id>.jpg and
_downloads/web/sources.csv (label, file, source, page, license, attribution).

Unlike evals/vision, these photos are NOT hand-screened: queries are chosen to
keep noise low (larval stage, on tomato where possible), and training is
tolerant to some label noise. Any photo from an observation used in the eval
sets is skipped; prepare.py also drops near-duplicates of eval images.
"""

import collections
import concurrent.futures
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "_downloads" / "web"
EVALS = ROOT.parents[1] / "evals" / "vision"
UA = {"User-Agent": "farmlens-data/0.1 (training set build)"}
INAT = ("https://api.inaturalist.org/v1/observations?quality_grade=research,needs_id"
        "&photo_license=cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa")
COMMONS = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
           "&gsrnamespace=6&gsrlimit=50&prop=imageinfo&iiprop=url|extmetadata|mime&iiurlwidth=640&gsrsearch=")

LARVA = "&term_id=1&term_value_id=6"
# label -> list of (iNaturalist query, max photos). Earlier queries are preferred.
INAT_QUERIES = {
    # Only leaf mines, larvae and photos on tomato: most other Tuta photos are adult
    # moths on walls or light traps, which is not what a farmer photographs.
    "tuta_absoluta": [("taxon_id=332007&term_id=22&term_value_id=32", 50), ("taxon_id=332007&q=tomato", 60),
                      ("taxon_id=332007" + LARVA, 100)],
    "whitefly": [("taxon_name=Trialeurodes%20vaporariorum&q=tomato", 80), ("taxon_id=199414", 60),
                 ("taxon_name=Trialeurodes%20vaporariorum", 300)],
    "fruit_borer": [("taxon_id=83806&q=tomato" + LARVA, 80), ("taxon_id=83806" + LARVA, 300)],
    "spider_mites": [("taxon_name=Tetranychus&q=tomato", 40), ("taxon_name=Tetranychus%20evansi", 80),
                     ("taxon_id=203333", 300)],
}
# Field photos of other crops and common weeds: all "unknown".
NEGATIVE_TAXA = [
    "Solanum tuberosum", "Solanum melongena", "Capsicum annuum", "Abelmoschus esculentus", "Gossypium hirsutum",
    "Oryza sativa", "Zea mays", "Cucumis sativus", "Brassica oleracea", "Phaseolus vulgaris", "Arachis hypogaea",
    "Mangifera indica", "Triticum aestivum", "Glycine max", "Cucurbita", "Allium cepa", "Spinacia oleracea",
    "Parthenium hysterophorus", "Amaranthus", "Solanum nigrum", "Datura", "Physalis",
]
NEGATIVES_PER_TAXON = 40
COMMONS_QUERIES = [
    ("soil", "agricultural soil close up filetype:bitmap", 50),
    ("soil", "ploughed field soil filetype:bitmap", 30),
    ("soil", "mulch garden bed filetype:bitmap", 20),
    ("hand", "open hand palm photograph filetype:bitmap", 40),
    ("hand", "hands holding filetype:bitmap", 30),
]
TOTAL_TARGET = {"tuta_absoluta": 210, "whitefly": 300, "fruit_borer": 300, "spider_mites": 300}


def get_json(url):
    # API answers are cached on disk so an interrupted run resumes instantly.
    cache = WEB / "api_cache" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):
        try:
            data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
            return data
        except OSError:
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"failed: {url}")


def eval_pages():
    """Observation/page URLs already used in evals: never train on them."""
    pages = set()
    for name, col in (("field_web.csv", "observation_url"), ("ood_sources.csv", "page")):
        with (EVALS / name).open(newline="") as f:
            pages |= {r[col].rstrip("/") for r in csv.DictReader(f)}
    return pages


def inat(query, limit, skip, by_votes=True):
    # Sorting by votes picks better photos but is very slow on huge taxa
    # (maize: ~90 s per page), so negatives use the default order and small pages.
    per_page = min(200, limit + 20)
    order = "&order_by=votes" if by_votes else ""
    out, page = [], 1
    while len(out) < limit and page <= 10:
        res = get_json(f"{INAT}{order}&per_page={per_page}&{query}&page={page}")["results"]
        time.sleep(1)
        if not res:
            break
        for o in res:
            if o["uri"].rstrip("/") in skip or not o["photos"]:
                continue
            p = o["photos"][0]
            if not p.get("license_code"):
                continue
            skip.add(o["uri"].rstrip("/"))
            out.append({"source": "inaturalist", "id": str(p["id"]), "page": o["uri"],
                        "url": p["url"].replace("square", "medium"),
                        "license": p["license_code"], "attribution": p["attribution"]})
            if len(out) >= limit:
                break
        page += 1
    return out


def commons(query, limit, skip):
    pages = get_json(COMMONS + urllib.parse.quote(query)).get("query", {}).get("pages", {})
    time.sleep(1)
    out = []
    for pg in pages.values():
        ii = pg["imageinfo"][0]
        if ii.get("mime") != "image/jpeg" or ii["descriptionurl"].rstrip("/") in skip:
            continue
        meta = ii["extmetadata"]
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not any(k in lic for k in ("CC0", "CC BY", "Public domain")):
            continue
        artist = re.sub("<[^>]+>", "", urllib.parse.unquote(meta.get("Artist", {}).get("value", ""))).strip()[:80]
        skip.add(ii["descriptionurl"].rstrip("/"))
        out.append({"source": "commons", "id": str(pg["pageid"]), "page": ii["descriptionurl"],
                    "url": ii["thumburl"], "license": lic, "attribution": artist})
        if len(out) >= limit:
            break
    return out


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    sources_csv = WEB / "sources.csv"
    done = {}
    if sources_csv.exists():
        with sources_csv.open(newline="") as f:
            done = {r["file"]: r for r in csv.DictReader(f)}
    # The plan is deterministic (API answers are cached), so files already
    # downloaded are recognised by name and skipped, not re-planned.
    skip = eval_pages()

    plan = []  # (label, item)
    for label, queries in INAT_QUERIES.items():
        have = sum(r["label"] == label for r in done.values())
        for q, n in queries:
            want = min(n, TOTAL_TARGET[label] - sum(l == label for l, _ in plan))
            if want > 0:
                plan += [(label, it) for it in inat(q, want, skip)]
        print(f"{label}: {sum(l == label for l, _ in plan)} planned, {have} already downloaded")
    for taxon in NEGATIVE_TAXA:
        plan += [("unknown", it) for it in inat("taxon_name=" + urllib.parse.quote(taxon), NEGATIVES_PER_TAXON, skip,
                                                by_votes=False)]
        print(f"  negatives: {taxon}", flush=True)
    for _, q, n in COMMONS_QUERIES:
        plan += [("unknown", it) for it in commons(q, n, skip)]
    print(f"downloading {len(plan)} photos")

    todo = [(label, it) for label, it in plan if f"{label}/{it['source']}_{it['id']}.jpg" not in done]
    print(f"{len(plan)} planned, {len(todo)} still to download", flush=True)

    failures = collections.Counter()

    def fetch(job):
        label, it = job
        rel = f"{label}/{it['source']}_{it['id']}.jpg"
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(urllib.request.Request(it["url"], headers=UA), timeout=60).read()
                break
            except OSError as e:
                err = e
                time.sleep(2 * (attempt + 1))
        else:
            failures[type(err).__name__ + " " + str(getattr(err, "code", ""))] += 1
            return None
        if not data.startswith(b"\xff\xd8\xff"):
            failures["not a JPEG"] += 1
            return None
        (WEB / rel).parent.mkdir(parents=True, exist_ok=True)
        (WEB / rel).write_bytes(data)
        return {"label": label, "file": rel, "source": it["source"], "page": it["page"],
                "license": it["license"], "attribution": it["attribution"]}

    with sources_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "file", "source", "page", "license", "attribution"])
        if not done:
            w.writeheader()
        ok = 0
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            for i, row in enumerate(ex.map(fetch, todo), 1):
                if row:
                    ok += 1
                    w.writerow(row)
                    f.flush()
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} tried, {ok} saved, failures: {dict(failures)}", flush=True)


if __name__ == "__main__":
    main()
