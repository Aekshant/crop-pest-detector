#!/usr/bin/env python3
"""Download every source in kb/sources.csv into kb/_cache/ and check it (K1).

  python kb/fetch_sources.py           # fetch, report status and sha256 changes
  python kb/fetch_sources.py --strict  # exit 1 if any URL fails or any file changed

A changed sha256 means the publisher updated the document: re-read it before
the next KB build (K2 extracts treatment rows from these files).
"""

import csv
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

KB = Path(__file__).resolve().parent
CACHE = KB / "_cache"


def cache_name(url):
    p = urllib.parse.urlparse(url)
    return (p.netloc + "_" + urllib.parse.unquote(p.path).strip("/")).replace("/", "_").replace(" ", "_")


def main():
    strict = "--strict" in sys.argv
    rows = list(csv.DictReader((KB / "sources.csv").open(encoding="utf-8")))
    CACHE.mkdir(exist_ok=True)
    expected = {}
    for r in rows:
        expected.setdefault(r["url"], r["sha256"])
    failed, changed = [], []
    for url, want in expected.items():
        dest = CACHE / cache_name(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (FarmLens KB builder)"})
            data = urllib.request.urlopen(req, timeout=180).read()
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append(url)
            print(f"FAIL     {url}  ({e})", flush=True)
            continue
        dest.write_bytes(data)
        got = hashlib.sha256(data).hexdigest()
        status = "ok" if got == want else "CHANGED"
        if got != want:
            changed.append(url)
        print(f"{status:8s} {len(data) / 1024:8.0f} KB  {url}", flush=True)
    print(f"\n{len(expected)} documents, {len(failed)} failed, {len(changed)} changed since {rows[0]['accessed']}")
    return 1 if strict and (failed or changed) else 0


if __name__ == "__main__":
    sys.exit(main())
