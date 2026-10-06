#!/usr/bin/env python3
"""Download the web images listed in field_web.csv into field/ and add them to labels.csv.

field_web.csv is the hand-screened list: iNaturalist observations with a CC
license, chosen as photos a farmer would take (pest or damage on the plant).

  python evals/vision/fetch_web.py

Idempotent: existing web rows are replaced, PlantDoc and own rows are kept.
"""

import csv
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "field"
LABELS_CSV = OUT_DIR / "labels.csv"
WEB_CSV = ROOT / "field_web.csv"
SOURCE = "inaturalist"

FIELDS = ["file", "label", "source", "source_class", "source_split",
          "source_path", "sha256", "license", "verified"]


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "farmlens-evals/0.1"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=90).read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    with LABELS_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["source"] != SOURCE]
    for p in OUT_DIR.glob(f"{SOURCE}_*"):
        p.unlink()

    with WEB_CSV.open() as f:
        web = list(csv.DictReader(f))

    for w in web:
        data = download(w["photo_url"])
        if not data.startswith(b"\xff\xd8\xff"):
            raise SystemExit(f"not a JPEG: {w['photo_url']}")
        digest = hashlib.sha256(data).hexdigest()
        photo_id = w["photo_url"].rstrip("/").split("/")[-2]
        name = f"{SOURCE}_{w['label']}_{photo_id}.jpg"
        (OUT_DIR / name).write_bytes(data)
        rows.append({
            "file": name, "label": w["label"], "source": SOURCE,
            "source_class": "", "source_split": "",
            "source_path": w["observation_url"], "sha256": digest,
            "license": f"{w['license']} ({w['attribution']})", "verified": "0",
        })
        time.sleep(0.5)

    rows.sort(key=lambda r: (r["label"], r["file"]))
    with LABELS_CSV.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=FIELDS)
        out.writeheader()
        out.writerows(rows)
    print(f"web images: {len(web)}, total rows: {len(rows)}")


if __name__ == "__main__":
    main()
