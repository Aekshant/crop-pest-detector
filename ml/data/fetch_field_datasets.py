#!/usr/bin/env python3
"""Download public tomato FIELD datasets (real leaves, phone cameras).

  python ml/data/fetch_field_datasets.py    # resumable; then prepare.py --no-fetch

Lab photos (PlantVillage) are 80% of the training data, and field accuracy
lags far behind lab accuracy. These sets add real-world photos:

  pakistan      Mendeley 3mbnb82mxd v2, CC BY 4.0. 7,200 field photos, natural backgrounds.
  bangladesh    Mendeley jttrv2w27r v2, CC BY 4.0. Field photos incl. bacterial spot.
  fieldfruit    Mendeley swtt5wy2pr, CC BY 4.0. Field photos (resized set), leaves + fruit.
  tomatovillage github mamta-joshi-gehlot/Tomato-Village, NO LICENSE STATED (released
                publicly with Gehlot et al., Multimedia Systems 2023). Field leaves from
                Rajasthan, India, photographed on paper.

Leaf-miner classes are not downloaded: they show serpentine mines
(Liriomyza), not Tuta absoluta.
"""

import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DL = Path(__file__).resolve().parent / "_downloads"
UA = {"User-Agent": "farmlens-data/0.1"}
MENDELEY = "https://data.mendeley.com/public-api/datasets/{}"
MENDELEY_FILE = "https://data.mendeley.com/public-files/datasets/{}/files/{}/file_downloaded"

ZIPS = {  # name: (dataset id, filename)
    "pakistan": ("3mbnb82mxd", "Tomato_Leaf_Disease_Dataset_Pakistan.zip"),
    "fieldfruit": ("swtt5wy2pr", "Resize Dataset.zip"),
}
BANGLADESH = "jttrv2w27r"
BANGLADESH_CLASSES = {"Bacteiral Spot", "Healthy", "Early Blight", "Late Blight", "Curl", "Magnesium Deficiency"}
TOMATO_VILLAGE = "https://github.com/mamta-joshi-gehlot/Tomato-Village.git"
TOMATO_VILLAGE_COMMIT = "main"


def meta(dataset):
    return json.load(urllib.request.urlopen(urllib.request.Request(MENDELEY.format(dataset), headers=UA), timeout=60))


def stream(url, dest, total):
    """Resumable download with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    if total and pos >= total:
        return
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("ab" if pos else "wb") as f:
        if pos and r.status != 206:  # server ignored the range: start over
            f.seek(0)
            f.truncate()
            pos = 0
        last = time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            pos += len(chunk)
            if time.time() - last > 5:
                print(f"  {dest.name}: {pos / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB", flush=True)
                last = time.time()


def fetch_zip(name, dataset, filename):
    out = DL / name
    if (out / ".extracted").exists():
        return
    f = next(x for x in meta(dataset)["files"] if x["filename"] == filename)
    zpath = DL / "mendeley" / f"{name}.zip"
    stream(f["content_details"]["download_url"], zpath, f.get("size") or f["content_details"].get("size"))
    print(f"  extracting {zpath.name}", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    (out / ".extracted").write_text("")
    zpath.unlink()


def fetch_bangladesh():
    out = DL / "bangladesh"
    files = [f for f in meta(BANGLADESH)["files"]
             if f["filename"].rsplit("_", 1)[0] in BANGLADESH_CLASSES]
    todo = [f for f in files if not (out / f["filename"]).exists()]
    print(f"  bangladesh: {len(files) - len(todo)}/{len(files)} already downloaded", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    def get(f):
        for attempt in range(4):
            try:
                url = MENDELEY_FILE.format(BANGLADESH, f["id"])
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
                if data[:3] == b"\xff\xd8\xff":
                    (out / f["filename"]).write_bytes(data)
                    return True
            except OSError:
                time.sleep(2 * (attempt + 1))
        return False

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        ok = sum(ex.map(get, todo))
    print(f"  bangladesh: {ok}/{len(todo)} new files saved", flush=True)


def fetch_tomato_village():
    dest = DL / "tomatovillage"
    if (dest / ".git").exists():
        return
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", TOMATO_VILLAGE, str(dest)], check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest, check=True)
    (dest / ".git" / "info" / "sparse-checkout").write_text(
        "/Variant-a(Multiclass Classification)/*\n!/Variant-a(Multiclass Classification)/*/Leaf Miner/*\n")
    subprocess.run(["git", "checkout", "-q", TOMATO_VILLAGE_COMMIT], cwd=dest, check=True)


def main():
    jobs = {"tomatovillage": fetch_tomato_village, "bangladesh": fetch_bangladesh}
    jobs.update({name: (lambda n=name, d=d, fn=fn: fetch_zip(n, d, fn)) for name, (d, fn) in ZIPS.items()})
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as ex:
        futures = {ex.submit(fn): name for name, fn in jobs.items()}
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"{name}: done", flush=True)
            except Exception as e:  # keep the others going; re-run to retry
                print(f"{name}: FAILED ({e}); re-run to resume", flush=True)


if __name__ == "__main__":
    sys.exit(main())
