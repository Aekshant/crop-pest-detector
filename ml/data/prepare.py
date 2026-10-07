#!/usr/bin/env python3
"""Fetch the training datasets and build ml/data/manifest.csv.

  pip install pillow
  python ml/data/prepare.py            # fetch (idempotent) + build manifest
  python ml/data/prepare.py --no-fetch # rebuild the manifest from _downloads/

Sources (see README.md for licenses):
  PlantVillage  tomato, lab photos, leaf-grouped train/test split
  PlantDoc      tomato, field photos, train split only
  PlantVillage  other crops, a capped sample per class, as "unknown" negatives
  Web           iNaturalist / Wikimedia Commons photos from fetch_web_train.py
                (pests; other crops, weeds, soil, hands as "unknown")
  IP102         class 72 (Trialeurodes vaporariorum, greenhouse whitefly); optional

Images stay where they were downloaded (_downloads/, gitignored); the manifest
points at them. Anything in the vision eval sets is kept out: PlantDoc's test
split, PlantDoc train images used or rejected by evals/vision, and any image
whose perceptual hash is a near-duplicate of an eval image.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DL = ROOT / "_downloads"
REPO = ROOT.parents[1]
EVAL_VISION = REPO / "evals" / "vision"
MANIFEST = ROOT / "manifest.csv"

PLANTVILLAGE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTVILLAGE_COMMIT = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
PLANTVILLAGE_SPLITS = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_{}.txt"
PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"
IP102_URL = ("https://drive.usercontent.google.com/download?"
             "id=1EL9TA-J5XsiBR4M3nQkMR_e-2dP48eQN&export=download&confirm=t")
IP102_SIZE = 3250758656

LICENSES = {
    "plantvillage": "CC-BY-SA-3.0 (Hughes & Salathe 2015; Mohanty et al. 2016)",
    "plantdoc": "CC-BY-4.0 (Singh et al. 2020)",
    "ip102": "academic use only; other use needs permission (Wu et al. 2019)",
}

# Source class -> v1 label, from the one label map. Classes not listed are skipped.
_MAP = json.loads((ROOT.parent / "labels_v1.json").read_text())["source_map"]
PLANTVILLAGE_MAP = {k: v for k, v in _MAP["plantvillage"].items() if k != "note"}
PLANTDOC_MAP = {k: v for k, v in _MAP["plantdoc"].items() if k != "note"}
# labels_v1 uses IP102's 1-based "class N"; the split files use 0-based ids.
IP102_MAP = {int(k.split()[1]) - 1: v for k, v in _MAP["ip102"].items() if k != "note"}
OTHER_CROPS = _MAP["plantvillage_other_crops"]

NEAR_DUP_BITS = 6  # dHash Hamming distance at or below which images count as the same photo

FIELDS = ["path", "label", "source", "source_label", "split", "sha256", "license"]


# ---------- fetch ----------

def run(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def sparse_clone(url, commit, dest, patterns):
    if (dest / ".git").exists():
        return
    run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest))
    run("git", "sparse-checkout", "init", "--no-cone", cwd=dest)
    (dest / ".git" / "info" / "sparse-checkout").write_text("\n".join(patterns) + "\n")
    run("git", "checkout", "-q", commit, cwd=dest)


def download(url, dest):
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "farmlens-data/0.1"})
    dest.write_bytes(urllib.request.urlopen(req, timeout=120).read())


def download_ranged(url, dest, total):
    """Google Drive refuses whole-file downloads of popular files ("quota
    exceeded") but serves small byte ranges. The size it accepts moves over
    time, so halve the chunk when refused and grow it again after successes.
    Resumable: re-running continues from the end of the partial file."""
    lo, hi = 512 * 1024, 16 * 1024 * 1024
    chunk, failures = 4 * 1024 * 1024, 0
    with dest.open("ab") as f:
        while (pos := f.tell()) < total:
            end = min(pos + chunk, total) - 1
            data = None
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end}",
                                                           "User-Agent": "farmlens-data/0.1"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status == 206:
                        data = r.read()
            except OSError:
                pass
            if data is not None and len(data) == end - pos + 1:
                f.write(data)
                failures = 0
                chunk = min(chunk * 2, hi)
                print(f"  ip102 {f.tell() / total:.0%}", end="\r", flush=True)
                continue
            failures += 1
            if failures > 20:
                sys.exit(f"IP102 download stalled at byte {pos}; re-run later to resume")
            chunk = max(chunk // 2, lo)
            time.sleep(min(5 * failures, 60))
    print()


def extract_ip102(tar_path, dest):
    if (dest / "test.txt").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        splits = [m for m in members if m.name.endswith((".txt",)) and m.isfile()]
        tar.extractall(dest, members=splits)
        wanted = set()
        for m in splits:
            for line in (dest / m.name).read_text().splitlines():
                name, cls = line.split()
                if int(cls) in IP102_MAP:
                    wanted.add(name)
        tar.extractall(dest, members=[m for m in members if Path(m.name).name in wanted])
    # Flatten ip102_v1.1/ so paths are stable.
    inner = dest / "ip102_v1.1"
    if inner.exists():
        for p in inner.iterdir():
            p.rename(dest / p.name)
        inner.rmdir()


def fetch():
    DL.mkdir(exist_ok=True)
    print("PlantVillage (tomato)")
    sparse_clone(PLANTVILLAGE_REPO, PLANTVILLAGE_COMMIT, DL / "plantvillage", ["/raw/color/*"])
    for split in ("train", "test"):
        download(PLANTVILLAGE_SPLITS.format(split), DL / "plantvillage" / f"color_{split}.txt")
    print("PlantDoc (tomato)")
    sparse_clone(PLANTDOC_REPO, PLANTDOC_COMMIT, DL / "plantdoc", ["/train/Tomato*", "/test/Tomato*"])
    print("Web photos (pests, negatives)")
    subprocess.run([sys.executable, str(ROOT / "fetch_web_train.py")], check=True)
    print("IP102")
    tar = DL / "ip102_v1.1.tar"
    if not (DL / "ip102" / "test.txt").exists():
        try:
            download_ranged(IP102_URL, tar, IP102_SIZE)
            extract_ip102(tar, DL / "ip102")
        except SystemExit as e:
            print(f"WARNING: {e}; continuing without IP102")


# ---------- build ----------

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path):
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:
        return None
    px = list(g.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def near_dup(h, eval_hashes):
    return any(bin(h ^ e).count("1") <= NEAR_DUP_BITS for e in eval_hashes)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def candidates():
    """Yield (path, label, source, source_label, split, license)."""
    pv = DL / "plantvillage"
    split_of = {}
    for split in ("train", "test"):
        for line in (pv / f"color_{split}.txt").read_text().splitlines():
            split_of[line.strip()] = split
    for folder, label in PLANTVILLAGE_MAP.items():
        for p in sorted((pv / "raw" / "color" / folder).iterdir()):
            rel = p.relative_to(pv).as_posix()
            yield p, label, "plantvillage", folder, split_of.get(rel, "train"), LICENSES["plantvillage"]
    # Other crops' leaves are negatives for "unknown"; a stable sample per class.
    for d in sorted((pv / "raw" / "color").iterdir()):
        if d.name in PLANTVILLAGE_MAP or not d.is_dir():
            continue
        files = sorted(d.iterdir(), key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
        for p in files[:OTHER_CROPS["max_per_class"]]:
            rel = p.relative_to(pv).as_posix()
            yield p, OTHER_CROPS["label"], "plantvillage", d.name, split_of.get(rel, "train"), LICENSES["plantvillage"]

    pd = DL / "plantdoc"
    for folder, label in PLANTDOC_MAP.items():
        d = pd / "train" / folder
        if d.is_dir():
            for p in sorted(d.iterdir()):
                yield p, label, "plantdoc", folder, "train", LICENSES["plantdoc"]

    web = DL / "web" / "sources.csv"
    if web.exists():
        for r in read_csv(web):
            yield (DL / "web" / r["file"], r["label"], r["source"], r["page"], "train",
                   f"{r['license']} ({r['attribution']})")
    else:
        print("WARNING: no web photos yet (run fetch_web_train.py); manifest has no pest/negative web rows")

    ip = DL / "ip102"
    if not (ip / "test.txt").exists():
        print("WARNING: IP102 not downloaded yet; manifest has no IP102 rows (re-run to add them)")
        return
    for split in ("train", "val", "test"):
        for line in (ip / f"{split}.txt").read_text().splitlines():
            name, cls = line.split()
            if int(cls) in IP102_MAP:
                yield (ip / "images" / name, IP102_MAP[int(cls)], "ip102", f"class {int(cls) + 1}", split,
                       LICENSES["ip102"])


def build():
    eval_files = [EVAL_VISION / "field" / r["file"] for r in read_csv(EVAL_VISION / "field" / "labels.csv")]
    eval_files += [EVAL_VISION / "ood" / r["file"] for r in read_csv(EVAL_VISION / "ood" / "labels.csv")]
    eval_sha = {sha256(p) for p in eval_files}
    eval_hashes = [h for h in map(dhash, eval_files) if h is not None]
    blocked_sha = {r["sha256"] for r in read_csv(EVAL_VISION / "field_rejects.csv")}
    blocked_sha |= {line.split()[0] for line in (EVAL_VISION / "field_exclude_from_training.txt").read_text().splitlines()
                    if line and not line.startswith("#")}

    rows, dropped, seen = [], {"eval_exact": 0, "eval_near_dup": 0, "rejected_or_eval": 0,
                               "unreadable": 0, "duplicate": 0}, set()
    for path, label, source, source_label, split, license in candidates():
        if not path.is_file():
            dropped["unreadable"] += 1
            continue
        digest = sha256(path)
        if digest in eval_sha:
            dropped["eval_exact"] += 1
            continue
        if digest in blocked_sha:
            dropped["rejected_or_eval"] += 1
            continue
        if digest in seen:
            dropped["duplicate"] += 1
            continue
        h = dhash(path)
        if h is None:
            dropped["unreadable"] += 1
            continue
        if near_dup(h, eval_hashes):
            dropped["eval_near_dup"] += 1
            continue
        seen.add(digest)
        rows.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "source": source,
                     "source_label": source_label, "split": split, "sha256": digest,
                     "license": license})

    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    table = {}
    for r in rows:
        table.setdefault(r["label"], {}).setdefault(r["source"], 0)
        table[r["label"]][r["source"]] += 1
    sources = sorted({r["source"] for r in rows})
    print(f"{'label':24s} " + " ".join(f"{s:>12s}" for s in sources) + f" {'total':>7s}")
    for label in sorted(table):
        n = [table[label].get(s, 0) for s in sources]
        print(f"{label:24s} " + " ".join(f"{x:12d}" for x in n) + f" {sum(n):7d}")
    print(f"{'total':24s} {len(rows)}")
    print("dropped:", ", ".join(f"{k} {v}" for k, v in dropped.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
