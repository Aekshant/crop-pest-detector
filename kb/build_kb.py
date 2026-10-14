#!/usr/bin/env python3
"""K2: tomato treatment rows from the CIB&RC "Major Uses of Pesticides" lists.

  python kb/fetch_sources.py          # the lists into kb/_cache/
  uvx --with pdfplumber python kb/build_kb.py

Writes:
  kb/cibrc_tomato.csv   every tomato row in the four lists, as printed (audit trail)
  kb/kb_treatment.csv   one row per (label, product) for FarmLens labels

Numbers are copied, never computed: the dose is the per-hectare formulation
quantity and water volume exactly as the list prints them (a per-litre or
per-pump conversion would be a generated number, which L9 forbids). A product
is `registered` only because it appears in a CIB&RC list for tomato against
that pest or disease.
"""

import csv
import re
import sys
from pathlib import Path

import pdfplumber

KB = Path(__file__).resolve().parent
CACHE = KB / "_cache"
KB_VERSION = "2026-10-14"

# Which pest/disease text maps to which FarmLens label. Serpentine leaf miner
# (Liriomyza) is NOT Tuta absoluta, so plain "leaf miner" maps to nothing.
LABEL_PATTERNS = {
    "early_blight": r"early\s*(&|and)?\s*(late\s*)?blight|alternaria",
    "late_blight": r"late\s*blight|phytophthora|early\s*(&|and)\s*late",
    "septoria_leaf_spot": r"septoria",
    "bacterial_spot": r"bacterial\s*(leaf\s*)?spot|xanthomonas",
    "tuta_absoluta": r"tuta|pin\s*worm|south\s*american",
    "whitefly": r"white\s*fl|bemisia",
    "fruit_borer": r"fruit\s*borer|helicoverpa",
    "spider_mites": r"mite",
}
LIQUID = {"EC", "SC", "SL", "OD", "SE", "CS", "EW", "ME", "ZC", "DC", "FS", "AS", "OL", "EO", "LIQUID", "L"}
# Doses on another basis (nursery trays, seed, trees) can't be read as per-hectare sprays.
SPECIAL_BASIS = r"seedling|/\s*kg\s*seed|kg seed|seed treat|nursery|per tree|/tree|sq\.?\s*m|soil application"
MIN_SPRAY_LITRES = 200
MIN_HA_DOSE = 25
SOLID = {"WP", "WG", "WDG", "SP", "SG", "G", "GR", "DP", "DF", "WS", "DS", "CG", "WSG", "D", "TAB", "POWDER", "WSP"}


def clean(cell):
    return re.sub(r"\s+", " ", cell or "").strip()


def is_product_header(cells):
    filled = [c for c in cells if c]
    return len(filled) == 1 and cells[0] and re.search(r"\d\s*%|%\s*w/|\b(EC|SC|WP|WG|SL|OD|SP|GR|DF|CS|SE|EW|ME|FS|DS|WS|ZC|DC)\b", cells[0])


def is_column_header(cells):
    return any(re.search(r"common name|dosage|waiting|crop\b", c or "", re.I) for c in cells[:3])


def formulation(product):
    m = re.search(r"\b([A-Z]{1,6})\s*\)?\s*$", product.replace(".", " ").strip())
    return m.group(1).upper() if m else ""


def extract(path, list_name):
    rows, product, crop, special = [], None, None, False
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                for raw in table:
                    cells = [clean(c) for c in raw] + ["", ""]
                    if not any(cells) or is_column_header(cells):
                        continue
                    if is_product_header(cells):
                        product, crop, special = cells[0], None, False
                        continue
                    if re.search(SPECIAL_BASIS, " ".join(cells), re.I):
                        special = True  # this product's doses aren't per hectare of spray

                    if cells[0]:
                        crop = cells[0]
                    if not crop or "tomato" not in crop.lower() or product is None:
                        continue
                    values = [c for c in cells[2:] if c]
                    rows.append({"list": list_name, "page": page_no, "product": product, "crop": crop,
                                 "pest": cells[1], "values": values, "special_basis": special})
    return rows


def label_for(pest):
    p = pest.lower()
    return [label for label, pat in LABEL_PATTERNS.items() if re.search(pat, p)]


def dose_text(product, values):
    """'<formulation qty> per hectare in <water> litres of water', copied from the list; None if unclear."""
    if len(values) < 4:
        return None, None
    _ai, form_qty, water, phi = values[-4:]
    if not re.search(r"\d", form_qty):
        return None, None
    qty = form_qty
    # Concentrations ("0.2%", "0.8 ml/Ltr") and per-ha granules ("10 Kg/Ha") don't fit the spray template.
    if re.search(r"%|/\s*(l|lt|ltr|lit|litre|ha)\b", qty, re.I) or formulation(product) in {"GR", "G", "CG"}:
        return None, None
    if not re.search(r"(?i)\d\s*(ml|g|gm|gms|kg|l|lit\w*)\b", qty):  # no unit: it follows from the formulation type
        f = formulation(product)
        unit = "ml" if f in LIQUID else "g" if f in SOLID else None
        if unit is None:
            return None, None
        qty = f"{qty} {unit}"
    # Per-hectare doses are tens of g/ml or more; "3 ml" is a per-litre figure printed in the wrong column.
    m = re.match(r"\s*([\d.]+)\s*(ml|g|gm|gms)?\b", qty, re.I)
    if m and (m.group(2) or not re.search(r"kg|l\b|lit", qty, re.I)) and float(m.group(1)) < MIN_HA_DOSE:
        return None, None
    if re.search(r"\d", water) and not re.search(r"[a-zA-Z]", water):
        dose = f"{qty} per hectare in {water} litres of water"
    else:
        dose = f"{qty} per hectare"
    # A range ("3-5") keeps its longest printed waiting period; "-" (none printed) can't be shown.
    m = re.fullmatch(r"0*(\d+)\s*(?:[-–]\s*0*(\d+))?\s*(days?)?", phi.strip(), re.I)
    return dose, int(m.group(2) or m.group(1)) if m else None


def main():
    lists = {
        "fungicide": "Fungicide%20MUP%20%20as%20on%2030.09.2026.pdf",
        "insecticide": "MUP%20%28insecticide%29%20%2030.09.2026.pdf",
        "bio_fungicide": "3.%20bio%20fungicides%20MUP%20as%20on%2030.09.2026.pdf",
        "bio_insecticide": "MUP%20%28bio%20insecticide%29%2030.09.2026.pdf",
    }
    sys.path.insert(0, str(KB))
    from fetch_sources import cache_name  # noqa: E402
    base = "https://ppqs.gov.in/sites/default/files/"
    all_rows = []
    for name, file in lists.items():
        path = CACHE / cache_name(base + file)
        if not path.exists():
            raise SystemExit(f"missing {path}: run kb/fetch_sources.py first")
        rows = extract(path, name)
        for r in rows:
            r["url"] = base + file
        print(f"{name:16s} {len(rows):3d} tomato rows", flush=True)
        all_rows += rows

    with (KB / "cibrc_tomato.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["list", "page", "product", "pest", "values", "labels", "special_basis"])
        for r in all_rows:
            w.writerow([r["list"], r["page"], r["product"], r["pest"], " | ".join(r["values"]), ";".join(label_for(r["pest"])), r["special_basis"]])

    out, skipped = {}, []
    for r in all_rows:
        for label in label_for(r["pest"]):
            dose, phi = dose_text(r["product"], r["values"])
            water = re.findall(r"\d+", r["values"][-2]) if len(r["values"]) >= 2 else []
            if water and int(water[0]) < MIN_SPRAY_LITRES:
                dose = None  # e.g. 10 L "per 1000 seedlings": not a field spray
            if r["special_basis"] or re.search(SPECIAL_BASIS, " ".join(r["values"]) + r["pest"], re.I):
                dose = None
            if dose is None or phi is None:
                skipped.append((label, r["product"], r["values"], r["list"], r["page"]))
                continue
            key = (label, r["product"])
            out.setdefault(key, {"label": label, "product": r["product"], "dose": dose, "phi_days": phi,
                                 "registration_status": "registered", "pest_as_listed": r["pest"],
                                 "source_url": f"{r['url']}#page={r['page']}", "kb_version": KB_VERSION})
    rows = sorted(out.values(), key=lambda x: (x["label"], x["product"].lower()))
    with (KB / "kb_treatment.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nkb_treatment.csv: {len(rows)} rows")
    for label in LABEL_PATTERNS:
        print(f"  {label:20s} {sum(r['label'] == label for r in rows)}")
    print(f"skipped (no waiting period printed, not a per-hectare spray, or unclear unit): {len(skipped)}")
    for s in skipped:
        print("   ", s)


if __name__ == "__main__":
    main()
