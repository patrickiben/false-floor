"""Phase 1: build a PAIRED-TAXONOMY fungi image set from GBIF (verified API).

Two design goals at once:
  * KNOWN pairs in `train/` -> the FGVR discrimination problem (edible vs its toxic twin).
  * Entire toxic species HELD OUT in `ood/` -> the OPEN-SET abstention test (does the system
    abstain on a toxic species it never trained on, instead of calling it the nearest edible?).

GBIF gives image URL + lat/lon + eventDate (-> month). It does NOT reliably give substrate
(habitat came back None) -- that needs iNaturalist observation fields; left as a TODO.
NOTE: GBIF verifies the SPECIES id, not edibility -- the edibility label is OUR mapping below,
and "research grade" is an iNaturalist concept (filter via datasetKey if you need it strictly).

Usage:
    python fetch_fungi.py --dry-run                 # resolve taxa + counts, download nothing
    python fetch_fungi.py --per-species 150         # download into data/fungi/ + metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request

# (latin name, edibility, split). split='train' -> FGVR pair; split='ood' -> held-out (open-set).
SPECIES = [
    # --- known pairs the discriminator trains on ---
    ("Cantharellus cibarius",  "edible",    "train", "chanterelle"),
    ("Omphalotus illudens",    "poisonous", "train", "jack-o-lantern (false gills)"),
    ("Morchella esculenta",    "edible",    "train", "yellow morel (hollow)"),
    ("Gyromitra esculenta",    "poisonous", "train", "false morel (chambered)"),
    # --- the open-set test: train the EDIBLE, HOLD OUT its deadly toxic twin ---
    ("Agaricus campestris",    "edible",    "train", "field mushroom (pink->brown gills)"),
    ("Amanita bisporigera",    "poisonous", "ood",   "destroying angel (white gills + volva) -- NEVER trained"),
    # a held-out edible too, so 'ood' isn't all-poisonous
    ("Cantharellus lateritius","edible",    "ood",   "smooth chanterelle -- novel edible"),
]

GBIF = "https://api.gbif.org/v1"


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def taxon_key(name: str):
    m = _get(f"{GBIF}/species/match?name=" + urllib.parse.quote(name))
    return m.get("usageKey"), m.get("matchType")


def image_occurrences(key: int, limit: int):
    out, offset = [], 0
    while len(out) < limit:
        page = _get(f"{GBIF}/occurrence/search?taxonKey={key}&mediaType=StillImage"
                    f"&limit=100&offset={offset}")
        for r in page["results"]:
            media = (r.get("media") or [{}])[0]
            url = media.get("identifier")
            if not url:
                continue
            date = r.get("eventDate") or ""
            month = int(date[5:7]) if len(date) >= 7 and date[5:7].isdigit() else 0
            out.append(dict(url=url, lat=r.get("decimalLatitude"), lon=r.get("decimalLongitude"),
                            month=month))
            if len(out) >= limit:
                break
        if page["endOfRecords"] or not page["results"]:
            break
        offset += 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=150)
    ap.add_argument("--out", default="data/fungi")
    ap.add_argument("--dry-run", action="store_true", help="resolve + count only, no download")
    args = ap.parse_args()

    rows = []
    for name, edible, split, note in SPECIES:
        key, mt = taxon_key(name)
        if not key:
            print(f"  [skip] {name}: no GBIF match"); continue
        occ = image_occurrences(key, args.per_species)
        print(f"  {split:5} {edible:9} {name:28} key={key:<9} match={mt:<6} images={len(occ)}  ({note})")
        if args.dry_run:
            continue
        d = os.path.join(args.out, split, edible, name.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        for i, o in enumerate(occ):
            fp = os.path.join(d, f"{i:04d}.jpg")
            try:
                urllib.request.urlretrieve(o["url"], fp)
                rows.append([fp, name, edible, split, o["lat"], o["lon"], o["month"]])
            except Exception:
                pass
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "species", "edibility", "split", "lat", "lon", "month"])
            w.writerows(rows)
        print(f"\nwrote {len(rows)} images + metadata.csv to {args.out}/")
    print("\nSAFETY: GBIF verifies species id, NOT edibility (our mapping). Research vehicle only.")


if __name__ == "__main__":
    main()
