"""Turn the DocumentData drop into a labeled ground truth + a seedable folder tree.

The drop is a flat folder of JPEGs named
    <country>_<docType>_<side>_img[_<index>].jpg       e.g. egy_residencePermit_front_img_1.jpg
which carries all three labels the classifier predicts (type, country, side).

Two outputs:
1. a staging tree that mirrors the NFS service-folder layout, e.g.
       <staging>/EGY_RESIDENT_ID_OCR/egy_residencePermit_front_img_1.jpg
   so `seed_emulators.py --strip-prefix <staging>` produces the same key shape as the NFS zips
   (`storage_key()` gives "EGY_RESIDENT_ID_OCR/egy_...jpg").
2. a ground-truth CSV (GROUND_TRUTH_COLUMNS) pointing at the staged copies.

The trailing index pairs the two sides of ONE card: egy_..._front_img_1 and egy_..._back_img_1 are the
same document, so they share a transaction_id and differ only in image_id (= the side), exactly as the
NFS zips of one transaction do. A name without an index is its own document.

Unlike the NFS folders these are plain JPEGs, not zips. The pipeline reads either: record_from_payload()
sniffs the zip magic and falls through to a plain image (sources.py).

    python scripts/ingest_document_data.py --source ~/Downloads/DocumentData
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from id_classifier.types import GROUND_TRUTH_COLUMNS, label_problems

# <country>_<docType>_<side>_img[_<index>].(jpg|jpeg|png)
NAME = re.compile(
    r"^(?P<country>[a-z]{3})_(?P<doc_type>[A-Za-z]+)_(?P<side>front|back)_img(?:_(?P<index>\d+))?$"
)

# ISO 3166-1 alpha-3 (as used in the file names) -> alpha-2 (as stored in the labels).
COUNTRIES = {"egy": "EG", "ind": "IN", "irq": "IQ", "nig": "NG", "syr": "SY", "uae": "AE"}

# file-name docType -> document_type in DOCUMENT_TYPES. "mrz" (machine-readable zone) is a passport;
# "other" is a real Egyptian document whose type is not decided yet, so it keeps the generic label.
DOC_TYPES = {"id": "national_id", "residencePermit": "residence_permit", "mrz": "passport", "other": "other"}

# document_type -> NFS service-folder template. The share spells the residence-permit folder in upper
# case (UAE_RESIDENT_ID_OCR) and everything else in lower case (tun_nid_ocr, mar_driver_license_ocr).
FOLDERS = {
    "national_id": "{c}_nid_ocr",
    "passport": "{c}_passport_ocr",
    "residence_permit": "{C}_RESIDENT_ID_OCR",
    "other": "{c}_other_ocr",
}

FIRST_TRANSACTION_ID = 920000001   # far from the synthetic ids (9000000xx) and from real Core ids


def parse_name(stem: str) -> tuple[str, str, str, str]:
    """"egy_residencePermit_front_img_1" -> (country code, document_type, side, document key).

    The document key groups the sides of one card: same country + type + index = one transaction.
    """
    match = NAME.match(stem)
    if not match:
        raise ValueError(f"{stem!r} does not look like <country>_<docType>_<front|back>_img[_<n>]")
    country, doc_type_raw, side, index = match.group("country", "doc_type", "side", "index")
    if country not in COUNTRIES:
        raise ValueError(f"{stem!r}: unknown country {country!r} (known: {sorted(COUNTRIES)})")
    if doc_type_raw not in DOC_TYPES:
        raise ValueError(f"{stem!r}: unknown docType {doc_type_raw!r} (known: {sorted(DOC_TYPES)})")
    document_type = DOC_TYPES[doc_type_raw]
    # No index = the only card of its country+type in the drop, so its two sides still pair up
    # (egy_other_front_img + egy_other_back_img = one document). A second unindexed card of the same
    # country+type would wrongly merge into it, but the file names give nothing to tell them apart.
    return COUNTRIES[country], document_type, side, f"{country}/{document_type}/{index or 'single'}"


def service_folder(country_alpha2: str, document_type: str, country_alpha3: str) -> str:
    return FOLDERS[document_type].format(c=country_alpha3, C=country_alpha3.upper())


def build(source: Path, staging: Path, out_csv: Path) -> int:
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        print(f"no images found in {source}")
        return 1

    parsed = []
    for path in images:
        country, document_type, side, document_key = parse_name(path.stem)
        problems = label_problems(document_type, country, side)
        if problems:
            raise ValueError(f"{path.name}: {'; '.join(problems)}")
        parsed.append((path, country, document_type, side, document_key))

    # One transaction id per physical document, assigned in sorted order so re-runs are identical.
    ids = {key: FIRST_TRANSACTION_ID + i
           for i, key in enumerate(sorted({p[4] for p in parsed}))}

    rows = []
    for path, country, document_type, side, document_key in parsed:
        folder = service_folder(country, document_type, path.stem.split("_", 1)[0])
        staged = staging / folder / path.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists() or staged.stat().st_size != path.stat().st_size:
            shutil.copy2(path, staged)
        rows.append([ids[document_key], side, staged.resolve().as_posix(), document_type, country, side])

    rows.sort(key=lambda row: row[2])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerows(rows)

    per_class = Counter(f"{row[3]}/{row[4]}/{row[5]}" for row in rows)
    print(f"{len(rows)} images -> {out_csv}")
    print(f"staging tree: {staging.resolve()}")
    print(f"{len(ids)} documents (transaction ids {min(ids.values())}-{max(ids.values())})")
    for label, count in sorted(per_class.items()):
        print(f"  {label}: {count}")
    folders = Counter(row[2].rsplit("/", 2)[-2] for row in rows)
    print("service folders: " + ", ".join(f"{k} {v}" for k, v in sorted(folders.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the DocumentData folder")
    parser.add_argument("--staging", type=Path, default=Path("data/document_data"),
                        help="where the service-folder tree is written (default: data/document_data)")
    parser.add_argument("--out", type=Path, default=Path("data/document_data_ground_truth.csv"))
    args = parser.parse_args(argv)
    if not args.source.is_dir():
        print(f"source folder not found: {args.source}")
        return 1
    return build(args.source, args.staging, args.out)


if __name__ == "__main__":
    sys.exit(main())
