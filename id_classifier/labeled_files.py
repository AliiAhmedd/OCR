"""Ground truth from a folder of hand-labeled image files, e.g. ~/Downloads/DocumentData.

The label is in the file name:  <country>_<docType>_<side>_img[_<n>].<ext>
    egy_nid_front_img_2.jpg   -> national_id / EG / front
    uae_mrz_back_img.png      -> passport / AE / back      ("mrz" = the passport data page with the MRZ)
    egy_other_front_img.jpg   -> none / unknown / n/a      ("other" = blank photo or no document/card)

Front and back with the same country, document type and number are the same card, so they share one
(made-up) transaction id; the leave-one-out evaluation then knows they belong together.

Every problem is reported at once (unknown country, unknown document type, bad name), never skipped silently:
a mislabeled reference teaches the kNN classifier the wrong thing.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from id_classifier.types import GROUND_TRUTH_COLUMNS, NO_DOCUMENT, UNKNOWN, label_problems

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FIRST_TRANSACTION_ID = 800_000_001   # made-up ids for hand-labeled files; far from Core's real ids

# Country token in the file name -> ISO 3166-1 alpha-2 (the label the pipeline uses).
COUNTRY_CODES = {
    "egy": "EG", "egypt": "EG", "eg": "EG",
    "uae": "AE", "are": "AE", "ae": "AE",
    "ind": "IN", "india": "IN",
    "irq": "IQ", "iraq": "IQ",
    "nig": "NG", "nga": "NG", "nigeria": "NG",      # nig = Nigeria (NG), not Niger (NE)
    "syr": "SY", "syria": "SY",
    "tun": "TN", "tunisia": "TN",
    "mar": "MA", "morocco": "MA",
    "dza": "DZ", "alg": "DZ", "algeria": "DZ",
    "sdn": "SD", "sudan": "SD",
    "sau": "SA", "ksa": "SA",
    "jor": "JO", "jordan": "JO",
    "tur": "TR", "turkey": "TR",
    "lby": "LY", "libya": "LY",
    "yem": "YE", "yemen": "YE",
    "lbn": "LB", "lebanon": "LB",
    "pse": "PS", "palestine": "PS",
    "kwt": "KW", "qat": "QA", "omn": "OM", "bhr": "BH",
    "pak": "PK", "bgd": "BD", "phl": "PH", "som": "SO", "eri": "ER", "eth": "ET",
}

# Document-type token -> document_type of the taxonomy (types.DOCUMENT_TYPES).
DOCUMENT_TYPES = {
    "nid": "national_id", "id": "national_id", "nationalid": "national_id", "national_id": "national_id",
    "mrz": "passport", "passport": "passport", "pp": "passport",
    "rp": "residence_permit", "residence": "residence_permit", "resident": "residence_permit",
    "residence_permit": "residence_permit", "rid": "residence_permit",
    "dl": "driving_license", "license": "driving_license", "licence": "driving_license",
    "driver_license": "driving_license", "driving_license": "driving_license",
    "other": NO_DOCUMENT, "none": NO_DOCUMENT, "blank": NO_DOCUMENT,    # other = blank / no document
}

SIDES = {"front": "front", "back": "back"}

# Keep hand labels consistent with the NFS folder labels (ground_truth.FOLDER_LABELS), otherwise one card
# type ends up split across two classes. The UAE "Emirates ID" of a resident is filed as a residence permit
# there (folder UAE_RESIDENT_ID_OCR), so a file called uae_nid_... joins that class.
CLASS_OVERRIDES = {
    ("national_id", "AE"): "residence_permit",
}

_NAME = re.compile(r"^(?P<country>[a-z]+)_(?P<doctype>[a-z_]+?)_(?P<side>[a-z]+)_img(?:[_-]?(?P<n>\d+))?$", re.I)


@dataclass(frozen=True)
class LabeledFile:
    path: Path
    document_type: str
    issuing_country: str
    document_side: str
    card_key: tuple[str, str, str]    # (country token, docType token, n): front + back of one card share it
    override: str | None = None       # note when CLASS_OVERRIDES changed the label


def parse_file_name(path: Path) -> LabeledFile:
    """The labels in one file name. Raises ValueError saying exactly what is wrong."""
    match = _NAME.match(path.stem)
    if not match:
        raise ValueError(f"{path.name}: name is not <country>_<docType>_<side>_img[_<n>]")
    country_token, type_token, side_token = (match[g].lower() for g in ("country", "doctype", "side"))
    problems = []
    if type_token not in DOCUMENT_TYPES:
        problems.append(f"unknown docType {type_token!r} (known: {sorted(DOCUMENT_TYPES)})")
    if side_token not in SIDES:
        problems.append(f"unknown side {side_token!r} (known: {sorted(SIDES)})")
    document_type = DOCUMENT_TYPES.get(type_token)
    if document_type == NO_DOCUMENT:                       # no document: no country, no side
        country, side = UNKNOWN, "n/a"
    else:
        country, side = COUNTRY_CODES.get(country_token), SIDES.get(side_token)
        if country is None:
            problems.append(f"unknown country {country_token!r}: add it to labeled_files.COUNTRY_CODES")
    if problems:
        raise ValueError(f"{path.name}: {'; '.join(problems)}")

    override = None
    if (document_type, country) in CLASS_OVERRIDES:
        new_type = CLASS_OVERRIDES[(document_type, country)]
        override = f"{document_type}/{country} -> {new_type}/{country}"
        document_type = new_type
    problems = label_problems(document_type, country, side)
    if problems:
        raise ValueError(f"{path.name}: {'; '.join(problems)}")
    return LabeledFile(path, document_type, country, side, (country_token, type_token, match["n"] or ""), override)


def read_labeled_folder(folder: str | Path) -> list[LabeledFile]:
    """Every image file in the folder (not recursive). Raises one ValueError listing ALL bad names."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files, problems = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue                                        # not an image (e.g. desktop.ini)
        try:
            files.append(parse_file_name(path))
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("Cannot label these files:\n  " + "\n  ".join(problems))
    if not files:
        raise ValueError(f"No image files in {folder}")
    return files


def ground_truth_rows(files: list[LabeledFile]) -> list[list]:
    """Ground-truth CSV rows: one made-up transaction id per card, image_id = side (+ "-2", "-3" on clashes)."""
    card_ids = {key: FIRST_TRANSACTION_ID + i for i, key in enumerate(sorted({f.card_key for f in files}))}
    rows, used = [], Counter()
    for f in files:
        transaction_id = card_ids[f.card_key]
        base = f.document_side if f.document_side != "n/a" else "page"
        used[(transaction_id, base)] += 1
        image_id = base if used[(transaction_id, base)] == 1 else f"{base}-{used[(transaction_id, base)]}"
        rows.append([transaction_id, image_id, f.path.resolve().as_posix(), f.document_type, f.issuing_country,
                     f.document_side])
    return rows


def write_rows(out_csv: str | Path, rows: list[list]) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerows(rows)
    return out_csv


def merge_ground_truths(csv_paths: list[str | Path], out_csv: str | Path) -> Path:
    """Concatenates ground-truth CSVs. A (transaction_id, image_id) key or a file in two of them is an error."""
    rows, keys, paths = [], {}, {}
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as f:
            for number, row in enumerate(csv.DictReader(f), start=2):
                where = f"{csv_path} row {number}"
                key = (row["transaction_id"], row["image_id"])
                if key in keys:
                    raise ValueError(f"{where}: key {key} already in {keys[key]}")
                if row["image_path"] in paths:
                    raise ValueError(f"{where}: file {row['image_path']} already in {paths[row['image_path']]}")
                keys[key], paths[row["image_path"]] = where, where
                rows.append([row[c] for c in GROUND_TRUTH_COLUMNS])
    return write_rows(out_csv, rows)


def summarize(files: list[LabeledFile]) -> str:
    """Human-readable table: how many images per class, and which labels an override changed."""
    counts = Counter((f.document_type, f.issuing_country, f.document_side) for f in files)
    lines = [f"{len(files)} labeled images:"]
    lines += [f"  {t:<17} {c:<8} {s:<6} {n}" for (t, c, s), n in sorted(counts.items())]
    overrides = Counter(f.override for f in files if f.override)
    lines += [f"  (relabeled {n}x: {o})" for o, n in overrides.items()]
    return "\n".join(lines)
