"""Ground truth from the NFS share: every service folder holds one document type of one country, and the
file name gives the side, so a labeled CSV for the evaluation harness can be built without hand labeling.

Only DISTINCT pictures are kept: the same upload saved again, or the same card photographed again
(perceptual distance <= 40, see sources.sample_nfs_records), appears once. Otherwise a card counted 27
times would make one country look 27 times bigger.

Caveat: a folder holds what was SUBMITTED to that service. Rejected uploads (e.g. a wrong document,
error 4201) end up in the same folder, so a few labels may be wrong. Spot-check the errors a model makes
before trusting a low score on one country.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import Counter
from pathlib import Path

from id_classifier.sources import sample_nfs_records
from id_classifier.types import GROUND_TRUTH_COLUMNS, label_problems


log = logging.getLogger(__name__)

# NFS folder (= Core service name) -> (document_type, issuing_country)
FOLDER_LABELS = {
    "tun_nid_ocr": ("national_id", "TN"),
    "mar_nid_ocr": ("national_id", "MA"),
    "dza_nid_ocr": ("national_id", "DZ"),
    "mar_driver_license_ocr": ("driving_license", "MA"),
    "UAE_RESIDENT_ID_OCR": ("residence_permit", "AE"),
}


EXCLUDED_IMAGES = Path("configs/excluded_images.csv")


def read_excluded(path: str | Path | None = EXCLUDED_IMAGES) -> dict[str, str]:
    """<folder>/<file> -> reason, for images that must never enter the ground truth (wrong label found by a
    human, e.g. a card of another country in a folder). The NFS share itself is never changed."""
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {row["key"].strip(): row["reason"] for row in csv.DictReader(f) if row.get("key", "").strip()}


RELABELLED_IMAGES = Path("configs/relabelled_images.csv")


def read_relabelled(path: str | Path | None = RELABELLED_IMAGES) -> dict[str, tuple[str, str, str]]:
    """<folder>/<file> -> (document_type, issuing_country, document_side) that replaces the folder's label, for
    images a human found to be something else (e.g. a selfie in an ID folder -> none/unknown/n/a)."""
    if path is None or not Path(path).exists():
        return {}
    labels = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("key", "").strip():
                continue
            label = (row["document_type"], row["issuing_country"], row["document_side"])
            problems = label_problems(*label)
            if problems:
                raise ValueError(f"{path}: {row['key']}: {'; '.join(problems)}")
            labels[row["key"].strip()] = label
    return labels


def build_ground_truth(
    root: str | Path,
    out_csv: str | Path,
    folders: dict[str, tuple[str, str]] | None = None,
    per_folder: int | None = None,
    seed: int = 42,
    excluded: str | Path | None = EXCLUDED_IMAGES,
    relabelled: str | Path | None = RELABELLED_IMAGES,
) -> Path:
    """Writes a ground-truth CSV for the labeled NFS folders. image_path is absolute (the root you pass in),
    so pass the root as the machine running `evaluate` sees it (in WSL: /mnt/nfs).

    per_folder: keep at most this many DISTINCT pictures per folder (random, same seed = same pictures).
    excluded: CSV of <folder>/<file> keys left out (read_excluded); None = leave nothing out.
    relabelled: CSV of <folder>/<file> keys whose label replaces the folder's (read_relabelled).
    """
    folders = folders or FOLDER_LABELS
    skip = read_excluded(excluded)
    new_labels = read_relabelled(relabelled)
    rows = []
    for folder, (document_type, country) in folders.items():
        if not (Path(root) / folder).is_dir():
            log.warning("Ground truth: folder %s not found under %s, skipped", folder, root)
            continue
        for record in sample_nfs_records(root, folder, per_folder or 10**9, seed, unique_only=True):
            key = f"{folder}/{Path(record.image_reference).name}"
            if key in skip:
                log.info("Ground truth: %s left out (%s)", key, skip[key])
                record.image = None
                continue
            problems = label_problems(document_type, country, record.image_id)
            if problems:
                raise ValueError(f"{folder}: {'; '.join(problems)}")
            label = new_labels.get(key, (document_type, country, record.image_id))
            rows.append([record.transaction_id, record.image_id, record.image_reference, *label])
            record.image = None                                     # only the label is needed
    rows.sort(key=lambda row: (row[2]))                             # stable file order

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerows(rows)
    per_country = Counter(f"{row[3]}/{row[4]}" for row in rows)
    if new_labels:
        log.info("Ground truth: %d relabelled from %s", sum(f"{Path(r[2]).parent.name}/{Path(r[2]).name}" in new_labels for r in rows), relabelled)
    log.info("Ground truth: %d images written to %s %s", len(rows), out_csv, dict(per_country))
    return out_csv

