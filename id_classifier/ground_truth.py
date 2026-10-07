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


def build_ground_truth(
    root: str | Path,
    out_csv: str | Path,
    folders: dict[str, tuple[str, str]] | None = None,
    per_folder: int | None = None,
    seed: int = 42,
) -> Path:
    """Writes a ground-truth CSV for the labeled NFS folders. image_path is absolute (the root you pass in),
    so pass the root as the machine running `evaluate` sees it (in WSL: /mnt/nfs).

    per_folder: keep at most this many DISTINCT pictures per folder (random, same seed = same pictures).
    """
    folders = folders or FOLDER_LABELS
    rows = []
    for folder, (document_type, country) in folders.items():
        if not (Path(root) / folder).is_dir():
            log.warning("Ground truth: folder %s not found under %s, skipped", folder, root)
            continue
        for record in sample_nfs_records(root, folder, per_folder or 10**9, seed, unique_only=True):
            problems = label_problems(document_type, country, record.image_id)
            if problems:
                raise ValueError(f"{folder}: {'; '.join(problems)}")
            rows.append([record.transaction_id, record.image_id, record.image_reference, document_type, country,
                         record.image_id])
            record.image = None                                     # only the label is needed
    rows.sort(key=lambda row: (row[2]))                             # stable file order

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerows(rows)
    per_country = Counter(f"{row[3]}/{row[4]}" for row in rows)
    log.info("Ground truth: %d images written to %s %s", len(rows), out_csv, dict(per_country))
    return out_csv

