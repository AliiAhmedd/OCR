"""Duplicate check: which NFS images are byte-for-byte the same upload under different transaction ids.

Duplicates make ground truth and reference sets look bigger than they are (dza_nid_ocr: 62 zips, 7
distinct images). The same image in two DIFFERENT folders means one of the two labels is wrong.

The hash is the sha256 of the zip's original.jpeg, the same value the pipeline stores as image_hash.
Images are read in memory and never decoded or written anywhere.

hash of the zip is obtained by reading the zip member original.jpeg and hashing its bytes
"""

from __future__ import annotations

import csv
import hashlib
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

from id_classifier.sources import NfsZipSource, read_local_bytes, unpack_payload

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ("folder", "zips", "distinct_images", "duplicate_zips", "unreadable")
GROUP_COLUMNS = ("image_hash", "copies", "folders", "cross_folder", "uploads")


def find_duplicates(root: str | Path, folders: list[str], output_dir: str | Path) -> tuple[Path, Path]:
    """Hashes every zip in `folders` and writes summary.csv (one row per folder) and groups.csv (one row
    per image that appears more than once). Returns both paths. summary.csv includes the number of unreadable zips"""
    uploads_by_hash: dict[str, list[tuple[str, int, str]]] = defaultdict(list)   # hash -> [(folder, txn, side)]
    summary_rows = []
    for folder in folders:
        keys = NfsZipSource(root, [folder]).list_keys()
        hashes, unreadable = [], 0
        for transaction_id, image_id, path in keys:
            try:
                digest = hashlib.sha256(unpack_payload(read_local_bytes(path))).hexdigest()   # original.jpeg
            except (OSError, zipfile.BadZipFile, KeyError):
                unreadable += 1
                continue
            hashes.append(digest)
            uploads_by_hash[digest].append((folder, transaction_id, image_id))
        distinct = len(set(hashes))
        summary_rows.append([folder, len(keys), distinct, len(hashes) - distinct, unreadable])
        log.info("Duplicates: %s has %d zips, %d distinct images", folder, len(keys), distinct)

    group_rows = []
    for digest, uploads in uploads_by_hash.items():
        if len(uploads) < 2:
            continue
        group_folders = sorted({folder for folder, _, _ in uploads})
        group_rows.append([
            digest[:16], len(uploads), " ".join(group_folders), len(group_folders) > 1,
            " ".join(f"{folder}/{txn}_{side}" for folder, txn, side in sorted(uploads)),
        ])
    group_rows.sort(key=lambda row: (-row[1], row[0]))    # most copied first

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, groups_path = output_dir / "summary.csv", output_dir / "groups.csv"
    _write(summary_path, SUMMARY_COLUMNS, summary_rows)
    _write(groups_path, GROUP_COLUMNS, group_rows)
    return summary_path, groups_path


def _write(path: Path, columns: tuple[str, ...], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
