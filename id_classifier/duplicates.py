"""Duplicate check: which NFS images are the same upload under different transaction ids.

Two kinds of copies:
- exact: byte-for-byte the same file (same sha256 of the zip's original.jpeg)
- near:  the same picture re-saved or re-compressed: different bytes, but the perceptual hashes differ in
         at most `max_distance` of 256 bits (see sources.perceptual_hash)

Duplicates make ground truth and reference sets look bigger than they are. The same picture in two
DIFFERENT folders means one of the two labels is wrong.

Files written to output_dir:
    summary.csv      per folder: zips, distinct images (exact only / exact + near), unreadable
    groups.csv       one row per picture uploaded more than once, with every upload
    distances.csv    histogram: how many image pairs are at perceptual distance 0, 1, 2, ... 256
    close_pairs.csv  every pair up to review_distance, flagged duplicate or not: to check the threshold
    review/          (only with review_images) each close pair side by side: REAL ID IMAGES, delete after
"""

from __future__ import annotations

import csv
import itertools
import logging
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from id_classifier.sources import (NEAR_DUPLICATE_DISTANCE, NfsZipSource, hash_distance, perceptual_hash,
                                   read_image_file)

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ("folder", "zips", "distinct_exact", "distinct_near", "unreadable")
GROUP_COLUMNS = ("image_hash", "copies", "kind", "max_distance", "folders", "cross_folder", "uploads")
DISTANCE_COLUMNS = ("distance", "pairs_same_folder", "pairs_cross_folder")
PAIR_COLUMNS = ("distance", "duplicate", "upload_a", "upload_b")
REVIEW_HEIGHT = 500          # each side of a review image is shrunk to this height
MAX_REVIEW_IMAGES = 300      # never write more than this many review images
PERCEPTUAL_BITS = 256        # bits in a perceptual hash (16 x 16)


def find_duplicates(
    root: str | Path,
    folders: list[str],
    output_dir: str | Path,
    max_distance: int = NEAR_DUPLICATE_DISTANCE,
    review_distance: int = 40,
    review_images: bool = False,
) -> Path:
    """Reads every zip of `folders` once (one image in memory at a time) and writes the files listed above.
    Returns output_dir."""
    # 1. one fingerprint per distinct file (sha256); each file remembers every upload that holds it
    uploads: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)   # sha -> [(folder, txn, side, path)]
    looks: dict[str, int] = {}                                                # sha -> perceptual hash
    unreadable: Counter = Counter()
    zips: Counter = Counter()
    for folder in folders:
        for transaction_id, image_id, path in NfsZipSource(root, [folder]).list_keys():
            zips[folder] += 1
            record = read_image_file(transaction_id, image_id, path)
            if record.image is None:
                unreadable[folder] += 1
                continue
            uploads[record.image_hash].append((folder, transaction_id, image_id, path))
            if record.image_hash not in looks:
                looks[record.image_hash] = perceptual_hash(record.image)
            record.image = None                                            # free the pixels
        log.info("Duplicates: %s read (%d zips)", folder, zips[folder])

    # 2. compare every pair of distinct files; near copies are joined into one group (union-find)
    parent = {sha: sha for sha in looks}

    def root_of(sha: str) -> str:
        while parent[sha] != sha:
            parent[sha] = parent[parent[sha]]
            sha = parent[sha]
        return sha

    histogram: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # distance -> [same folder, cross folder]
    close_pairs = []
    for a, b in itertools.combinations(sorted(looks), 2):
        distance = hash_distance(looks[a], looks[b])
        first_a, first_b = uploads[a][0], uploads[b][0]
        histogram[distance][first_a[0] != first_b[0]] += 1
        if distance <= max_distance:
            parent[root_of(a)] = root_of(b)
        if distance <= review_distance:
            close_pairs.append((distance, distance <= max_distance, first_a, first_b))
    close_pairs.sort(key=lambda pair: (pair[0], _name(pair[2]), _name(pair[3])))

    groups: dict[str, list[str]] = defaultdict(list)                 # group root -> member shas
    for sha in looks:
        groups[root_of(sha)].append(sha)

    # 3. reports
    summary_rows = []
    for folder in folders:
        exact = {sha for sha, ups in uploads.items() if any(u[0] == folder for u in ups)}
        near = {root_of(sha) for sha in exact}
        summary_rows.append([folder, zips[folder], len(exact), len(near), unreadable[folder]])

    group_rows = []
    for members in groups.values():
        group_uploads = sorted(u for sha in members for u in uploads[sha])
        if len(group_uploads) < 2:
            continue
        group_folders = sorted({u[0] for u in group_uploads})
        spread = max((hash_distance(looks[a], looks[b]) for a, b in itertools.combinations(members, 2)), default=0)
        group_rows.append([
            sorted(members)[0][:16], len(group_uploads), "exact" if len(members) == 1 else "near", spread,
            " ".join(group_folders), len(group_folders) > 1, " ".join(_name(u) for u in group_uploads),
        ])
    group_rows.sort(key=lambda row: (-row[1], row[0]))              # most copied first

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    _write(output_dir / "groups.csv", GROUP_COLUMNS, group_rows)
    _write(output_dir / "distances.csv", DISTANCE_COLUMNS,
           [[d, *histogram[d]] for d in range(PERCEPTUAL_BITS + 1) if d in histogram])
    _write(output_dir / "close_pairs.csv", PAIR_COLUMNS,
           [[d, dup, _name(a), _name(b)] for d, dup, a, b in close_pairs])
    if review_images:
        _write_review_images(close_pairs[:MAX_REVIEW_IMAGES], output_dir / "review")
    return output_dir



def _name(upload: tuple[str, int, str, str]) -> str:
    folder, transaction_id, image_id, _ = upload
    return f"{folder}/{transaction_id}_{image_id}"


def _write_review_images(close_pairs: list, review_dir: Path) -> None:
    """One jpg per close pair: both images side by side, the distance and verdict on top."""
    review_dir.mkdir(parents=True, exist_ok=True)
    for distance, duplicate, a, b in close_pairs:
        sides = []
        for folder, transaction_id, image_id, path in (a, b):
            image = read_image_file(transaction_id, image_id, path).image
            image.thumbnail((REVIEW_HEIGHT * 2, REVIEW_HEIGHT))
            sides.append(image)
        canvas = Image.new("RGB", (sides[0].width + sides[1].width + 10, REVIEW_HEIGHT + 40), "white")
        canvas.paste(sides[0], (0, 40))
        canvas.paste(sides[1], (sides[0].width + 10, 40))
        verdict = "DUPLICATE" if duplicate else "kept as different"
        ImageDraw.Draw(canvas).text((8, 8), f"distance {distance}: {verdict}   {_name(a)}  |  {_name(b)}",
                                    fill="black", font_size=18)
        label = "dup" if duplicate else "diff"
        name = f"{distance:03d}_{label}_{a[1]}_{a[2]}__{b[1]}_{b[2]}.jpg"
        canvas.save(review_dir / name, quality=85)


def _write(path: Path, columns: tuple[str, ...], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
