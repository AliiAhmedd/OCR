"""Visual check of a detector: run it on a few NFS images and save what it found, for a human to look at.

For every sampled image it writes, under <output_dir>/<folder>/:
    <txn>_<side>_boxes.jpg    the image with every box drawn and labelled "<index>: <confidence>"
    <txn>_<side>_crop<i>.jpg  each crop exactly as a classifier will receive it
and one summary.csv with the boxes, confidences and timing of every image.

PRIVACY: unlike the pipeline, this DOES write (copies of) real ID images to disk. Keep output_dir under
data/ (gitignored) and delete it after the check.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw

from id_classifier.detectors import Detector
from id_classifier.sources import sample_nfs_records
from id_classifier.types import Region

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ("folder", "transaction_id", "image_id", "image_hash", "image_width", "image_height", "regions",
                   "confidences", "bboxes", "latency_ms")
BOX_COLOURS = ("red", "lime", "cyan", "yellow", "magenta")   # region 0 red, region 1 green, ...
MAX_PREVIEW_SIDE = 1600                                       # shrink big photos so the files stay small


def preview_detector(
    detector: Detector,
    root: str | Path,
    folders: list[str],
    output_dir: str | Path,
    per_folder: int = 5,
    seed: int = 42,
    unique_only: bool = True,
) -> Path:
    """Runs `detector` on `per_folder` random images of each folder and writes the preview files.
    Samples exactly like the detector sweep (same seed = same images, duplicates skipped).
    Returns the path of summary.csv."""
    output_dir = Path(output_dir)
    rows = []
    for folder in folders:
        folder_dir = output_dir / folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for record in sample_nfs_records(root, folder, per_folder, seed, unique_only):
            count += 1
            start = time.perf_counter()
            regions = detector.detect(record)
            latency_ms = (time.perf_counter() - start) * 1000
            stem = f"{record.transaction_id}_{record.image_id}"
            draw_boxes(record.image, regions).save(folder_dir / f"{stem}_boxes.jpg", quality=85)
            for index, region in enumerate(regions):
                region.crop.save(folder_dir / f"{stem}_crop{index}.jpg", quality=90)
            width, height = record.image.size
            rows.append([
                folder, record.transaction_id, record.image_id, (record.image_hash or "")[:16], width, height,
                len(regions), " ".join(f"{r.confidence:.3f}" for r in regions),
                " ".join(",".join(map(str, r.bbox)) for r in regions), f"{latency_ms:.0f}",
            ])
            record.image = None                   # free the pixels before the next image
        log.info("Preview: %s done (%d distinct images)", folder, count)

    summary = output_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)
    return summary


def draw_boxes(image: Image.Image, regions: list[Region]) -> Image.Image:
    """A copy of the image with each region's box and "<index>: <confidence>" drawn on it."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, round(max(canvas.size) / 300))          # thicker lines on bigger photos
    for index, region in enumerate(regions):
        colour = BOX_COLOURS[index % len(BOX_COLOURS)]
        draw.rectangle(region.bbox, outline=colour, width=line)
        x0, y0 = region.bbox[0], region.bbox[1]
        draw.text((x0 + line, y0 + line), f"{index}: {region.confidence:.2f}", fill=colour,
                  font_size=max(14, line * 8))
    canvas.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE))
    return canvas
