"""Detector sweep: how many images does each detector setting find a document in, per NFS folder?

Each detector (e.g. YOLOE with one prompt set) runs ONCE per image, at the lowest threshold. The counts
for the higher thresholds are then read from the confidences it returned, so adding a threshold costs
nothing. Writes numbers only, never images:
    sweep.md       detected / images per folder, prompt set and threshold (the table to read)
    sweep.csv      the same numbers, one row per folder x prompt set x threshold
    per_image.csv  every box confidence per image and prompt set, to look up single cases
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

from id_classifier.detectors import Detector
from id_classifier.sources import sample_nfs_records

log = logging.getLogger(__name__)

SWEEP_COLUMNS = ("prompt_set", "folder", "threshold", "images", "detected", "multiple", "detection_rate")
PER_IMAGE_COLUMNS = ("prompt_set", "folder", "transaction_id", "image_id", "image_hash", "confidences", "bboxes",
                     "latency_ms")


def sweep_detectors(
    detectors: dict[str, Detector],
    thresholds: list[float],
    root: str | Path,
    folders: list[str],
    output_dir: str | Path,
    per_folder: int = 20,
    seed: int = 42,
    unique_only: bool = True,
) -> Path:
    """Runs every detector on the same sampled images and writes the result files into a new
    <output_dir>/sweep_<timestamp>/ folder, which is returned. Each detector must already use the
    LOWEST threshold, otherwise the low-threshold counts miss boxes."""
    thresholds = sorted(thresholds)
    per_image_rows = []
    top_confidences: dict[tuple[str, str], list[list[float]]] = {}   # (set, folder) -> per image: confidences
    latencies: dict[str, list[float]] = {name: [] for name in detectors}
    for folder in folders:
        count = 0
        for record in sample_nfs_records(root, folder, per_folder, seed, unique_only):
            count += 1
            for name, detector in detectors.items():
                start = time.perf_counter()
                regions = detector.detect(record)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies[name].append(latency_ms)
                confidences = sorted((r.confidence for r in regions), reverse=True)
                top_confidences.setdefault((name, folder), []).append(confidences)
                per_image_rows.append([
                    name, folder, record.transaction_id, record.image_id, (record.image_hash or "")[:16],
                    " ".join(f"{c:.3f}" for c in confidences),
                    " ".join(",".join(map(str, r.bbox)) for r in regions), f"{latency_ms:.0f}",
                ])
            record.image = None              # free the pixels before the next image
        log.info("Sweep: %s done (%d distinct images)", folder, count)

    sweep_rows = []
    for name in detectors:
        for folder in folders:
            images = top_confidences.get((name, folder), [])
            for threshold in thresholds:
                detected = sum(1 for c in images if c and c[0] >= threshold)
                multiple = sum(1 for c in images if len([x for x in c if x >= threshold]) >= 2)
                rate = round(detected / len(images), 3) if images else None
                sweep_rows.append([name, folder, threshold, len(images), detected, multiple, rate])

    run_dir = Path(output_dir) / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "sweep.csv", SWEEP_COLUMNS, sweep_rows)
    _write(run_dir / "per_image.csv", PER_IMAGE_COLUMNS, per_image_rows)
    (run_dir / "sweep.md").write_text(_markdown(detectors, thresholds, folders, sweep_rows, latencies),
                                      encoding="utf-8")
    return run_dir


def _markdown(detectors, thresholds, folders, sweep_rows, latencies) -> str:
    """One table: a row per prompt set x folder, a column per threshold, cells "detected/images"."""
    cells = {(name, folder, t): (detected, images, multiple)
             for name, folder, t, images, detected, multiple, _ in sweep_rows}
    lines = ["# Detector sweep", "", "Cells: images with a detection / images sampled (images with 2+ boxes).", ""]
    for name, detector in detectors.items():
        mean = sum(latencies[name]) / len(latencies[name]) if latencies[name] else 0
        lines.append(f"- **{name}**: prompts {getattr(detector, 'prompts', '')}, mean {mean:.0f} ms/image")
    lines += ["", "| prompt set | folder | " + " | ".join(f">= {t}" for t in thresholds) + " |",
              "|" + " --- |" * (len(thresholds) + 2)]
    for name in detectors:
        for folder in folders:
            row = []
            for t in thresholds:
                detected, images, multiple = cells[(name, folder, t)]
                row.append(f"{detected}/{images}" + (f" ({multiple})" if multiple else ""))
            lines.append(f"| {name} | {folder} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _write(path: Path, columns: tuple[str, ...], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
