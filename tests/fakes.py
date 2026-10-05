"""Test-only stand-ins (not part of the package): solid-colour images whose colour encodes the label,
and a classifier that reads the colour back. Lets the pipeline and harness run without a real model."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

from PIL import Image

from id_classifier.classifiers.base import Classifier
from id_classifier.types import STATUS_OK, STATUS_PARSE_ERROR, Prediction

# colour -> (document_type, issuing_country, document_side)
COLOR_LABELS = {
    (255, 0, 0): ("national_id", "TN", "front"),
    (0, 255, 0): ("national_id", "TN", "back"),
    (0, 0, 255): ("national_id", "MA", "front"),
    (255, 255, 0): ("driving_license", "MA", "front"),
    (0, 0, 0): ("none", "unknown", "n/a"),          # no document
}
COLOR_OF = {label: color for color, label in COLOR_LABELS.items()}


def image_bytes(color: tuple[int, int, int], fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, fmt)
    return buffer.getvalue()


def write_nfs_zip(folder: Path, transaction_id: int, side: str, color: tuple[int, int, int]) -> Path:
    """A zip shaped like the NFS ones: <txn>_<date>_<time>_<side>_img.zip holding original.jpeg."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{transaction_id}_2026-10-01_15-29-25-873057_{side}_img.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("original.jpeg", image_bytes(color, "JPEG"))
    return path


class ColorClassifier(Classifier):
    """Answers with the label of the nearest colour in COLOR_LABELS (JPEG shifts colours slightly)."""

    def __init__(self, name: str = "color", wrong_country: str | None = None, parse_error: bool = False):
        self.name = name
        self.wrong_country = wrong_country   # answer this country for every document (a "bad model")
        self.parse_error = parse_error       # pretend every answer was broken JSON
        self.model_version = f"wrong_country={wrong_country},parse_error={parse_error}"

    def classify(self, crop: Image.Image) -> Prediction:
        start = time.perf_counter()
        if self.parse_error:
            return Prediction(None, None, None, None, STATUS_PARSE_ERROR, "<<broken>>", self.name, self.model_version,
                              0.0, error="simulated parse error")
        pixel = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
        color = min(COLOR_LABELS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel)))
        document_type, country, side = COLOR_LABELS[color]
        if self.wrong_country and document_type != "none":
            country = self.wrong_country
        return Prediction(document_type, country, side, 0.95, STATUS_OK, "{}", self.name, self.model_version,
                          (time.perf_counter() - start) * 1000)
