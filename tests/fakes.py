"""Test-only stand-ins (not part of the package): solid-colour images whose colour encodes the label,
and a classifier that reads the colour back. Lets the pipeline and harness run without a real model."""

from __future__ import annotations

import io
import random
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from id_classifier.classifiers.base import Classifier
from id_classifier.detectors import Detector
from id_classifier.types import STATUS_OK, STATUS_PARSE_ERROR, ImageRecord, Prediction, Region, make_region_id

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


class _Values(list):
    """A list with .tolist(), like the torch tensors in a real YOLO result."""

    def tolist(self):
        return list(self)


class FakeYoloModel:
    """Stands in for an Ultralytics model: predict() returns the boxes given here, in the same shape as
    a real result (result.boxes.xyxy / result.boxes.conf), and remembers the settings it was called with."""

    def __init__(self, boxes: list[tuple[float, float, float, float]], confidences: list[float]):
        self.boxes = boxes
        self.confidences = confidences
        self.calls: list[dict] = []

    def predict(self, image, **settings):
        self.calls.append({"image_size": image.size, **settings})
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=_Values(self.boxes), conf=_Values(self.confidences)))
        return [result]


def position_image(width: int, height: int) -> Image.Image:
    """Every pixel's colour is its own position: pixel (x, y) = (x, y, 0). A crop's top-left pixel
    therefore tells where in the original image the crop was taken from."""
    image = Image.new("RGB", (width, height))
    image.putdata([(x, y, 0) for y in range(height) for x in range(width)])
    return image


class FixedDetector(Detector):
    """Finds the same boxes in every image: one whole-image region per confidence given."""

    def __init__(self, confidences: list[float], name: str = "fixed"):
        self.name = name
        self.confidences = confidences
        self.prompts = ["fake prompt"]

    def detect(self, record: ImageRecord) -> list[Region]:
        if record.image is None:
            return []
        box = (0, 0, *record.image.size)
        return [Region(make_region_id(record.transaction_id, record.image_id, self.name, i), record.transaction_id,
                       record.image_id, box, c, self.name, crop=record.image) for i, c in enumerate(self.confidences)]


def picture(number: int, size: tuple[int, int] = (64, 48)) -> Image.Image:
    """A random 8x8-block grey pattern; the same number always gives the same picture, different numbers
    look clearly different (solid colours would all look alike to a perceptual hash)."""
    rng = random.Random(number)
    image = Image.new("RGB", size)
    for x in range(0, size[0], 8):
        for y in range(0, size[1], 8):
            grey = rng.randrange(256)
            image.paste((grey, grey, grey), (x, y, x + 8, y + 8))
    return image


def write_picture_zip(folder: Path, transaction_id: int, side: str, number: int, quality: int = 90) -> Path:
    """An NFS-shaped zip holding picture(number). Another `quality` = the same picture re-saved:
    different bytes (different sha256), same look (perceptual distance close to 0)."""
    folder.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    picture(number).save(buffer, "JPEG", quality=quality)
    path = folder / f"{transaction_id}_2026-10-01_15-29-25-873057_{side}_img.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("original.jpeg", buffer.getvalue())
    return path
