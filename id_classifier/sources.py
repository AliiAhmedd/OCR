"""Image sources: where images come from.

Every source answers two questions:
- list_keys(): which images exist? -> (transaction_id, image_id, reference) tuples
- fetch(...):  give me one image    -> ImageRecord (a failed fetch is returned as a record, never raised)

In Airflow the keys will come from our database (ocr_error_4201 + the image-path mapping), so the
DAG only needs fetch(). list_keys() is for local folders and synthetic data.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageOps

from id_classifier.types import ImageRecord

log = logging.getLogger(__name__)

ImageKey = tuple[int, str, str]  # (transaction_id, image_id, reference)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FILENAME_PATTERN = re.compile(r"^(?P<transaction_id>\d+)_(?P<image_id>[A-Za-z0-9-]+)$")  # file stem like "123456_front"


class ImageSource(ABC):
    """Interface every image source implements."""

    @abstractmethod
    def list_keys(self) -> list[ImageKey]:
        """All images this source can provide."""

    @abstractmethod
    def fetch(self, transaction_id: int, image_id: str, reference: str) -> ImageRecord:
        """Load one image. On any problem return a record with fetch_status='failed' instead of raising."""


def failed_record(transaction_id: int, image_id: str, reference: str, reason: str) -> ImageRecord:
    return ImageRecord(transaction_id, image_id, reference, fetch_status="failed", failure_reason=reason)


def record_from_bytes(transaction_id: int, image_id: str, reference: str, data: bytes) -> ImageRecord:
    """Turns raw file bytes into an ImageRecord (hash + decoded RGB image). Shared by every source."""
    image_hash = hashlib.sha256(data).hexdigest()  # hash of the original bytes, before any decoding
    try:
        image = Image.open(io.BytesIO(data))
        image.load()                               # decode now, so a broken file fails here and not later
        image = ImageOps.exif_transpose(image)     # phone photos: apply the EXIF rotation
        image = image.convert("RGB")               # every detector/classifier gets the same colour mode
    except Exception as exc:                       # Pillow raises several error types for bad files
        return failed_record(transaction_id, image_id, reference, f"decode_error: {type(exc).__name__}")
    return ImageRecord(transaction_id, image_id, reference, image=image, image_hash=image_hash)


def read_image_file(transaction_id: int, image_id: str, path: str | Path) -> ImageRecord:
    """Reads one image file from disk."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return failed_record(transaction_id, image_id, str(path), f"read_error: {type(exc).__name__}")
    return record_from_bytes(transaction_id, image_id, str(path), data)


class LocalFolderSource(ImageSource):
    """Images in a folder, named <transaction_id>_<image_id>.<ext>, e.g. 123456_front.jpg."""

    def __init__(self, path: str | Path, recursive: bool = False):
        self.path = Path(path)
        self.recursive = recursive

    def list_keys(self) -> list[ImageKey]:
        if not self.path.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.path}")
        files = self.path.rglob("*") if self.recursive else self.path.glob("*")
        keys, skipped = [], 0
        for file in sorted(files):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue                                    # not an image (e.g. a CSV next to the images)
            match = FILENAME_PATTERN.match(file.stem)
            if not match:
                skipped += 1                                # an image, but the name does not tell us its key
                continue
            keys.append((int(match["transaction_id"]), match["image_id"], str(file)))
        log.info("LocalFolderSource: %d images found, %d skipped (name is not <transaction_id>_<image_id>)", len(keys), skipped)
        return keys

    def fetch(self, transaction_id: int, image_id: str, reference: str) -> ImageRecord:
        return read_image_file(transaction_id, image_id, reference)


class BlobSource(ImageSource):
    """Images in the company blob storage. NOT IMPLEMENTED YET: provider and path mapping not confirmed.

    TODO once access is granted:
    1. Confirm the provider (S3 / Azure Blob / GCS / MinIO / other) and add its SDK as an optional extra.
    2. Read credentials from an Airflow connection or environment variables, never from code or YAML.
    3. fetch(): download the object at `reference` into memory (no temp files with ID images on disk)
       and return record_from_bytes(...). Return failed_record(...) for "not found", "access denied", timeouts.
    4. list_keys(): probably not needed in production; the keys come from the transaction -> image-path
       mapping query in configs/pipeline.yaml (e.g. front_url / back_url in services_egyptiannationalid).
    """

    def __init__(self, **settings):
        self.settings = settings  # e.g. bucket / container name, connection id; kept for the future implementation

    def list_keys(self) -> list[ImageKey]:
        raise NotImplementedError("BlobSource.list_keys: image keys come from the database mapping, see the class docstring")

    def fetch(self, transaction_id: int, image_id: str, reference: str) -> ImageRecord:
        raise NotImplementedError("BlobSource.fetch: blob provider not confirmed yet, see the TODOs in the class docstring")
