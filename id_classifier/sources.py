"""Image sources: where images come from.

Every source answers two questions:
- list_keys(): which images exist? -> (transaction_id, image_id, reference) tuples
- fetch(...):  give me one image    -> ImageRecord (a failed fetch is returned as a record, never raised)

Images on the NFS share are stored one per zip (see NfsZipSource). read_image_file() opens both plain
image files and those zips, so the evaluation harness can point a ground-truth CSV at either.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageOps

from id_classifier.types import ImageRecord

log = logging.getLogger(__name__)

ImageKey = tuple[int, str, str]  # (transaction_id, image_id, reference)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FILENAME_PATTERN = re.compile(r"^(?P<transaction_id>\d+)_(?P<image_id>[A-Za-z0-9-]+)$")  # file stem like "123456_front"

# NFS zip names: <transaction_id>_<date>_<time>_<side>_<img|url>.zip, e.g. 1575232_2026-10-01_15-29-25-873057_front_img.zip
NFS_ZIP_PATTERN = re.compile(
    r"^(?P<transaction_id>\d+)_\d{4}-\d{2}-\d{2}_[\d.-]+_(?P<image_id>front|back)_(?:img|url)\.zip$"
)
ZIP_IMAGE_MEMBER = "original.jpeg"  # the uploaded image inside every NFS zip


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


def read_zip_member(path: str | Path, member: str = ZIP_IMAGE_MEMBER) -> bytes:
    """Reads one file out of a zip in memory (the image is never written to disk)."""
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)


def read_image_file(transaction_id: int, image_id: str, path: str | Path) -> ImageRecord:
    """Reads one image from disk: a plain image file, or an NFS zip holding original.jpeg."""
    try:
        data = read_zip_member(path) if Path(path).suffix.lower() == ".zip" else Path(path).read_bytes()
    except OSError as exc:
        return failed_record(transaction_id, image_id, str(path), f"read_error: {type(exc).__name__}")
    except (zipfile.BadZipFile, KeyError) as exc:  # not a zip, or no original.jpeg inside
        return failed_record(transaction_id, image_id, str(path), f"zip_error: {type(exc).__name__}")
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


class NfsZipSource(ImageSource):
    """Images on the NFS share: one folder per Core service (= services_service.name, e.g. "tun_nid_ocr"),
    one zip per uploaded image, named <transaction_id>_<date>_<time>_<front|back>_<img|url>.zip.

    The image is read from the zip in memory. In production the keys will come from the database
    (4201 transaction id + its service name); list_keys() covers listing whole folders.
    """

    def __init__(self, root: str | Path, folders: list[str]):
        self.root = Path(root)
        self.folders = list(folders)

    def list_keys(self) -> list[ImageKey]:
        keys, skipped = [], 0
        for folder in self.folders:
            path = self.root / folder
            if not path.is_dir():
                raise FileNotFoundError(f"NFS folder not found: {path}")
            for file in sorted(path.glob("*.zip")):
                match = NFS_ZIP_PATTERN.match(file.name)
                if not match:
                    skipped += 1
                    continue
                keys.append((int(match["transaction_id"]), match["image_id"], str(file)))
        log.info("NfsZipSource: %d images found in %d folders, %d skipped (unexpected name)", len(keys), len(self.folders), skipped)
        return keys

    def fetch(self, transaction_id: int, image_id: str, reference: str) -> ImageRecord:
        return read_image_file(transaction_id, image_id, reference)
