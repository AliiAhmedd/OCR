"""Image sources: where images come from.

Every source answers two questions:
- list_keys(): which images exist? -> (transaction_id, image_id, reference) tuples
- fetch(...):  give me one image    -> ImageRecord (a failed fetch is returned as a record, never raised)

Reading one image is two steps, so a new storage (e.g. a cloud bucket) only has to provide the first:
1. get the bytes:     read_local_bytes() (a file on disk or the NFS share)
2. read the payload:  record_from_payload() works out what the bytes are (an NFS zip holding original.jpeg,
                      base64 text, or a plain image file) and decodes the image.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import random
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageOps

from id_classifier.types import ImageRecord

log = logging.getLogger(__name__)

ImageKey = tuple[int, str, str]  # (transaction_id, image_id, reference)


# NFS zip names: <transaction_id>_<date>_<time>_<side>_<img|url>.zip, e.g. 1575232_2026-10-01_15-29-25-873057_front_img.zip
NFS_ZIP_PATTERN = re.compile(
    r"^(?P<transaction_id>\d+)_\d{4}-\d{2}-\d{2}_[\d.-]+_(?P<image_id>front|back)_(?:img|url)\.zip$"
)
ZIP_IMAGE_MEMBER = "original.jpeg"  # the uploaded image inside every NFS zip
ZIP_MAGIC = b"PK\x03\x04"           # first bytes of every zip file
BASE64_TEXT = re.compile(rb"^[A-Za-z0-9+/_-]+={0,2}$")  # standard or URL-safe alphabet, whitespace already removed
MAX_PAYLOAD_LAYERS = 3              # e.g. zip -> base64 -> jpeg; stops a file that unpacks forever


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


def base64_text(data: bytes) -> bytes | None:
    """The base64 text inside `data` (data-URI prefix and whitespace removed, padding added), or None when
    `data` is not base64 text. A real image or zip always has non-ASCII bytes, so it never matches."""
    if not data.isascii():
        return None
    if data.startswith(b"data:"):                 # data:image/jpeg;base64,<text>
        data = data.partition(b",")[2]
    text = b"".join(data.split())                 # base64 is often wrapped over several lines
    if not text or not BASE64_TEXT.match(text):
        return None
    return text + b"=" * (-len(text) % 4)         # some writers drop the trailing '=' padding


def unpack_payload(data: bytes) -> bytes:
    """Unwraps stored bytes down to the image file bytes: an NFS zip -> its original.jpeg, base64 text ->
    the decoded bytes, anything else is returned as it is (Pillow decides whether it is an image).
    Layers can be nested (a zip holding base64 text). Raises zipfile.BadZipFile / KeyError for a broken zip
    or one without original.jpeg, binascii.Error for broken base64."""
    for _ in range(MAX_PAYLOAD_LAYERS):
        if data.startswith(ZIP_MAGIC):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:   # in memory, never written to disk
                data = archive.read(ZIP_IMAGE_MEMBER)
        elif (text := base64_text(data)) is not None:
            data = base64.b64decode(text.replace(b"-", b"+").replace(b"_", b"/"), validate=True)
        else:
            break
    return data


def record_from_payload(transaction_id: int, image_id: str, reference: str, data: bytes) -> ImageRecord:
    """Stored bytes (zip / base64 / plain image) -> ImageRecord. Every source calls this after getting the bytes."""
    try:
        image_data = unpack_payload(data)
    except (zipfile.BadZipFile, KeyError) as exc:   # broken zip, or no original.jpeg inside
        return failed_record(transaction_id, image_id, reference, f"zip_error: {type(exc).__name__}")
    except binascii.Error as exc:                   # looked like base64 but does not decode
        return failed_record(transaction_id, image_id, reference, f"base64_error: {type(exc).__name__}")
    return record_from_bytes(transaction_id, image_id, reference, image_data)


def read_local_bytes(path: str | Path) -> bytes:
    """The bytes of a file on disk or on the NFS share."""
    return Path(path).read_bytes()


def read_image_file(transaction_id: int, image_id: str, path: str | Path) -> ImageRecord:
    """Reads one image from disk: a plain image file, an NFS zip holding original.jpeg, or base64 text."""
    try:
        data = read_local_bytes(path)
    except OSError as exc:
        return failed_record(transaction_id, image_id, str(path), f"read_error: {type(exc).__name__}")
    return record_from_payload(transaction_id, image_id, str(path), data)


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


PERCEPTUAL_HASH_SIZE = 16          # 16 x 16 = 256 bits per image
NEAR_DUPLICATE_DISTANCE = 40       # at most this many of the 256 bits differ = the same card (re-saved or re-shot); user-checked


def perceptual_hash(image: Image.Image, size: int = PERCEPTUAL_HASH_SIZE) -> int:
    """Difference hash (dHash): a fingerprint of how the image LOOKS, not of its bytes. The image is shrunk
    to (size+1) x size grey pixels; each bit says "is this pixel brighter than its right neighbour?".
    A re-saved or re-compressed copy gives (almost) the same bits; a different picture differs in about
    half of them. Returned as one int of size*size bits."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = small.tobytes()                          # one byte per grey pixel, row by row
    bits = 0
    for y in range(size):
        row = pixels[y * (size + 1):(y + 1) * (size + 1)]
        for x in range(size):
            bits = (bits << 1) | (row[x] > row[x + 1])
    return bits


def hash_distance(a: int, b: int) -> int:
    """How many bits differ between two perceptual hashes (0 = look identical, ~128 = unrelated)."""
    return (a ^ b).bit_count()


def sample_nfs_records(root: str | Path, folder: str, per_folder: int, seed: int = 42, unique_only: bool = True,
                       max_distance: int = NEAR_DUPLICATE_DISTANCE):
    """Yields up to `per_folder` readable images of one NFS folder in a random order (same seed = same images).
    With unique_only, an image is skipped when it is a copy of one already yielded: the same bytes (sha256),
    or the same picture re-saved (perceptual distance <= max_distance). Many NFS folders hold the same
    upload dozens of times. Used by build-ground-truth, so every card is counted once."""
    source = NfsZipSource(root, [folder])
    keys = source.list_keys()
    random.Random(seed).shuffle(keys)
    seen_bytes, seen_looks, taken = set(), [], 0
    for transaction_id, image_id, reference in keys:
        if taken == per_folder:
            return
        record = source.fetch(transaction_id, image_id, reference)
        if record.image is None:
            continue
        if unique_only:
            if record.image_hash in seen_bytes:
                continue
            looks = perceptual_hash(record.image)
            if any(hash_distance(looks, seen) <= max_distance for seen in seen_looks):
                continue
            seen_bytes.add(record.image_hash)
            seen_looks.append(looks)
        taken += 1
        yield record
