import base64
import hashlib
import io
import zipfile

import pytest

from fakes import image_bytes
from id_classifier.sources import read_image_file, unpack_payload

JPEG = image_bytes((255, 0, 0), "JPEG")
PNG = image_bytes((0, 0, 255), "PNG")


def zipped(data: bytes, member: str = "original.jpeg") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, data)
    return buffer.getvalue()


def wrapped(text: bytes, width: int = 76) -> bytes:
    """Base64 split over several lines, as many tools write it."""
    return b"\n".join(text[i:i + width] for i in range(0, len(text), width))


# name -> (stored bytes, image bytes they hold)
PAYLOADS = {
    "plain_jpeg": (JPEG, JPEG),
    "plain_png": (PNG, PNG),
    "nfs_zip": (zipped(JPEG), JPEG),
    "base64": (base64.b64encode(JPEG), JPEG),
    "base64_data_uri": (b"data:image/jpeg;base64," + base64.b64encode(JPEG), JPEG),
    "base64_wrapped_lines": (wrapped(base64.b64encode(JPEG)), JPEG),
    "base64_urlsafe_no_padding": (base64.urlsafe_b64encode(JPEG).rstrip(b"="), JPEG),
    "zip_holding_base64": (zipped(base64.b64encode(JPEG)), JPEG),
    "base64_of_zip": (base64.b64encode(zipped(JPEG)), JPEG),
}


@pytest.mark.parametrize("name", PAYLOADS)
def test_every_payload_format_gives_the_image(tmp_path, name):
    stored, image = PAYLOADS[name]
    path = tmp_path / f"{name}.bin"                  # the extension is not used: the content decides
    path.write_bytes(stored)
    record = read_image_file(7, "front", path)
    assert record.fetch_status == "ok", record.failure_reason
    assert record.image.mode == "RGB" and record.image.size == (32, 24)
    assert record.image_hash == hashlib.sha256(image).hexdigest()   # hash of the image, not of its wrapping
    assert record.image_reference == str(path)


def test_same_picture_has_the_same_hash_in_every_wrapping(tmp_path):
    hashes = set()
    for name, (stored, image) in PAYLOADS.items():
        if image is JPEG:
            path = tmp_path / name
            path.write_bytes(stored)
            hashes.add(read_image_file(7, "front", path).image_hash)
    assert len(hashes) == 1                          # de-duplication works across storages


def test_unknown_bytes_are_returned_unchanged():
    assert unpack_payload(b"\x00\x01binary") == b"\x00\x01binary"


@pytest.mark.parametrize("stored, reason", [
    (b"PK\x03\x04 broken zip", "zip_error: BadZipFile"),
    (zipped(JPEG, member="other.jpeg"), "zip_error: KeyError"),     # no original.jpeg inside
    (b"QUJDR", "base64_error: Error"),                               # 5 base64 characters cannot decode
    (b"\x00\x01 not an image", "decode_error: UnidentifiedImageError"),
])
def test_bad_payloads_are_failed_records(tmp_path, stored, reason):
    path = tmp_path / "bad.zip"
    path.write_bytes(stored)
    record = read_image_file(7, "front", path)
    assert record.fetch_status == "failed" and record.image is None
    assert record.failure_reason == reason


def test_missing_file_is_a_read_error(tmp_path):
    record = read_image_file(7, "front", tmp_path / "missing.zip")
    assert record.fetch_status == "failed"
    assert record.failure_reason == "read_error: FileNotFoundError"
