"""GcsFetcher against a fake bucket client (no Google account, no network, no real images)."""

import base64
import hashlib

import pytest

pytest.importorskip("google.cloud.storage")
from google.api_core.exceptions import (  # noqa: E402
    Forbidden,
    NotFound,
    RetryError,
    ServiceUnavailable,
    TooManyRequests,
    Unauthorized,
)
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError  # noqa: E402
from requests.exceptions import ConnectionError as RequestsConnectionError  # noqa: E402

from fakes import image_bytes, zip_bytes  # noqa: E402
from id_classifier.fetchers import FetchError, GcsFetcher, S3Fetcher, fetch_record  # noqa: E402
from id_classifier.references import storage_key  # noqa: E402

BUCKET = "core-media"
JPEG = image_bytes((255, 0, 0), "JPEG")


class FakeBlob:
    def __init__(self, name: str, data: bytes | None, error: Exception | None, calls: list):
        self.name = name
        self.size = None if data is None else len(data)
        self._data = data
        self._error = error
        self._calls = calls

    def reload(self, **kwargs):
        self._calls.append(("reload", self.name, kwargs))
        if self._error is not None:
            raise self._error
        if self._data is None:
            raise NotFound(f"No such object: {BUCKET}/{self.name}")

    def download_as_bytes(self, **kwargs):
        self._calls.append(("download_as_bytes", self.name, kwargs))
        if self._error is not None:
            raise self._error
        return self._data


class FakeBucketClient:
    """Stands in for google.cloud.storage's Bucket: serves objects from a dict, records every call."""

    def __init__(self, objects: dict[str, bytes] | None = None, error: Exception | None = None):
        self.objects = objects or {}
        self.error = error
        self.calls: list[tuple] = []

    def blob(self, name):
        self.calls.append(("blob", name, {}))
        return FakeBlob(name, self.objects.get(name), self.error, self.calls)


def fetcher(bucket=BUCKET, objects=None, error=None, **kwargs) -> GcsFetcher:
    return GcsFetcher(bucket, client=FakeBucketClient(objects, error), **kwargs)


def test_reads_the_object_under_the_prefix():
    client = FakeBucketClient({"nfs/tun_nid_ocr/x.zip": b"stored bytes"})
    gcs = GcsFetcher(BUCKET, prefix="/nfs/", client=client)    # slashes around the prefix are tidied
    assert gcs.object_name("tun_nid_ocr/x.zip") == "nfs/tun_nid_ocr/x.zip"
    assert gcs.read_bytes("tun_nid_ocr/x.zip") == b"stored bytes"


@pytest.mark.parametrize("stored", [zip_bytes(JPEG), base64.b64encode(JPEG), JPEG], ids=["zip", "base64", "jpeg"])
def test_reference_to_image_end_to_end(stored):
    """Core's reference -> key -> GCS object -> payload -> image, as the pipeline will do it."""
    objects = {"tun_nid_ocr/123_front_img.zip": stored}
    key = storage_key("/mnt/nfs/tun_nid_ocr/123_front_img.zip", strip_prefixes=["/mnt/nfs"])
    record = fetch_record(fetcher(objects=objects), 123, "front", key)
    assert record.fetch_status == "ok", record.failure_reason
    assert record.image.size == (32, 24)
    assert record.image_hash == hashlib.sha256(JPEG).hexdigest()   # same hash as the NFS, S3 and Azure copies
    assert record.image_reference == "tun_nid_ocr/123_front_img.zip"


def test_missing_object_is_not_found():
    with pytest.raises(FetchError, match="not_found"):
        fetcher().read_bytes("nope.zip")


def test_missing_bucket_is_told_apart_from_a_missing_object():
    """GCS answers 404 for both; only the message differs, so this is the one thing holding them apart."""
    error = NotFound("The specified bucket does not exist.")
    with pytest.raises(FetchError, match="bucket_not_found"):
        fetcher(error=error).read_bytes("x.zip")


def test_an_unrecognised_404_message_degrades_to_not_found():
    with pytest.raises(FetchError) as caught:
        fetcher(error=NotFound("something new google changed")).read_bytes("x.zip")
    assert caught.value.reason == "not_found"


def test_too_large_object_is_refused_before_it_is_downloaded():
    client = FakeBucketClient({"big.zip": b"x" * 100})
    with pytest.raises(FetchError, match="too_large"):
        GcsFetcher(BUCKET, max_bytes=10, client=client).read_bytes("big.zip")
    assert [name for name, _, _ in client.calls] == ["blob", "reload"]   # no download happened


def test_failed_fetch_becomes_a_failed_record():
    record = fetch_record(fetcher(), 9, "back", "nope.zip")
    assert record.fetch_status == "failed" and record.image is None
    assert record.failure_reason == "not_found"
    assert record.image_reference == "nope.zip"


@pytest.mark.parametrize("error, reason", [
    (Forbidden("no read permission"), "access_denied"),
    (Unauthorized("bad token"), "bad_credentials"),
    (RefreshError("key revoked"), "bad_credentials"),
    (DefaultCredentialsError("nothing configured"), "no_credentials"),
    (TransportError("dns"), "connection_error: TransportError"),
    (RequestsConnectionError("refused"), "connection_error: ConnectionError"),
    (RetryError("deadline", cause=None), "connection_error: RetryError"),
    (TooManyRequests("slow down"), "gcs_error: 429"),
    (ServiceUnavailable("backend"), "gcs_error: 503"),
])
def test_errors_become_short_reasons(error, reason):
    client = FakeBucketClient(error=error)
    with pytest.raises(FetchError) as caught:
        GcsFetcher(BUCKET, client=client).read_bytes("x.zip")
    assert caught.value.reason == reason
    # read-only: the metadata read and nothing else once it has failed (no list, no write)
    assert [name for name, _, _ in client.calls] == ["blob", "reload"]


def test_timeouts_and_a_retry_deadline_are_passed_to_the_sdk():
    client = FakeBucketClient({"x.zip": b"bytes"})
    GcsFetcher(BUCKET, connect_timeout=2, read_timeout=9, retry_timeout=30, client=client).read_bytes("x.zip")
    reload_kwargs = next(kwargs for name, _, kwargs in client.calls if name == "reload")
    assert reload_kwargs["timeout"] == (2, 9)
    assert reload_kwargs["retry"].timeout == 30


def test_bucket_is_required():
    with pytest.raises(ValueError):
        GcsFetcher("", client=object())


def test_hmac_keys_would_be_read_by_the_s3_fetcher_instead():
    """Q6 may say Core uses HMAC keys. GCS speaks the S3 protocol, so S3Fetcher covers that case and no
    GCS code is involved. This only pins the wiring (naming + endpoint); the transfer itself is S3Fetcher's,
    already tested against moto in test_fetchers_s3.py."""
    s3 = S3Fetcher(BUCKET, prefix="nfs", endpoint_url="https://storage.googleapis.com", client=object())
    assert s3.object_name("tun_nid_ocr/x.zip") == "nfs/tun_nid_ocr/x.zip"
    assert s3.name == f"s3:{BUCKET}"
