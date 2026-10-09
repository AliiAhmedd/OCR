"""S3Fetcher against moto's in-memory S3 (no AWS account, no network, no real images)."""

import base64
import hashlib

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError  # noqa: E402

from fakes import image_bytes, zip_bytes  # noqa: E402
from id_classifier.fetchers import FetchError, S3Fetcher, fetch_record  # noqa: E402
from id_classifier.references import storage_key  # noqa: E402

BUCKET = "core-media"
JPEG = image_bytes((255, 0, 0), "JPEG")


@pytest.fixture
def s3(monkeypatch):
    """A fake S3 with one empty bucket. Fake credentials, so a real AWS account can never be reached."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def test_reads_the_object_under_the_prefix(s3):
    s3.put_object(Bucket=BUCKET, Key="nfs/tun_nid_ocr/x.zip", Body=b"stored bytes")
    fetcher = S3Fetcher(BUCKET, prefix="/nfs/", region="us-east-1")    # slashes around the prefix are tidied
    assert fetcher.object_name("tun_nid_ocr/x.zip") == "nfs/tun_nid_ocr/x.zip"
    assert fetcher.read_bytes("tun_nid_ocr/x.zip") == b"stored bytes"


@pytest.mark.parametrize("stored", [zip_bytes(JPEG), base64.b64encode(JPEG), JPEG], ids=["zip", "base64", "jpeg"])
def test_reference_to_image_end_to_end(s3, stored):
    """Core's reference -> key -> S3 object -> payload -> image, as the pipeline will do it."""
    s3.put_object(Bucket=BUCKET, Key="tun_nid_ocr/123_front_img.zip", Body=stored)
    key = storage_key("/mnt/nfs/tun_nid_ocr/123_front_img.zip", strip_prefixes=["/mnt/nfs"])
    record = fetch_record(S3Fetcher(BUCKET, region="us-east-1"), 123, "front", key)
    assert record.fetch_status == "ok", record.failure_reason
    assert record.image.size == (32, 24)
    assert record.image_hash == hashlib.sha256(JPEG).hexdigest()   # same hash as the NFS copy would give
    assert record.image_reference == "tun_nid_ocr/123_front_img.zip"


def test_missing_object_is_not_found(s3):
    with pytest.raises(FetchError, match="not_found"):
        S3Fetcher(BUCKET, region="us-east-1").read_bytes("nope.zip")


def test_missing_bucket(s3):
    with pytest.raises(FetchError, match="bucket_not_found"):
        S3Fetcher("no-such-bucket", region="us-east-1").read_bytes("x.zip")


def test_too_large_object_is_refused(s3):
    s3.put_object(Bucket=BUCKET, Key="big.zip", Body=b"x" * 100)
    with pytest.raises(FetchError, match="too_large"):
        S3Fetcher(BUCKET, region="us-east-1", max_bytes=10).read_bytes("big.zip")


def test_failed_fetch_becomes_a_failed_record(s3):
    record = fetch_record(S3Fetcher(BUCKET, region="us-east-1"), 9, "back", "nope.zip")
    assert record.fetch_status == "failed" and record.image is None
    assert record.failure_reason == "not_found"
    assert record.image_reference == "nope.zip"


class FailingClient:
    """Stands in for boto3's client: get_object raises the given error, and every call is recorded."""

    def __init__(self, error: Exception):
        self.error = error
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        raise self.error


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "..."}}, "GetObject")


@pytest.mark.parametrize("error, reason", [
    (client_error("AccessDenied"), "access_denied"),
    (client_error("InvalidAccessKeyId"), "bad_credentials"),
    (client_error("SignatureDoesNotMatch"), "bad_credentials"),
    (client_error("SlowDown"), "s3_error: SlowDown"),
    (NoCredentialsError(), "no_credentials"),
    (EndpointConnectionError(endpoint_url="https://s3.example"), "connection_error: EndpointConnectionError"),
])
def test_errors_become_short_reasons(error, reason):
    client = FailingClient(error)
    with pytest.raises(FetchError) as caught:
        S3Fetcher(BUCKET, client=client).read_bytes("x.zip")
    assert caught.value.reason == reason
    assert client.calls == [("get_object", {"Bucket": BUCKET, "Key": "x.zip"})]   # read-only: one GetObject, nothing else


def test_bucket_is_required():
    with pytest.raises(ValueError):
        S3Fetcher("", client=object())
