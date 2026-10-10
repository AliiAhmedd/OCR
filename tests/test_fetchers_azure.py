"""AzureBlobFetcher against a fake container client (no Azure account, no network, no real images)."""

import base64
import hashlib

import pytest

pytest.importorskip("azure.storage.blob")
from azure.core.exceptions import (  # noqa: E402
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from fakes import image_bytes, zip_bytes  # noqa: E402
from id_classifier.fetchers import (  # noqa: E402
    AZURE_CONNECTION_STRING_ENV,
    AZURE_IDENTITY_ENVS,
    AZURE_SAS_TOKEN_ENV,
    AzureBlobFetcher,
    FetchError,
    fetch_record,
)
from id_classifier.references import storage_key  # noqa: E402

CONTAINER = "core-media"
JPEG = image_bytes((255, 0, 0), "JPEG")


class FakeDownloader:
    def __init__(self, data: bytes):
        self.size = len(data)       # Azure's StorageStreamDownloader exposes the blob size before readall()
        self._data = data

    def readall(self) -> bytes:
        return self._data


class FakeContainerClient:
    """Stands in for azure's ContainerClient: serves blobs from a dict, and records every call made."""

    def __init__(self, blobs: dict[str, bytes] | None = None, error: Exception | None = None):
        self.blobs = blobs or {}
        self.error = error
        self.calls: list[tuple[str, tuple, dict]] = []

    def download_blob(self, blob, **kwargs):
        self.calls.append(("download_blob", (blob,), kwargs))
        if self.error is not None:
            raise self.error
        if blob not in self.blobs:
            raise storage_error("BlobNotFound", 404)
        return FakeDownloader(self.blobs[blob])


class FakeResponse:
    """The minimum azure-core's HttpResponseError reads off a response."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.reason = "error"
        self.headers: dict[str, str] = {}
        self.content_type = "application/xml"

    def text(self, encoding=None):
        return ""

    def json(self):
        raise ValueError("no json body")


def storage_error(code: str | None, status_code: int, kind=None) -> HttpResponseError:
    """An error shaped like the ones azure-storage raises: the x-ms-error-code lands on `.error_code`."""
    kind = kind or (ResourceNotFoundError if status_code == 404 else HttpResponseError)
    error = kind(message=code or "error", response=FakeResponse(status_code))
    if code is not None:
        error.error_code = code     # exactly what the SDK's process_storage_error() does
    return error


def fetcher(container=CONTAINER, blobs=None, error=None, **kwargs) -> AzureBlobFetcher:
    return AzureBlobFetcher(container, client=FakeContainerClient(blobs, error), **kwargs)


def test_reads_the_blob_under_the_prefix():
    client = FakeContainerClient({"nfs/tun_nid_ocr/x.zip": b"stored bytes"})
    blob = AzureBlobFetcher(CONTAINER, prefix="/nfs/", client=client)   # slashes around the prefix are tidied
    assert blob.blob_name("tun_nid_ocr/x.zip") == "nfs/tun_nid_ocr/x.zip"
    assert blob.read_bytes("tun_nid_ocr/x.zip") == b"stored bytes"


@pytest.mark.parametrize("stored", [zip_bytes(JPEG), base64.b64encode(JPEG), JPEG], ids=["zip", "base64", "jpeg"])
def test_reference_to_image_end_to_end(stored):
    """Core's reference -> key -> Azure blob -> payload -> image, as the pipeline will do it."""
    blobs = {"tun_nid_ocr/123_front_img.zip": stored}
    key = storage_key("/mnt/nfs/tun_nid_ocr/123_front_img.zip", strip_prefixes=["/mnt/nfs"])
    record = fetch_record(fetcher(blobs=blobs), 123, "front", key)
    assert record.fetch_status == "ok", record.failure_reason
    assert record.image.size == (32, 24)
    assert record.image_hash == hashlib.sha256(JPEG).hexdigest()   # same hash as the NFS and S3 copies
    assert record.image_reference == "tun_nid_ocr/123_front_img.zip"


def test_missing_blob_is_not_found():
    with pytest.raises(FetchError, match="not_found"):
        fetcher().read_bytes("nope.zip")


def test_missing_container():
    with pytest.raises(FetchError, match="bucket_not_found"):
        fetcher(error=storage_error("ContainerNotFound", 404)).read_bytes("x.zip")


def test_too_large_blob_is_refused_before_reading():
    client = FakeContainerClient({"big.zip": b"x" * 100})
    with pytest.raises(FetchError, match="too_large"):
        AzureBlobFetcher(CONTAINER, max_bytes=10, client=client).read_bytes("big.zip")


def test_failed_fetch_becomes_a_failed_record():
    record = fetch_record(fetcher(), 9, "back", "nope.zip")
    assert record.fetch_status == "failed" and record.image is None
    assert record.failure_reason == "not_found"
    assert record.image_reference == "nope.zip"


@pytest.mark.parametrize("error, reason", [
    (storage_error("BlobNotFound", 404), "not_found"),
    (storage_error("ContainerNotFound", 404), "bucket_not_found"),
    (storage_error("AuthorizationFailure", 403), "access_denied"),
    (storage_error("AuthorizationPermissionMismatch", 403), "access_denied"),
    (storage_error("InsufficientAccountPermissions", 403), "access_denied"),
    (storage_error("AccountIsDisabled", 403), "access_denied"),
    (storage_error("AuthenticationFailed", 403, ClientAuthenticationError), "bad_credentials"),
    (storage_error("InvalidAuthenticationInfo", 401, ClientAuthenticationError), "bad_credentials"),
    (storage_error("ServerBusy", 503), "azure_error: ServerBusy"),
    (storage_error(None, 403), "access_denied"),          # no x-ms-error-code header: fall back to the status
    (storage_error(None, 404), "not_found"),
    (storage_error(None, 500), "azure_error: 500"),
    (ServiceRequestError(message="dns"), "connection_error: ServiceRequestError"),
])
def test_errors_become_short_reasons(error, reason):
    client = FakeContainerClient(error=error)
    with pytest.raises(FetchError) as caught:
        AzureBlobFetcher(CONTAINER, client=client).read_bytes("x.zip")
    assert caught.value.reason == reason
    # read-only: one download_blob, nothing else (no list, no properties, no write)
    assert [name for name, _, _ in client.calls] == ["download_blob"]


def test_the_storage_error_code_enum_is_reported_by_its_value():
    """The SDK attaches a StorageErrorCode enum, whose str() is "StorageErrorCode.SERVER_BUSY" - not loggable."""
    from azure.storage.blob import StorageErrorCode

    error = storage_error(None, 503)
    error.error_code = StorageErrorCode.SERVER_BUSY
    with pytest.raises(FetchError) as caught:
        AzureBlobFetcher(CONTAINER, client=FakeContainerClient(error=error)).read_bytes("x.zip")
    assert caught.value.reason == "azure_error: ServerBusy"


def test_container_is_required():
    with pytest.raises(ValueError):
        AzureBlobFetcher("", client=object())


def test_unknown_credential_source_is_refused():
    with pytest.raises(ValueError, match="credential_source"):
        AzureBlobFetcher(CONTAINER, credential_source="password", client=object())


class TestCredentials:
    """Which credential "auto" picks, and that none of them is ever an argument or logged."""

    @pytest.fixture(autouse=True)
    def no_azure_env(self, monkeypatch):
        for name in (AZURE_SAS_TOKEN_ENV, AZURE_CONNECTION_STRING_ENV, *AZURE_IDENTITY_ENVS):
            monkeypatch.delenv(name, raising=False)

    def test_no_credentials_at_all_is_a_failed_record_not_a_crash(self):
        """Nothing configured: reported at once, rather than ~30 s of DefaultAzureCredential probing."""
        blob = AzureBlobFetcher(CONTAINER, account="acct")       # construction must not raise
        record = fetch_record(blob, 1, "front", "x.zip")
        assert record.fetch_status == "failed" and record.failure_reason == "no_credentials"

    def test_sas_wins_over_connection_string(self, monkeypatch):
        monkeypatch.setenv(AZURE_SAS_TOKEN_ENV, "?sv=2024-01-01&sig=fake")
        monkeypatch.setenv(AZURE_CONNECTION_STRING_ENV, connection_string())
        client = AzureBlobFetcher(CONTAINER, account="acct").container_client()
        assert "sig=fake" in client.url and client.credential is None     # a SAS rides in the URL
        assert client.account_name == "acct"

    def test_auto_uses_an_identity_when_a_service_principal_is_configured(self, monkeypatch):
        for name in AZURE_IDENTITY_ENVS:
            monkeypatch.setenv(name, "fake")
        client = AzureBlobFetcher(CONTAINER, account="acct").container_client()
        assert type(client.credential).__name__ == "DefaultAzureCredential"

    def test_a_managed_identity_has_to_be_asked_for(self, monkeypatch):
        """No variables to detect, so "auto" skips it; credential_source="identity" uses it anyway."""
        assert AzureBlobFetcher(CONTAINER, account="acct", credential_source="identity").container_client()

    def test_connection_string_used_when_there_is_no_sas(self, monkeypatch):
        monkeypatch.setenv(AZURE_CONNECTION_STRING_ENV, connection_string())
        client = AzureBlobFetcher(CONTAINER, account="acct").container_client()
        assert client.container_name == CONTAINER
        assert client.credential.account_name == "acct"

    def test_sas_needs_an_account_url_so_it_falls_through(self, monkeypatch):
        """A SAS alone cannot address a container: without `account` the next source is used."""
        monkeypatch.setenv(AZURE_SAS_TOKEN_ENV, "?sv=2024-01-01&sig=fake")
        monkeypatch.setenv(AZURE_CONNECTION_STRING_ENV, connection_string())
        client = AzureBlobFetcher(CONTAINER).container_client()
        assert client.credential.account_name == "acct"                  # the connection string

    def test_an_explicit_source_does_not_fall_back(self, monkeypatch):
        monkeypatch.setenv(AZURE_CONNECTION_STRING_ENV, connection_string())
        blob = AzureBlobFetcher(CONTAINER, account="acct", credential_source="sas")
        with pytest.raises(FetchError, match="no_credentials"):
            blob.container_client()

    def test_the_secret_is_never_logged(self, monkeypatch, caplog):
        secret = "?sv=2024-01-01&sig=SUPERSECRETSIGNATURE"
        monkeypatch.setenv(AZURE_SAS_TOKEN_ENV, secret)
        with caplog.at_level("DEBUG"):
            AzureBlobFetcher(CONTAINER, account="acct").container_client()
        assert "SUPERSECRETSIGNATURE" not in caplog.text
        assert "sas" in caplog.text                                      # the source name is logged

    def test_the_client_is_built_once(self, monkeypatch):
        monkeypatch.setenv(AZURE_CONNECTION_STRING_ENV, connection_string())
        blob = AzureBlobFetcher(CONTAINER, account="acct")
        assert blob.container_client() is blob.container_client()


def connection_string(account: str = "acct") -> str:
    key = base64.b64encode(b"not-a-real-account-key").decode()
    return f"DefaultEndpointsProtocol=https;AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"
