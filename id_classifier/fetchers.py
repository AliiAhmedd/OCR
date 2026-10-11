"""Fetchers: get the stored bytes of one image from the storage Core keeps it in.

A fetcher only answers "give me the bytes of this key" (the key comes from references.storage_key()).
What the bytes are (zip / base64 / jpeg) is decided afterwards by sources.record_from_payload(), the same
for every storage. Fetchers are READ-ONLY: they never write, copy or delete anything.

Credentials never live in code or YAML. Each cloud SDK finds them itself (environment variables, a
profile file, or the identity of the machine it runs on); see each fetcher's docstring.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from id_classifier.sources import failed_record, record_from_payload
from id_classifier.types import ImageRecord

log = logging.getLogger(__name__)

MAX_OBJECT_BYTES = 50 * 1024 * 1024   # one ID image is well under 15 MB; bigger is refused before downloading


def prefixed(prefix: str, key: str) -> str:
    """The name of the stored object: the key under the folder where Core's files start (prefix may be "")."""
    return f"{prefix}/{key}" if prefix else key


class FetchError(Exception):
    """A fetch that failed. `reason` is short and safe to log and store (no credentials, no content)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Fetcher(ABC):
    """Interface every storage implements: Amazon S3, Azure Blob Storage, Google Cloud Storage."""

    name: str  # e.g. "s3:my-bucket", used in logs

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """The stored bytes of one object. Raises FetchError on any problem."""


def fetch_record(fetcher: Fetcher, transaction_id: int, image_id: str, key: str) -> ImageRecord:
    """Fetch one image and decode it. A failure becomes a failed record (never raised), like sources.py."""
    try:
        data = fetcher.read_bytes(key)
    except FetchError as exc:
        log.warning("Fetch failed (%s) for transaction %s %s: %s", fetcher.name, transaction_id, image_id, exc.reason)
        return failed_record(transaction_id, image_id, key, exc.reason)
    return record_from_payload(transaction_id, image_id, key, data)


# S3 error codes -> our reasons. Anything else becomes "s3_error: <code>".
S3_REASONS = {
    "NoSuchKey": "not_found",
    "404": "not_found",
    "NotFound": "not_found",
    "NoSuchBucket": "bucket_not_found",
    "AccessDenied": "access_denied",
    "403": "access_denied",
    "InvalidAccessKeyId": "bad_credentials",
    "SignatureDoesNotMatch": "bad_credentials",
    "ExpiredToken": "bad_credentials",
    "InvalidToken": "bad_credentials",
}


class S3Fetcher(Fetcher):
    """Reads objects from one Amazon S3 bucket (or any S3-compatible store via `endpoint_url`).

    Credentials: boto3's standard chain, in this order: the environment variables AWS_ACCESS_KEY_ID +
    AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN for temporary keys), the ~/.aws/credentials profile
    (AWS_PROFILE), then the IAM role of the machine. They are never passed in here.
    Permission needed: s3:GetObject on the bucket (read-only).

    The object name is `prefix/key` (prefix = the folder inside the bucket where Core's files start, may be "").
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        max_attempts: int = 3,
        connect_timeout: float = 5,
        read_timeout: float = 30,
        max_bytes: int = MAX_OBJECT_BYTES,
        client=None,
    ):
        if not bucket:
            raise ValueError("S3Fetcher needs a bucket name")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.max_bytes = max_bytes
        self.name = f"s3:{bucket}"
        if client is None:                      # imported here: boto3 is only needed when S3 is used
            import boto3
            from botocore.config import Config

            config = Config(
                retries={"max_attempts": max_attempts, "mode": "standard"},  # retries throttling / 5xx / network
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url, config=config)
        self.client = client

    def object_name(self, key: str) -> str:
        return prefixed(self.prefix, key)

    def read_bytes(self, key: str) -> bytes:
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.object_name(key))
            body = response["Body"]
            try:
                if response.get("ContentLength", 0) > self.max_bytes:
                    raise FetchError("too_large")       # refused before the content is downloaded
                return body.read()
            finally:
                body.close()
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", "unknown"))
            raise FetchError(S3_REASONS.get(code, f"s3_error: {code}")) from None
        except NoCredentialsError:
            raise FetchError("no_credentials") from None
        except BotoCoreError as exc:            # endpoint unreachable, timeout, ...
            raise FetchError(f"connection_error: {type(exc).__name__}") from None


# Azure storage error codes (the x-ms-error-code header) -> our reasons. Anything else: "azure_error: <code>".
AZURE_REASONS = {
    "BlobNotFound": "not_found",
    "ResourceNotFound": "not_found",
    "PathNotFound": "not_found",
    "ContainerNotFound": "bucket_not_found",
    "AuthenticationFailed": "bad_credentials",
    "InvalidAuthenticationInfo": "bad_credentials",
    "NoAuthenticationInformation": "bad_credentials",
    "AuthorizationFailure": "access_denied",
    "AuthorizationPermissionMismatch": "access_denied",   # RBAC role missing; not in the SDK's enum
    "InsufficientAccountPermissions": "access_denied",
    "AccountIsDisabled": "access_denied",
}

AZURE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
AZURE_SAS_TOKEN_ENV = "AZURE_STORAGE_SAS_TOKEN"
AZURE_IDENTITY_ENVS = ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET")   # a service principal
CREDENTIAL_SOURCES = ("sas", "connection_string", "identity")

AZURE_SINGLE_GET_BYTES = 16 * 1024 * 1024   # see AzureBlobFetcher.read_bytes: bounds a too-large download


class AzureBlobFetcher(Fetcher):
    """Reads blobs from one Azure Blob Storage container.

    Credentials come from the environment only, never from an argument or YAML. `credential_source`
    picks which one; "auto" (the default) takes the first that is usable, in this order:

        1. "sas"               a read-only, time-limited SAS token in AZURE_STORAGE_SAS_TOKEN (+ `account`).
                               Narrowest: scoped to what the token was signed for.
        2. "connection_string" AZURE_STORAGE_CONNECTION_STRING (account name + account key). This is what a
                               django-storages deployment usually sets, but the account key grants full
                               access to the whole account, so it is not preferred over a SAS.
        3. "identity"          DefaultAzureCredential (+ `account`): a service principal
                               (AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET). No secret of ours.

    An explicit secret wins over the ambient machine identity: if an operator exported a credential,
    honouring it is predictable, whereas DefaultAzureCredential has its own internal chain (it will pick up
    an Azure CLI login on a dev box) and fails slowly rather than fast. For the same reason "auto" only
    reaches step 3 when the three service-principal variables are set: otherwise DefaultAzureCredential
    would spend ~30 s probing IMDS, the Azure CLI and VS Code and then report `bad_credentials`, hiding a
    deployment with no credentials at all behind a hang. To use a machine's **managed identity** (no
    variables to detect), ask for it: `credential_source="identity"`.

    Which one the real deployment uses is Q6 in OPEN_QUESTIONS.md; set `credential_source` explicitly once
    that is answered instead of relying on "auto".

    Permission needed: *Storage Blob Data Reader* on the container, or a SAS with read permission only.
    The blob name is `prefix/key` (prefix = the folder inside the container where Core's files start).
    """

    def __init__(
        self,
        container: str,
        prefix: str = "",
        account: str | None = None,
        account_url: str | None = None,
        credential_source: str = "auto",
        max_attempts: int = 3,
        connect_timeout: float = 5,
        read_timeout: float = 30,
        max_bytes: int = MAX_OBJECT_BYTES,
        single_get_bytes: int = AZURE_SINGLE_GET_BYTES,
        client=None,
    ):
        if not container:
            raise ValueError("AzureBlobFetcher needs a container name")
        if credential_source != "auto" and credential_source not in CREDENTIAL_SOURCES:
            raise ValueError(f"credential_source must be 'auto' or one of {CREDENTIAL_SOURCES}")
        self.container = container
        self.prefix = prefix.strip("/")
        self.max_bytes = max_bytes
        self.name = f"azure:{container}"
        self.account_url = account_url or (f"https://{account}.blob.core.windows.net" if account else None)
        self.credential_source = credential_source
        self._settings = {
            "retry_total": max_attempts,
            "connection_timeout": connect_timeout,
            "read_timeout": read_timeout,
            "max_single_get_size": min(single_get_bytes, max_bytes),
            "max_chunk_get_size": min(single_get_bytes, max_bytes),
        }
        self._client = client   # built on first use, so a missing credential is a failed record, not a crash

    def blob_name(self, key: str) -> str:
        return prefixed(self.prefix, key)

    def container_client(self):
        """The container client, built on first use (like boto3, credentials are resolved when first needed)."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        # imported here: the azure packages are only needed when Azure is used
        from azure.storage.blob import BlobServiceClient

        wanted = CREDENTIAL_SOURCES if self.credential_source == "auto" else (self.credential_source,)
        sas, connection_string = os.environ.get(AZURE_SAS_TOKEN_ENV), os.environ.get(AZURE_CONNECTION_STRING_ENV)
        # "auto" only tries an identity when a service principal is configured; a managed identity has
        # nothing to detect, so it has to be asked for (see the class docstring).
        identity_usable = self.credential_source == "identity" or all(map(os.environ.get, AZURE_IDENTITY_ENVS))

        for source in wanted:
            if source == "sas" and sas and self.account_url:
                service = BlobServiceClient(self.account_url, credential=sas, **self._settings)
            elif source == "connection_string" and connection_string:
                service = BlobServiceClient.from_connection_string(connection_string, **self._settings)
            elif source == "identity" and self.account_url and identity_usable:
                from azure.identity import DefaultAzureCredential

                service = BlobServiceClient(self.account_url, credential=DefaultAzureCredential(), **self._settings)
            else:
                continue
            log.info("Azure credentials for %s from %s", self.name, source)   # the source only, never the secret
            return service.get_container_client(self.container)

        raise FetchError("no_credentials")

    def read_bytes(self, key: str) -> bytes:
        """Downloads one blob. Unlike S3's GetObject, Azure's download_blob() already fetches its first
        chunk, so `max_single_get_size` bounds that chunk: a too-large blob costs at most ~16 MB (well above
        a real ID image, so a normal image is still a single request) instead of the whole object."""
        from azure.core.exceptions import (
            AzureError,
            ClientAuthenticationError,
            HttpResponseError,
            ServiceRequestError,
        )

        try:
            downloader = self.container_client().download_blob(self.blob_name(key), max_concurrency=1)
            if downloader.size > self.max_bytes:
                raise FetchError("too_large")
            return downloader.readall()
        except ClientAuthenticationError:       # subclass of HttpResponseError, so caught before it
            raise FetchError("bad_credentials") from None
        except ServiceRequestError as exc:      # DNS, TLS, connect / read timeout, after the retries
            raise FetchError(f"connection_error: {type(exc).__name__}") from None
        except HttpResponseError as exc:        # covers ResourceNotFoundError (blob or container missing)
            raise FetchError(azure_reason(exc)) from None
        except AzureError as exc:               # anything else the SDK raises
            raise FetchError(f"connection_error: {type(exc).__name__}") from None


def azure_reason(exc) -> str:
    """Our short reason for an Azure HttpResponseError, from the x-ms-error-code the SDK attached to it."""
    code = getattr(exc, "error_code", None)
    code = getattr(code, "value", code)         # StorageErrorCode is an enum: take its string ("BlobNotFound")
    if code:
        return AZURE_REASONS.get(code, f"azure_error: {code}")
    status = getattr(exc, "status_code", None)  # no x-ms-error-code header: fall back to the status
    return {403: "access_denied", 404: "not_found"}.get(status, f"azure_error: {status or type(exc).__name__}")


GCS_RETRY_TIMEOUT = 60      # seconds the SDK may keep retrying throttling / 5xx / connection errors


class GcsFetcher(Fetcher):
    """Reads objects from one Google Cloud Storage bucket.

    Credentials: Application Default Credentials, which the SDK resolves itself, in this order: the
    service-account JSON file named by GOOGLE_APPLICATION_CREDENTIALS, then a gcloud user login, then the
    service account of the machine (GKE workload identity / Compute Engine). They are never passed in here.
    Unlike Azure there is no choice to make: this is a real standard chain, like boto3's.
    Permission needed: *Storage Object Viewer* (`roles/storage.objectViewer`) on the bucket.

    If Q6 says Core uses **HMAC keys** instead, no GCS code is needed at all: GCS also speaks the S3
    protocol, so `S3Fetcher(bucket, endpoint_url="https://storage.googleapis.com")` reads it.

    The object name is `prefix/key` (prefix = the folder inside the bucket where Core's files start).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        retry_timeout: float = GCS_RETRY_TIMEOUT,
        connect_timeout: float = 5,
        read_timeout: float = 30,
        max_bytes: int = MAX_OBJECT_BYTES,
        client=None,
    ):
        if not bucket:
            raise ValueError("GcsFetcher needs a bucket name")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.project = project
        self.max_bytes = max_bytes
        self.name = f"gcs:{bucket}"
        self.timeout = (connect_timeout, read_timeout)
        self.retry_timeout = retry_timeout
        self._client = client   # built on first use, so a missing credential is a failed record, not a crash

    def object_name(self, key: str) -> str:
        return prefixed(self.prefix, key)

    def bucket_client(self):
        """The bucket handle, built on first use (Client() is what raises when there are no credentials)."""
        if self._client is None:
            from google.cloud.storage import Client   # imported here: only needed when GCS is used

            self._client = (Client(project=self.project) if self.project else Client()).bucket(self.bucket)
        return self._client

    def read_bytes(self, key: str) -> bytes:
        """Reads the object's metadata first, so an oversized object is refused before it is downloaded.
        That is one extra round trip per image, which GCS gives no way around: download_as_bytes() returns
        the whole object and reports no size beforehand."""
        from google.api_core.exceptions import (
            Forbidden,
            GoogleAPICallError,
            NotFound,
            RetryError,
            Unauthorized,
        )
        from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, TransportError
        from google.cloud.storage.retry import DEFAULT_RETRY
        from requests.exceptions import RequestException

        retry = DEFAULT_RETRY.with_timeout(self.retry_timeout)
        try:
            blob = self.bucket_client().blob(self.object_name(key))
            blob.reload(timeout=self.timeout, retry=retry)      # metadata: gives .size, and 404s if missing
            if (blob.size or 0) > self.max_bytes:
                raise FetchError("too_large")
            return blob.download_as_bytes(timeout=self.timeout, retry=retry)
        except NotFound as exc:                 # both subclasses of GoogleAPICallError, so caught before it
            raise FetchError(gcs_not_found_reason(exc)) from None
        except Unauthorized:
            raise FetchError("bad_credentials") from None
        except Forbidden:
            raise FetchError("access_denied") from None
        except DefaultCredentialsError:         # the SDK found no credentials at all
            raise FetchError("no_credentials") from None
        except (TransportError, RetryError, RequestException) as exc:   # unreachable, or retried until the deadline
            raise FetchError(f"connection_error: {type(exc).__name__}") from None
        except GoogleAuthError:                 # RefreshError: a key that no longer works
            raise FetchError("bad_credentials") from None
        except GoogleAPICallError as exc:       # anything else the API returned, with its status
            raise FetchError(f"gcs_error: {exc.code or type(exc).__name__}") from None


def gcs_not_found_reason(exc) -> str:
    """A missing bucket and a missing object are both a plain 404 with reason "notFound" and an empty
    `errors` list, so only the message tells them apart. It degrades to the safer `not_found`."""
    message = str(getattr(exc, "message", "") or exc)
    return "bucket_not_found" if "bucket does not exist" in message else "not_found"
