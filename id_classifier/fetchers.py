"""Fetchers: get the stored bytes of one image from the storage Core keeps it in.

A fetcher only answers "give me the bytes of this key" (the key comes from references.storage_key()).
What the bytes are (zip / base64 / jpeg) is decided afterwards by sources.record_from_payload(), the same
for every storage. Fetchers are READ-ONLY: they never write, copy or delete anything.

Credentials never live in code or YAML. Each cloud SDK finds them itself (environment variables, a
profile file, or the identity of the machine it runs on); see the S3Fetcher docstring.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from id_classifier.sources import failed_record, record_from_payload
from id_classifier.types import ImageRecord

log = logging.getLogger(__name__)

MAX_OBJECT_BYTES = 50 * 1024 * 1024   # one ID image is well under 15 MB; bigger is refused before downloading


class FetchError(Exception):
    """A fetch that failed. `reason` is short and safe to log and store (no credentials, no content)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Fetcher(ABC):
    """Interface every storage implements (S3 now; Azure Blob and GCS next)."""

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
        return f"{self.prefix}/{key}" if self.prefix else key

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
