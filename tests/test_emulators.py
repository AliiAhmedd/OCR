"""The real fetchers against the local emulators (docker-compose.yml), seeded by scripts/seed_emulators.py.

Unlike the other cloud tests (fakes), these talk HTTP to real storage servers: Azurite for AzureBlobFetcher
and MinIO for S3Fetcher. The seeded objects are the ground-truth NFS zips (real ID images, which never leave
this machine), so the checks are on hashes only and nothing about an image is printed.

Skipped unless both emulators answer on 127.0.0.1 and data/emulator_seed.csv exists:
    docker compose up -d && python scripts/seed_emulators.py && python -m pytest -v tests/test_emulators.py
"""

import csv
import hashlib
import socket
from pathlib import Path

import pytest

from id_classifier.fetchers import AzureBlobFetcher, FetchError, S3Fetcher, fetch_record
from id_classifier.sources import read_image_file

MANIFEST = Path("data/emulator_seed.csv")
AZURITE_PORT, MINIO_PORT = 10000, 9222
# Public dev credentials of the emulators (the same as in scripts/seed_emulators.py), not secrets.
AZURITE_ACCOUNT_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
MINIO_ENDPOINT, MINIO_USER, MINIO_PASSWORD = f"http://127.0.0.1:{MINIO_PORT}", "rootuser", "rootpass123"
CONTAINER = BUCKET = "ids"


def azurite_connection_string(account_key: str = AZURITE_ACCOUNT_KEY) -> str:
    return (f"DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey={account_key};"
            f"BlobEndpoint=http://127.0.0.1:{AZURITE_PORT}/devstoreaccount1;")


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


if not (_listening(AZURITE_PORT) and _listening(MINIO_PORT) and MANIFEST.is_file()):
    pytest.skip("emulators not running or not seeded (see CLIhelp.md: Local cloud emulators)",
                allow_module_level=True)

pytest.importorskip("azure.storage.blob")
pytest.importorskip("boto3")

with MANIFEST.open(newline="", encoding="utf-8") as f:
    SEEDED = list(csv.DictReader(f))      # key, nfs_path, bytes, sha256 (of the zip file)


@pytest.fixture
def azure(monkeypatch):
    """AzureBlobFetcher -> Azurite, credentials from the environment as in production."""
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azurite_connection_string())
    return AzureBlobFetcher(CONTAINER, credential_source="connection_string")


@pytest.fixture
def s3(monkeypatch):
    """S3Fetcher -> MinIO: access key + secret in the environment, endpoint URL as a setting."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", MINIO_USER)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", MINIO_PASSWORD)
    return S3Fetcher(BUCKET, region="us-east-1", endpoint_url=MINIO_ENDPOINT)


def _transaction_id(key: str) -> int:
    return int(Path(key).name.split("_")[0])        # NFS names start with the transaction id


def test_every_seeded_object_arrives_byte_for_byte(azure, s3):
    """The bytes each fetcher returns are exactly the NFS zip (same size, same sha256)."""
    wrong = []
    for row in SEEDED:
        for fetcher in (azure, s3):
            data = fetcher.read_bytes(row["key"])
            if len(data) != int(row["bytes"]) or hashlib.sha256(data).hexdigest() != row["sha256"]:
                wrong.append((fetcher.name, row["key"]))
    assert wrong == [], f"{len(wrong)} objects differ from their NFS copy, e.g. {wrong[:3]}"
    assert len(SEEDED) > 0


def test_every_seeded_object_decodes_to_the_same_image_as_nfs(azure, s3):
    """fetch_record: same unwrapped-image sha256 from Azure, S3 and (when mounted) the NFS share."""
    nfs_mounted = Path(SEEDED[0]["nfs_path"]).is_file()
    wrong = []
    for row in SEEDED:
        txn = _transaction_id(row["key"])
        from_azure = fetch_record(azure, txn, "img", row["key"])
        from_s3 = fetch_record(s3, txn, "img", row["key"])
        hashes = {from_azure.image_hash, from_s3.image_hash}
        if nfs_mounted:
            hashes.add(read_image_file(txn, "img", row["nfs_path"]).image_hash)
        ok = from_azure.fetch_status == from_s3.fetch_status == "ok"
        if not ok or len(hashes) != 1 or None in hashes:
            wrong.append(row["key"])
    assert wrong == [], f"{len(wrong)} images differ between storages, e.g. {wrong[:3]}"


def test_missing_object_is_not_found(azure, s3):
    for fetcher in (azure, s3):
        with pytest.raises(FetchError) as exc:
            fetcher.read_bytes("tun_nid_ocr/does_not_exist.zip")
        assert exc.value.reason == "not_found", fetcher.name


def test_missing_container_or_bucket(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azurite_connection_string())
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", MINIO_USER)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", MINIO_PASSWORD)
    fetchers = [AzureBlobFetcher("no-such-container", credential_source="connection_string"),
                S3Fetcher("no-such-bucket", region="us-east-1", endpoint_url=MINIO_ENDPOINT)]
    for fetcher in fetchers:
        with pytest.raises(FetchError) as exc:
            fetcher.read_bytes(SEEDED[0]["key"])
        assert exc.value.reason == "bucket_not_found", fetcher.name


def test_wrong_credentials(monkeypatch, azure, s3):
    """A wrong key is refused, also after fetchers with the right key were used in this process
    (boto3's default session would have kept the first credentials: S3Fetcher has its own session)."""
    for fetcher in (azure, s3):                                    # the right keys work first
        fetcher.read_bytes(SEEDED[0]["key"])
    wrong_key = "A" * 86 + "=="                                    # well-formed base64, wrong account key
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azurite_connection_string(wrong_key))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-the-password")
    # MinIO answers SignatureDoesNotMatch, like S3. Azurite answers 403 AuthorizationFailure where real
    # Azure answers AuthenticationFailed (-> bad_credentials): an emulator difference, pinned as it is.
    expected = [(AzureBlobFetcher(CONTAINER, credential_source="connection_string", max_attempts=0), "access_denied"),
                (S3Fetcher(BUCKET, region="us-east-1", endpoint_url=MINIO_ENDPOINT, max_attempts=1), "bad_credentials")]
    for fetcher, reason in expected:
        with pytest.raises(FetchError) as exc:
            fetcher.read_bytes(SEEDED[0]["key"])
        assert exc.value.reason == reason, fetcher.name
