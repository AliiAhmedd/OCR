"""Seed the local cloud emulators (docker-compose.yml): create the bucket and upload every ground-truth image.

Every file listed in the ground truth (NFS zips and the hand-labeled DocumentData images) is uploaded, byte for
byte, to
    Azurite (Azure Blob emulator)  container `ids`
    MinIO   (S3-compatible)        bucket    `ids`
under its storage key, e.g. /mnt/nfs/tun_nid_ocr/x.zip -> tun_nid_ocr/x.zip, the same key
references.storage_key() gives the pipeline. So AzureBlobFetcher / S3Fetcher pointed at the emulators
read exactly the files the NFS share holds.

It talks to 127.0.0.1 only, with the emulators' well-known dev credentials (not secrets): the images are real
IDs and never leave this machine. Safe to re-run: an object that is already there is skipped. Writes
data/emulator_seed.csv (key, size, sha256 of every uploaded file).

    python scripts/seed_emulators.py --ground-truth data/ground_truth_plus.csv \\
        --strip-prefix /mnt/nfs /home/<you>/Downloads          # NFS root + the folder holding DocumentData/
    python scripts/seed_emulators.py ... --only azure          # or: --only s3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from id_classifier.references import storage_key

# Azurite's fixed development account (published in Microsoft's docs, the same on every machine).
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)
# MinIO's root user from docker-compose.yml.
MINIO_ENDPOINT = "http://127.0.0.1:9222"
MINIO_ACCESS_KEY = "rootuser"
MINIO_SECRET_KEY = "rootpass123"
MINIO_REGION = "us-east-1"


def read_files(ground_truth: Path, strip_prefixes: list[str]) -> list[tuple[str, Path]]:
    """(storage key, local file path) for every ground-truth row, without duplicates. The key is the path minus
    the longest matching root: /mnt/nfs/tun_nid_ocr/x.zip -> tun_nid_ocr/x.zip, <Downloads>/DocumentData/y.jpg ->
    DocumentData/y.jpg (the same key the experiment configs' strip_prefixes give)."""
    with ground_truth.open(newline="", encoding="utf-8") as f:
        paths = [row["image_path"] for row in csv.DictReader(f)]
    files = {storage_key(p, strip_prefixes): Path(p) for p in paths}
    return sorted(files.items())


def seed_azure(files, container: str) -> tuple[int, int]:
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(AZURITE_CONNECTION_STRING)
    client = service.get_container_client(container)
    try:
        client.create_container()
        print(f"azure: created container '{container}'")
    except ResourceExistsError:
        pass
    uploaded = skipped = 0
    for key, path in files:
        blob = client.get_blob_client(key)
        if blob.exists():
            skipped += 1
            continue
        blob.upload_blob(path.read_bytes())
        uploaded += 1
    return uploaded, skipped


def seed_s3(files, bucket: str) -> tuple[int, int]:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
    )
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"s3: created bucket '{bucket}'")
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    uploaded = skipped = 0
    for key, path in files:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            skipped += 1
            continue
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
        uploaded += 1
    return uploaded, skipped


def write_manifest(files, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "nfs_path", "bytes", "sha256"])
        for key, path in files:
            data = path.read_bytes()
            writer.writerow([key, path.as_posix(), len(data), hashlib.sha256(data).hexdigest()])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.csv"))
    parser.add_argument("--strip-prefix", nargs="+", default=["/mnt/nfs"],
                        help="root(s) removed to make the key, e.g. the NFS root and the folder holding DocumentData")
    parser.add_argument("--container", default="ids", help="Azurite container")
    parser.add_argument("--bucket", default="ids", help="MinIO bucket")
    parser.add_argument("--only", choices=["azure", "s3"], help="seed one emulator only")
    parser.add_argument("--manifest", type=Path, default=Path("data/emulator_seed.csv"))
    args = parser.parse_args(argv)

    files = read_files(args.ground_truth, args.strip_prefix)
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        print(f"{len(missing)} ground-truth zips not found, e.g. {missing[0]} (is the NFS share mounted?)")
        return 1
    folders = Counter(key.split("/", 1)[0] for key, _ in files)
    print(f"{len(files)} zips from {args.ground_truth}: " + ", ".join(f"{k} {v}" for k, v in sorted(folders.items())))

    if args.only in (None, "azure"):
        up, skip = seed_azure(files, args.container)
        print(f"azure (Azurite) container '{args.container}': {up} uploaded, {skip} already there")
    if args.only in (None, "s3"):
        up, skip = seed_s3(files, args.bucket)
        print(f"s3 (MinIO) bucket '{args.bucket}': {up} uploaded, {skip} already there")

    write_manifest(files, args.manifest)
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
