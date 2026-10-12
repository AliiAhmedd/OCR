"""The image source from YAML (`source:`): build, refuse secrets, Core reference -> key -> fetcher -> image,
and an experiment that gives the same results whether its images come from disk or from a bucket."""

import pytest
import yaml

from fakes import COLOR_OF, image_bytes, write_nfs_zip, zip_bytes
from id_classifier import config
from id_classifier.config import build_source
from id_classifier.evaluate import run_experiment
from id_classifier.fetchers import Fetcher, FetchError, FetcherSource
from id_classifier.sources import NfsZipSource, read_image_file


class DictFetcher(Fetcher):
    """A storage held in a dict: key -> bytes."""

    name = "dict:test"

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.asked = []

    def read_bytes(self, key: str) -> bytes:
        self.asked.append(key)
        if key not in self.objects:
            raise FetchError("not_found")
        return self.objects[key]


# ---------------------------------------------------------------- FetcherSource

def test_reference_is_turned_into_the_key_and_fetched(tmp_path):
    nfs_zip = write_nfs_zip(tmp_path / "tun_nid_ocr", 123, "front", COLOR_OF[("national_id", "TN", "front")])
    fetcher = DictFetcher({"tun_nid_ocr/" + nfs_zip.name: nfs_zip.read_bytes()})
    source = FetcherSource(fetcher, strip_prefixes=["/mnt/nfs"])

    record = source.fetch(123, "front", "/mnt/nfs/tun_nid_ocr/" + nfs_zip.name)

    assert fetcher.asked == ["tun_nid_ocr/" + nfs_zip.name]
    assert record.fetch_status == "ok"
    assert record.image_hash == read_image_file(123, "front", nfs_zip).image_hash   # same image as from disk


def test_bad_reference_never_reaches_the_storage():
    fetcher = DictFetcher({})
    record = FetcherSource(fetcher).fetch(1, "front", "/mnt/nfs/../etc/passwd")
    assert (record.fetch_status, record.failure_reason) == ("failed", "bad_reference")
    assert fetcher.asked == []


def test_missing_object_is_a_failed_record():
    record = FetcherSource(DictFetcher({})).fetch(1, "front", "tun_nid_ocr/x.zip")
    assert (record.fetch_status, record.failure_reason) == ("failed", "not_found")


def test_a_bucket_is_never_listed():
    with pytest.raises(NotImplementedError):
        FetcherSource(DictFetcher({})).list_keys()


# ---------------------------------------------------------------- building from YAML

def test_no_source_means_local_files():
    assert build_source(None) is None
    assert build_source({}) is None


def test_nfs_source():
    assert isinstance(build_source({"type": "nfs"}), NfsZipSource)


def test_s3_source(monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    source = build_source({"type": "s3", "bucket": "ids", "prefix": "core", "endpoint_url": "http://127.0.0.1:9222",
                           "strip_prefixes": ["/mnt/nfs"]})
    assert source.name == "s3:ids"
    assert source.strip_prefixes == ["/mnt/nfs"]
    assert source.fetcher.object_name("a/x.zip") == "core/a/x.zip"


def test_azure_source():
    pytest.importorskip("azure.storage.blob")
    source = build_source({"type": "azure", "container": "ids", "credential_source": "connection_string"})
    assert source.name == "azure:ids"                     # credential_source is a setting, not a secret
    assert source.fetcher.credential_source == "connection_string"


def test_gcs_source():
    pytest.importorskip("google.cloud.storage")
    source = build_source({"type": "gcs", "bucket": "ids", "prefix": "core"})
    assert source.name == "gcs:ids"


def test_unknown_source_type_lists_the_options():
    with pytest.raises(ValueError, match="Options: \\['azure', 'gcs', 'nfs', 's3'\\]"):
        build_source({"type": "dropbox"})


@pytest.mark.parametrize("spec, env", [
    ({"type": "s3", "bucket": "b", "aws_secret_access_key": "x"}, "AWS_SECRET_ACCESS_KEY"),
    ({"type": "s3", "bucket": "b", "access_key": "x"}, "AWS_ACCESS_KEY_ID"),
    ({"type": "azure", "container": "c", "connection_string": "x"}, "AZURE_STORAGE_CONNECTION_STRING"),
    ({"type": "azure", "container": "c", "sas_token": "x"}, "AZURE_STORAGE_SAS_TOKEN"),
    ({"type": "gcs", "bucket": "b", "credentials_file": "key.json"}, "GOOGLE_APPLICATION_CREDENTIALS"),
], ids=["s3-secret", "s3-access-key", "azure-connection-string", "azure-sas", "gcs-key-file"])
def test_credentials_in_yaml_are_refused(spec, env):
    with pytest.raises(ValueError, match=f"never go in YAML.*{env}"):
        build_source(spec)


# ---------------------------------------------------------------- an experiment reading from a bucket

LABELS = [(101, "front", ("national_id", "TN", "front")), (102, "back", ("national_id", "TN", "back")),
          (103, "front", ("national_id", "MA", "front")), (104, "front", ("driving_license", "MA", "front"))]


def _experiment(tmp_path, nfs_root, source=None):
    """A ground truth of NFS-style zips + an experiment YAML using the test fakes as detector and classifier."""
    rows = ["transaction_id,image_id,image_path,document_type,issuing_country,document_side"]
    for txn, side, label in LABELS:
        path = write_nfs_zip(nfs_root / "svc", txn, side, COLOR_OF[label])
        rows.append(f"{txn},{side},{path.as_posix()},{','.join(label)}")
    truth = tmp_path / "ground_truth.csv"
    truth.write_text("\n".join(rows) + "\n", encoding="utf-8")
    experiment = {
        "experiment_name": "src", "ground_truth": str(truth), "output_dir": str(tmp_path / "runs"),
        "configurations": [{"name": "fake", "detector": {"type": "fixed"}, "classifiers": [{"type": "color"}]}],
    }
    if source:
        experiment["source"] = source
    path = tmp_path / ("bucket.yaml" if source else "disk.yaml")
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return path


def _comparable(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("latency")}


def test_same_results_from_disk_and_from_a_bucket(tmp_path, monkeypatch):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setitem(config.REGISTRY["detector"], "fixed", "fakes:FixedDetector")
    monkeypatch.setitem(config.REGISTRY["classifier"], "color", "fakes:ColorClassifier")
    for name, value in [("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                        ("AWS_DEFAULT_REGION", "us-east-1")]:
        monkeypatch.setenv(name, value)
    nfs_root = tmp_path / "nfs"
    disk_yaml = _experiment(tmp_path, nfs_root)
    bucket_yaml = _experiment(tmp_path, nfs_root, {"type": "s3", "bucket": "ids", "region": "us-east-1",
                                                   "strip_prefixes": [nfs_root.as_posix()]})

    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="ids")
        for zip_path in (nfs_root / "svc").glob("*.zip"):           # the "cloud copy": same key as the NFS file
            s3.put_object(Bucket="ids", Key=f"svc/{zip_path.name}", Body=zip_path.read_bytes())
        from_bucket = run_experiment(bucket_yaml)
    from_disk = run_experiment(disk_yaml)

    assert _comparable(from_bucket.summaries[0]) == _comparable(from_disk.summaries[0])
    assert from_bucket.summaries[0]["country_accuracy"] == 1.0
    assert "Images read from: s3:ids" in (from_bucket.output_dir / "comparison.md").read_text(encoding="utf-8")
    assert "Images read from: local files" in (from_disk.output_dir / "comparison.md").read_text(encoding="utf-8")


def test_missing_cloud_object_is_counted_as_a_failed_fetch(tmp_path, monkeypatch):
    monkeypatch.setitem(config.REGISTRY["detector"], "fixed", "fakes:FixedDetector")
    monkeypatch.setitem(config.REGISTRY["classifier"], "color", "fakes:ColorClassifier")
    monkeypatch.setitem(config.REGISTRY["source"], "empty", "test_cloud_source:empty_source")
    result = run_experiment(_experiment(tmp_path, tmp_path / "nfs", {"type": "empty"}))
    report = (result.output_dir / "comparison.md").read_text(encoding="utf-8")
    assert f"{len(LABELS)} could not be loaded" in report             # nothing crashes; every image is reported


def empty_source():
    return FetcherSource(DictFetcher({}))


def test_payload_formats_in_a_bucket_give_the_same_image():
    """A bucket may hold the NFS zip or base64 text of the JPEG (Q5): the source reads both the same way."""
    import base64

    jpeg = image_bytes((255, 0, 0), "JPEG")
    source = FetcherSource(DictFetcher({"a/zip.zip": zip_bytes(jpeg), "a/b64.zip": base64.b64encode(jpeg)}))
    hashes = {source.fetch(1, "front", key).image_hash for key in ("a/zip.zip", "a/b64.zip")}
    assert len(hashes) == 1 and None not in hashes


@pytest.mark.parametrize("name", ["minio_eval", "azurite_eval"])
def test_emulator_experiment_configs_build_their_source(name):
    """configs/minio_eval.yaml and configs/azurite_eval.yaml: nfs_eval.yaml + a source, no secret in them."""
    pytest.importorskip("boto3" if name == "minio_eval" else "azure.storage.blob")
    experiment = config.load_yaml(f"configs/{name}.yaml")
    nfs = config.load_yaml("configs/nfs_eval.yaml")
    assert {k: v for k, v in experiment.items() if k not in ("source", "experiment_name")} == \
           {k: v for k, v in nfs.items() if k != "experiment_name"}           # the same experiment otherwise
    source = build_source(experiment["source"])
    assert source.strip_prefixes == ["/mnt/nfs"]
