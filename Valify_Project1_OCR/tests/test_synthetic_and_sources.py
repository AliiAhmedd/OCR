import csv

from id_classifier.sources import LocalFolderSource, read_image_file
from id_classifier.synthetic import LABEL_KEY, SyntheticSource
from id_classifier.types import NO_DOCUMENT, label_problems


def test_synthetic_is_deterministic():
    a, b = SyntheticSource(per_country=1, negatives=3, seed=1), SyntheticSource(per_country=1, negatives=3, seed=1)
    for (tid, iid, ref) in a.list_keys():
        assert a.fetch(tid, iid, ref).image_hash == b.fetch(tid, iid, ref).image_hash


def test_synthetic_seed_changes_images():
    a, b = SyntheticSource(per_country=1, negatives=0, seed=1), SyntheticSource(per_country=1, negatives=0, seed=2)
    tid, iid, ref = a.list_keys()[0]
    assert a.fetch(tid, iid, ref).image_hash != b.fetch(tid, iid, ref).image_hash


def test_synthetic_plan_has_fronts_backs_passports_and_negatives():
    source = SyntheticSource(per_country=2, negatives=3, seed=0, countries=["TR", "TN"])
    kinds = {(s.document_type, s.document_side) for s in source.samples}
    assert kinds == {("national_id", "front"), ("national_id", "back"), ("passport", "n/a"), (NO_DOCUMENT, "n/a")}
    for s in source.samples:
        assert label_problems(s.document_type, s.issuing_country, s.document_side) == []
    # one transaction carries two images (front + back)
    first_tid = source.samples[0].transaction_id
    assert [s.image_id for s in source.samples if s.transaction_id == first_tid] == ["front", "back"]


def test_fetched_image_carries_hidden_label():
    source = SyntheticSource(per_country=1, negatives=0, seed=0, countries=["TR"])
    tid, iid, ref = source.list_keys()[0]
    record = source.fetch(tid, iid, ref)
    assert record.fetch_status == "ok" and record.image.mode == "RGB"
    assert '"issuing_country": "TR"' in record.image.info[LABEL_KEY]


def test_unknown_synthetic_key_is_a_failed_record_not_an_exception():
    record = SyntheticSource(per_country=1, negatives=0).fetch(1, "front", "x")
    assert record.fetch_status == "failed" and record.failure_reason == "unknown_synthetic_key"


def test_write_dataset_csv_matches_files(synthetic_csv):
    with synthetic_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 28
    for row in rows:
        assert (synthetic_csv.parent / row["image_path"]).is_file()


def test_local_folder_source(tmp_path, synthetic_csv):
    good = synthetic_csv.parent / "images"
    (tmp_path / "123_front.png").write_bytes(next(good.glob("*.png")).read_bytes())
    (tmp_path / "124_back.png").write_bytes(b"this is not an image")   # corrupt file
    (tmp_path / "holiday.png").write_bytes(b"x")                       # name does not follow <tid>_<image_id>
    (tmp_path / "notes.txt").write_text("ignored")                     # not an image

    source = LocalFolderSource(tmp_path)
    keys = source.list_keys()
    assert [(k[0], k[1]) for k in keys] == [(123, "front"), (124, "back")]

    ok = source.fetch(*keys[0])
    assert ok.fetch_status == "ok" and len(ok.image_hash) == 64
    broken = source.fetch(*keys[1])
    assert broken.fetch_status == "failed" and broken.failure_reason.startswith("decode_error")


def test_missing_file_is_a_failed_record(tmp_path):
    record = read_image_file(1, "front", tmp_path / "missing.png")
    assert record.fetch_status == "failed" and record.failure_reason.startswith("read_error")
