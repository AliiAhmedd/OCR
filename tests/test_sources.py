import csv
import zipfile

from fakes import write_nfs_zip
from id_classifier.ground_truth import build_ground_truth
from id_classifier.sources import LocalFolderSource, NfsZipSource, read_image_file


def test_local_folder_source(tmp_path, images_dir):
    (tmp_path / "123_front.png").write_bytes((images_dir / "101_front.png").read_bytes())
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


def test_nfs_zip_source_lists_and_reads_zips(tmp_path):
    folder = tmp_path / "tun_nid_ocr"
    write_nfs_zip(folder, 1575232, "front", (255, 0, 0))
    write_nfs_zip(folder, 1575232, "back", (0, 255, 0))
    (folder / "1575233_whatever.zip").write_bytes(b"x")                # unexpected name: skipped

    source = NfsZipSource(tmp_path, ["tun_nid_ocr"])
    keys = source.list_keys()
    assert [(k[0], k[1]) for k in keys] == [(1575232, "back"), (1575232, "front")]
    record = source.fetch(*keys[0])
    assert record.fetch_status == "ok" and record.image.mode == "RGB"


def test_zip_without_original_jpeg_is_a_failed_record(tmp_path):
    path = tmp_path / "1_2026-10-01_00-00-00-000000_front_img.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("something_else.jpeg", b"x")
    record = read_image_file(1, "front", path)
    assert record.fetch_status == "failed" and record.failure_reason.startswith("zip_error")


def test_build_ground_truth_labels_from_folder_and_name(tmp_path):
    root = tmp_path / "nfs"
    for txn in (1, 2, 3):
        write_nfs_zip(root / "tun_nid_ocr", txn, "front", (255, 0, 0))
    write_nfs_zip(root / "mar_driver_license_ocr", 9, "back", (255, 255, 0))
    csv_path = build_ground_truth(root, tmp_path / "gt.csv", per_folder=2)   # absent folders are skipped
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3                                                    # 2 sampled from tun + 1 from mar
    tun = [r for r in rows if r["issuing_country"] == "TN"]
    assert len(tun) == 2 and all(r["document_type"] == "national_id" and r["document_side"] == "front" for r in tun)
    [mar] = [r for r in rows if r["issuing_country"] == "MA"]
    assert (mar["document_type"], mar["document_side"]) == ("driving_license", "back")
