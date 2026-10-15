"""build_ground_truth leaves out the images in configs/excluded_images.csv (labels found wrong by a human)."""

import csv
from pathlib import Path

from fakes import write_nfs_zip
from id_classifier.ground_truth import build_ground_truth, read_excluded


def test_excluded_image_never_enters_the_ground_truth(tmp_path):
    # one image per folder: solid-colour fakes count as the same picture within a folder (de-duplication)
    kept = write_nfs_zip(tmp_path / "nfs" / "dza_nid_ocr", 1, "front", (255, 0, 0))
    wrong = write_nfs_zip(tmp_path / "nfs" / "mar_nid_ocr", 2, "back", (0, 0, 255))   # e.g. an Egyptian card
    excluded = tmp_path / "excluded.csv"
    excluded.write_text(f"key,reason,reported\nmar_nid_ocr/{wrong.name},mislabelled,2026-10-14\n", encoding="utf-8")

    out = build_ground_truth(tmp_path / "nfs", tmp_path / "gt.csv", excluded=excluded,
                             folders={"dza_nid_ocr": ("national_id", "DZ"), "mar_nid_ocr": ("national_id", "MA")})

    paths = [row["image_path"] for row in csv.DictReader(out.open(encoding="utf-8"))]
    assert [Path(p).name for p in paths] == [kept.name]
    assert read_excluded(excluded) == {f"mar_nid_ocr/{wrong.name}": "mislabelled"}


def test_the_reported_moroccan_folder_image_is_excluded():
    """progress.md row 30: transaction 1575124 in mar_nid_ocr is an Egyptian national ID."""
    assert "mar_nid_ocr/1575124_2026-10-01_15-07-31-724250_back_img.zip" in read_excluded()


def test_relabelled_image_gets_the_human_label(tmp_path):
    from id_classifier.ground_truth import read_relabelled

    selfie = write_nfs_zip(tmp_path / "nfs" / "dza_nid_ocr", 3, "front", (0, 255, 0))
    relabelled = tmp_path / "relabelled.csv"
    relabelled.write_text("key,document_type,issuing_country,document_side,reason,reported\n"
                          f"dza_nid_ocr/{selfie.name},none,unknown,n/a,a selfie,2026-10-14\n", encoding="utf-8")

    out = build_ground_truth(tmp_path / "nfs", tmp_path / "gt.csv", excluded=None, relabelled=relabelled,
                             folders={"dza_nid_ocr": ("national_id", "DZ")})

    [row] = list(csv.DictReader(out.open(encoding="utf-8")))
    assert (row["document_type"], row["issuing_country"], row["document_side"]) == ("none", "unknown", "n/a")
    assert read_relabelled()["dza_nid_ocr/1572908_2026-09-14_14-21-11-332935_front_img.zip"] == ("none", "unknown", "n/a")
