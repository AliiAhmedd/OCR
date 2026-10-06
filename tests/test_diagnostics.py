"""find-duplicates and sweep-detector, on NFS-shaped zips. picture(n) in fakes.py: same n = same picture,
a different JPEG quality = the same picture re-saved (different bytes, same look)."""

import csv
from pathlib import Path

import yaml

from fakes import FixedDetector, write_picture_zip
from id_classifier.__main__ import main
from id_classifier.config import load_yaml
from id_classifier.duplicates import find_duplicates
from id_classifier.sources import sample_nfs_records
from id_classifier.sweep import sweep_detectors

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_find_duplicates_exact_near_and_across_folders(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1)              # exact copy (same bytes)
    write_picture_zip(nfs / "tun_nid_ocr", 3, "front", 1, quality=60)  # near copy: re-saved, other bytes
    write_picture_zip(nfs / "tun_nid_ocr", 4, "back", 2)
    write_picture_zip(nfs / "mar_nid_ocr", 5, "back", 2)               # same picture in ANOTHER folder
    write_picture_zip(nfs / "mar_nid_ocr", 6, "front", 3)              # unique

    out = find_duplicates(nfs, ["tun_nid_ocr", "mar_nid_ocr"], tmp_path / "out")

    summary = {row["folder"]: row for row in read_csv(out / "summary.csv")}
    tun, mar = summary["tun_nid_ocr"], summary["mar_nid_ocr"]
    assert (tun["zips"], tun["distinct_exact"], tun["distinct_near"]) == ("4", "3", "2")   # pictures 1 and 2
    assert (mar["zips"], mar["distinct_exact"], mar["distinct_near"]) == ("2", "2", "2")
    groups = {row["uploads"]: row for row in read_csv(out / "groups.csv")}
    near = groups["tun_nid_ocr/1_front tun_nid_ocr/2_front tun_nid_ocr/3_front"]
    assert (near["copies"], near["kind"], near["max_distance"]) == ("3", "near", "0")
    cross = groups["mar_nid_ocr/5_back tun_nid_ocr/4_back"]
    assert (cross["kind"], cross["cross_folder"]) == ("exact", "True")
    assert len(groups) == 2                                              # picture 3 is unique
    pairs = read_csv(out / "close_pairs.csv")
    assert all(p["duplicate"] == "True" for p in pairs)                  # different pictures are far apart
    histogram = {row["distance"]: row for row in read_csv(out / "distances.csv")}
    assert histogram["0"]["pairs_same_folder"] == "1"                    # the q90 / q60 pair of picture 1
    assert not (out / "review").exists()                                 # review images only when asked


def test_different_pictures_are_never_merged(tmp_path):
    nfs = tmp_path / "nfs"
    for number in range(1, 9):
        write_picture_zip(nfs / "tun_nid_ocr", number, "front", number)
    out = find_duplicates(nfs, ["tun_nid_ocr"], tmp_path / "out")
    assert read_csv(out / "summary.csv")[0]["distinct_near"] == "8"
    assert read_csv(out / "groups.csv") == []


def test_review_images_show_close_pairs(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1, quality=60)
    out = find_duplicates(nfs, ["tun_nid_ocr"], tmp_path / "out", review_images=True)
    assert [p.name for p in (out / "review").iterdir()] == ["000_dup_1_front__2_front.jpg"]


def test_find_duplicates_cli(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    out = tmp_path / "dups"
    assert main(["find-duplicates", "--root", str(nfs), "--folders", "tun_nid_ocr", "--out", str(out),
                 "--max-distance", "5"]) == 0
    assert all((out / name).exists() for name in ("summary.csv", "groups.csv", "distances.csv", "close_pairs.csv"))


def test_sampling_skips_resaved_copies(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1, quality=60)   # same picture, other bytes
    write_picture_zip(nfs / "tun_nid_ocr", 3, "back", 2)
    distinct = list(sample_nfs_records(nfs, "tun_nid_ocr", per_folder=10))
    assert len(distinct) == 2
    every = list(sample_nfs_records(nfs, "tun_nid_ocr", per_folder=10, unique_only=False))
    assert len(every) == 3


def test_sweep_counts_per_threshold_and_skips_duplicates(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1)        # duplicate: skipped with unique_only
    write_picture_zip(nfs / "tun_nid_ocr", 3, "back", 2)
    detectors = {"two_boxes": FixedDetector([0.30, 0.12]), "nothing": FixedDetector([])}

    run_dir = sweep_detectors(detectors, [0.05, 0.25], nfs, ["tun_nid_ocr"], tmp_path / "sweeps", per_folder=10)

    rows = {(r["prompt_set"], r["threshold"]): r for r in read_csv(run_dir / "sweep.csv")}
    low, high = rows[("two_boxes", "0.05")], rows[("two_boxes", "0.25")]
    assert (low["images"], low["detected"], low["multiple"]) == ("2", "2", "2")    # 2 distinct images
    assert (high["detected"], high["multiple"]) == ("2", "0")                      # only the 0.30 box is left
    assert rows[("nothing", "0.05")]["detected"] == "0"
    assert len(read_csv(run_dir / "per_image.csv")) == 4                           # 2 images x 2 detectors
    assert "| two_boxes | tun_nid_ocr | 2/2 (2) | 2/2 |" in (run_dir / "sweep.md").read_text(encoding="utf-8")


def test_sweep_keeps_duplicates_when_asked(tmp_path):
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1)
    run_dir = sweep_detectors({"f": FixedDetector([0.5])}, [0.1], nfs, ["tun_nid_ocr"], tmp_path / "s",
                              unique_only=False)
    assert read_csv(run_dir / "sweep.csv")[0]["images"] == "2"


def test_sweep_cli_builds_one_detector_per_prompt_set(tmp_path, monkeypatch):
    """The CLI turns each prompt set into a detector at the lowest threshold (FixedDetector stands in for YOLO)."""
    built = []

    def fake_build(kind, spec):
        built.append(spec)
        return FixedDetector([0.2], name=spec["name"])

    monkeypatch.setattr("id_classifier.__main__.build", fake_build)
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    config = tmp_path / "sweep.yaml"
    config.write_text(yaml.safe_dump({"sweep": {
        "root": str(nfs), "folders": ["tun_nid_ocr"], "output_dir": str(tmp_path / "sweeps"),
        "thresholds": [0.25, 0.05], "detector": {"type": "yolo", "weights": "w.pt"},
        "prompt_sets": {"a": ["identity card"], "b": ["card"]},
    }}), encoding="utf-8")

    assert main(["sweep-detector", "--config", str(config)]) == 0
    assert [(s["name"], s["prompts"], s["confidence"]) for s in built] == [("a", ["identity card"], 0.05),
                                                                         ("b", ["card"], 0.05)]
    assert len(list((tmp_path / "sweeps").glob("sweep_*/sweep.md"))) == 1


def test_detector_sweep_config_is_valid():
    sweep = load_yaml(REPO_ROOT / "configs" / "detector_sweep.yaml")["sweep"]
    assert sweep["detector"]["type"] == "yolo"
    assert sweep["unique_only"] is True
    assert "billing_reports" not in sweep["folders"]
