import csv
from pathlib import Path

import pytest
import yaml

from fakes import write_nfs_zip, write_picture_zip
from id_classifier.__main__ import main
from id_classifier.config import build_detector, load_yaml
from id_classifier.evaluate import SelectionRule, read_ground_truth, run_experiment, select_winner

REPO_ROOT = Path(__file__).resolve().parents[1]

PERFECT = {"name": "perfect", "detector": {"type": "full_image"}, "classifiers": [{"type": "fake", "name": "p"}]}
WRONG = {"name": "wrong", "classifiers": [{"type": "fake", "name": "w", "wrong_country": "DZ"}]}
BROKEN = {"name": "broken", "classifiers": [{"type": "fake", "name": "b", "parse_error": True}]}


def test_perfect_classifier_scores_perfectly(ground_truth_csv, write_experiment):
    result = run_experiment(write_experiment(ground_truth_csv, [PERFECT, WRONG, BROKEN]))
    perfect = next(s for s in result.summaries if s["config"] == "perfect")
    for metric in ("type_accuracy", "country_accuracy", "country_accuracy_macro", "side_accuracy",
                   "negative_rejection_rate", "coverage", "accepted_accuracy"):
        assert perfect[metric] == 1.0, metric
    assert perfect["parse_error_rate"] == 0.0
    assert perfect["review_rate"] == pytest.approx(1 / 5)   # only the image without a document (a human confirms "none")
    assert set(perfect["per_country"]) == {"TN", "MA"}
    assert result.winner == "perfect"

    wrong = next(s for s in result.summaries if s["config"] == "wrong")
    assert wrong["country_accuracy"] == 0.0 and wrong["type_accuracy"] == 1.0
    broken = next(s for s in result.summaries if s["config"] == "broken")
    assert broken["parse_error_rate"] == 1.0


def test_output_files(ground_truth_csv, write_experiment):
    result = run_experiment(write_experiment(ground_truth_csv, [PERFECT]))
    out = result.output_dir
    for name in ("comparison.md", "comparison.csv", "per_country.csv", "experiment.yaml",
                 "confusion_perfect.csv", "confusion_perfect.md", "predictions_perfect.csv"):
        assert (out / name).is_file(), name
    assert "## Winner: **perfect**" in (out / "comparison.md").read_text(encoding="utf-8")
    with (out / "confusion_perfect.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    for row in rows[1:]:                         # perfect classifier: everything on the diagonal
        for label, value in zip(labels, row[1:]):
            assert (int(value) > 0) == (label == row[0])
    with (out / "predictions_perfect.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 5 * 2   # 5 images x 2 repeats


def test_selection_rule():
    base = {"negative_rejection_rate": 1.0, "parse_error_rate": 0.0, "country_accuracy_macro": 0.9, "latency_mean_ms": 50.0}
    summaries = [
        {**base, "config": "fast"},
        {**base, "config": "slow", "latency_mean_ms": 900.0},                  # same accuracy, slower
        {**base, "config": "best_but_accepts_empty", "country_accuracy_macro": 0.99, "negative_rejection_rate": 0.5},
        {**base, "config": "broken_json", "country_accuracy_macro": 0.99, "parse_error_rate": 0.1},
    ]
    winner, notes = select_winner(summaries, SelectionRule())
    assert winner == "fast"
    assert "negative_rejection_rate" in notes["best_but_accepts_empty"]
    assert "parse_error_rate" in notes["broken_json"]


def test_ground_truth_without_negatives_can_still_have_a_winner():
    summaries = [{"config": "only", "negative_rejection_rate": None, "parse_error_rate": 0.0,
                  "country_accuracy_macro": 0.8, "latency_mean_ms": 10.0}]
    assert select_winner(summaries, SelectionRule())[0] == "only"


def test_selection_rule_rejects_unknown_metric():
    with pytest.raises(ValueError):
        SelectionRule.from_dict({"primary_metric": "vibes"})


def test_bad_ground_truth_row_is_reported(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "transaction_id,image_id,image_path,document_type,issuing_country,document_side\n"
        "1,front,a.png,national_id,Turkey,front\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row 2"):
        read_ground_truth(path)


def test_nfs_eval_config_is_valid():
    """configs/nfs_eval.yaml must stay valid: the selection rule parses and every configuration's detector builds."""
    config = load_yaml(REPO_ROOT / "configs" / "nfs_eval.yaml")
    SelectionRule.from_dict(config["selection_rule"])
    assert config["ground_truth"]
    for spec in config.get("configurations") or []:
        build_detector(spec.get("detector"))


def test_cli_end_to_end(tmp_path, write_experiment):
    """build-ground-truth from an NFS-shaped folder, then evaluate and classify, through the CLI."""
    nfs = tmp_path / "nfs"
    write_nfs_zip(nfs / "tun_nid_ocr", 1, "front", (255, 0, 0))
    write_nfs_zip(nfs / "tun_nid_ocr", 1, "back", (0, 255, 0))
    write_nfs_zip(nfs / "mar_nid_ocr", 2, "front", (0, 0, 255))
    gt = tmp_path / "gt.csv"
    assert main(["build-ground-truth", "--root", str(nfs), "--out", str(gt)]) == 0
    config = write_experiment(
        gt, [PERFECT, WRONG],
        classify={
            "source": {"type": "nfs_zip", "root": str(nfs), "folders": ["tun_nid_ocr", "mar_nid_ocr"]},
            "classifiers": [{"type": "fake", "name": "p"}],
            "database_url": f"sqlite:///{(tmp_path / 'pipeline.db').as_posix()}",
        },
    )
    assert main(["evaluate", "--config", str(config)]) == 0
    assert main(["classify", "--config", str(config), "--run-id", "test-run"]) == 0
    [report] = (tmp_path / "runs").glob("test_*/comparison.md")
    assert "## Winner: **perfect**" in report.read_text(encoding="utf-8")


def test_preview_detector_writes_boxes_crops_and_summary(tmp_path):
    """preview-detector reads only the listed folders and saves one boxes image + crops per sampled image."""
    nfs = tmp_path / "nfs"
    write_picture_zip(nfs / "tun_nid_ocr", 1, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 2, "front", 1)
    write_picture_zip(nfs / "tun_nid_ocr", 3, "back", 2)
    write_picture_zip(nfs / "billing_reports", 9, "front", 3)        # not on the list: never read
    out = tmp_path / "preview"
    config = tmp_path / "preview.yaml"
    config.write_text(yaml.safe_dump({"preview": {
        "root": str(nfs), "folders": ["tun_nid_ocr"], "per_folder": 2, "output_dir": str(out),
        "detector": {"type": "full_image"},
    }}), encoding="utf-8")

    assert main(["preview-detector", "--config", str(config)]) == 0
    with (out / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2                                          # per_folder sample
    assert len({row["image_hash"] for row in rows}) == 2           # 1 and 2 are the same image: only one shown
    assert {row["folder"] for row in rows} == {"tun_nid_ocr"}
    assert all(row["regions"] == "1" and row["bboxes"] == "0,0,64,48" for row in rows)
    for row in rows:
        stem = f"{row['transaction_id']}_{row['image_id']}"
        assert (out / "tun_nid_ocr" / f"{stem}_boxes.jpg").exists()
        assert (out / "tun_nid_ocr" / f"{stem}_crop0.jpg").exists()
    assert not (out / "billing_reports").exists()


def test_yolo_preview_config_is_valid():
    preview = load_yaml(REPO_ROOT / "configs" / "yolo_preview.yaml")["preview"]
    assert preview["detector"]["type"] == "yolo"
    assert "billing_reports" not in preview["folders"]
