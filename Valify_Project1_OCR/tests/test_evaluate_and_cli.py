import csv
from pathlib import Path

import pytest

from id_classifier.__main__ import main
from id_classifier.config import build_classifiers, build_detector, load_yaml
from id_classifier.evaluate import SelectionRule, read_ground_truth, run_experiment, select_winner

REPO_ROOT = Path(__file__).resolve().parents[1]

PERFECT = {"name": "perfect", "detector": {"type": "full_image"}, "classifiers": [{"type": "mock", "name": "p"}]}
NOISY = {"name": "noisy", "classifiers": [{"type": "mock", "name": "n", "error_rate": 0.5, "parse_error_rate": 0.1, "seed": 4}]}


def test_perfect_mock_scores_perfectly(synthetic_csv, write_experiment):
    result = run_experiment(write_experiment(synthetic_csv, [PERFECT, NOISY]))
    perfect = next(s for s in result.summaries if s["config"] == "perfect")
    for metric in ("type_accuracy", "country_accuracy", "country_accuracy_macro", "side_accuracy",
                   "negative_rejection_rate", "coverage", "accepted_accuracy"):
        assert perfect[metric] == 1.0, metric
    assert perfect["parse_error_rate"] == 0.0
    assert perfect["review_rate"] == pytest.approx(3 / 28)   # only the 3 negatives (a human confirms "none")
    assert set(perfect["per_country"]) == {"TR", "TN", "SA", "JO", "SD"}
    assert result.winner == "perfect"

    noisy = next(s for s in result.summaries if s["config"] == "noisy")
    assert noisy["country_accuracy"] < 1.0 and noisy["parse_error_rate"] > 0.0


def test_output_files(synthetic_csv, write_experiment):
    result = run_experiment(write_experiment(synthetic_csv, [PERFECT]))
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
        assert len(list(csv.DictReader(f))) == 28 * 2   # 28 images x 2 repeats


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


def test_demo_config_builds():
    """configs/demo.yaml must stay valid: every configuration builds with core dependencies only."""
    config = load_yaml(REPO_ROOT / "configs" / "demo.yaml")
    SelectionRule.from_dict(config["selection_rule"])
    for spec in config["configurations"]:
        build_detector(spec.get("detector"))
        build_classifiers(spec["classifiers"])


def test_cli_end_to_end(tmp_path, write_experiment):
    """Definition of done: generate-synthetic, then evaluate and classify, through the CLI."""
    data = tmp_path / "synthetic"
    assert main(["generate-synthetic", "--out", str(data), "--per-country", "1", "--negatives", "3"]) == 0
    config = write_experiment(
        data / "ground_truth.csv", [PERFECT, NOISY],
        classify={
            "source": {"type": "local_folder", "path": str(data / "images")},
            "classifiers": [{"type": "mock", "name": "p"}],
            "database_url": f"sqlite:///{(tmp_path / 'pipeline.db').as_posix()}",
        },
    )
    assert main(["evaluate", "--config", str(config)]) == 0
    assert main(["classify", "--config", str(config), "--run-id", "test-run"]) == 0
    assert list((tmp_path / "runs").glob("test_*/comparison.md"))
