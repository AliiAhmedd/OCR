"""Three outcomes in routing + evaluation, hand-labeled file ingestion, and the kNN sweep config."""

import csv
import inspect

import pytest
import yaml
from PIL import Image

from fakes import COLOR_LABELS, COLOR_OF, ColorClassifier, image_bytes
from id_classifier import config, routing
from id_classifier.__main__ import main
from id_classifier.classifiers.knn import EmbeddingKnnClassifier
from id_classifier.evaluate import SelectionRule, run_experiment, select_winner
from id_classifier.labeled_files import (
    ground_truth_rows,
    merge_ground_truths,
    parse_file_name,
    read_labeled_folder,
    write_rows,
)
from id_classifier.routing import RoutingConfig, combine
from id_classifier.types import (
    OUTCOME_CLASSIFIED,
    OUTCOME_NO_DOCUMENT,
    OUTCOME_UNRECOGNISED,
    STATUS_OK,
    Prediction,
    Region,
)

# ---------------------------------------------------------------- routing: three outcomes

def region(index="0"):
    return Region(f"1:front:yoloe:{index}", 1, "front", (0, 0, 10, 10), 0.5 if index != "full" else 0.0, "yoloe")


def pred(label, confidence=0.95, outcome=OUTCOME_CLASSIFIED):
    return Prediction(*label, confidence, STATUS_OK, "{}", "knn", "v", 1.0, outcome=outcome)


def test_classified():
    r = combine(region(), [pred(("national_id", "TN", "front"))], RoutingConfig())
    assert r.outcome == OUTCOME_CLASSIFIED and not r.needs_review


def test_unrecognised_keeps_best_guess_and_goes_to_review():
    r = combine(region(), [pred(("national_id", "DZ", "front"), 0.6, OUTCOME_UNRECOGNISED)], RoutingConfig())
    assert r.outcome == OUTCOME_UNRECOGNISED and r.issuing_country == "DZ"
    assert routing.UNRECOGNISED in r.reasons and routing.UNKNOWN_COUNTRY not in r.reasons and r.needs_review


def test_unrecognised_without_any_detected_card_is_no_document():
    r = combine(region("full"), [pred(("national_id", "DZ", "front"), 0.3, OUTCOME_UNRECOGNISED)], RoutingConfig())
    assert r.outcome == OUTCOME_NO_DOCUMENT
    assert (r.document_type, r.issuing_country, r.document_side) == ("none", "unknown", "n/a")
    assert r.reasons[-1] == routing.NO_DOCUMENT_FOUND


def test_recognised_close_up_without_a_box_is_still_classified():
    r = combine(region("full"), [pred(("national_id", "TN", "front"))], RoutingConfig())
    assert r.outcome == OUTCOME_CLASSIFIED and r.issuing_country == "TN"


# ---------------------------------------------------------------- labeled file names

@pytest.mark.parametrize("name, label", [
    ("egy_nid_front_img_2.jpg", ("national_id", "EG", "front")),
    ("uae_mrz_back_img.png", ("passport", "AE", "back")),               # mrz = passport data page
    ("nig_mrz_front_img.jpg", ("passport", "NG", "front")),             # nig = Nigeria
    ("syr_mrz_front_img_1.webp", ("passport", "SY", "front")),
    ("egy_other_front_img.jpg", ("none", "unknown", "n/a")),            # other = blank / no document
    ("uae_nid_front_img.jpg", ("residence_permit", "AE", "front")),     # same class as the NFS UAE folder
    ("mar_driver_license_back_img_3.jpg", ("driving_license", "MA", "back")),
    ("IRQ_NID_FRONT_IMG.JPG", ("national_id", "IQ", "front")),          # case does not matter
])
def test_file_names(tmp_path, name, label):
    f = parse_file_name(tmp_path / name)
    assert (f.document_type, f.issuing_country, f.document_side) == label


def test_bad_names_are_all_reported_at_once(tmp_path):
    for name in ("xyz_nid_front_img.jpg", "egy_visa_front_img.jpg", "egy_nid_top_img.jpg", "holiday.jpg"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    with pytest.raises(ValueError) as error:
        read_labeled_folder(tmp_path)
    message = str(error.value)
    for expected in ("unknown country 'xyz'", "unknown docType 'visa'", "unknown side 'top'", "holiday.jpg"):
        assert expected in message
    assert "notes.txt" not in message


def test_front_and_back_of_one_card_share_a_transaction(tmp_path):
    for name in ("egy_nid_front_img_1.jpg", "egy_nid_back_img_1.jpg", "egy_nid_front_img_2.jpg",
                 "uae_mrz_front_img.jpg", "uae_mrz_front_img-copy.jpg"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ValueError):                                   # "-copy" is not a valid name
        read_labeled_folder(tmp_path)
    (tmp_path / "uae_mrz_front_img-copy.jpg").unlink()
    rows = ground_truth_rows(read_labeled_folder(tmp_path))
    by_name = {r[2].rsplit("/", 1)[1]: r for r in rows}
    assert by_name["egy_nid_front_img_1.jpg"][0] == by_name["egy_nid_back_img_1.jpg"][0]
    assert by_name["egy_nid_front_img_1.jpg"][0] != by_name["egy_nid_front_img_2.jpg"][0]
    assert len({(r[0], r[1]) for r in rows}) == len(rows)            # keys are unique


def test_merge_refuses_duplicates(tmp_path):
    a = write_rows(tmp_path / "a.csv", [[1, "front", "/x/1.jpg", "national_id", "TN", "front"]])
    b = write_rows(tmp_path / "b.csv", [[2, "front", "/x/1.jpg", "national_id", "TN", "front"]])
    with pytest.raises(ValueError, match="already in"):
        merge_ground_truths([a, b], tmp_path / "m.csv")


def test_ingest_cli_merges_with_existing_ground_truth(tmp_path):
    folder = tmp_path / "DocumentData"
    folder.mkdir()
    for name in ("egy_nid_front_img.jpg", "nig_mrz_front_img.jpg"):
        (folder / name).write_bytes(image_bytes((1, 2, 3), "JPEG"))
    existing = write_rows(tmp_path / "gt.csv", [[1575232, "front", "/mnt/nfs/tun/1.zip", "national_id", "TN", "front"]])
    out, merged = tmp_path / "doc.csv", tmp_path / "plus.csv"
    assert main(["ingest-labeled-folder", "--folder", str(folder), "--out", str(out),
                 "--merge-with", str(existing), "--merged-out", str(merged)]) == 0
    with merged.open(encoding="utf-8") as f:
        countries = [row["issuing_country"] for row in csv.DictReader(f)]
    assert countries == ["TN", "EG", "NG"]


# ---------------------------------------------------------------- evaluation report

class MoroccoUnrecognised(ColorClassifier):
    """Like ColorClassifier, but every Moroccan document comes back 'unrecognised' (a small class)."""

    def classify(self, crop: Image.Image) -> Prediction:
        p = super().classify(crop)
        if p.issuing_country == "MA":
            p.outcome, p.confidence = OUTCOME_UNRECOGNISED, 0.5
        return p


def test_report_has_outcomes_and_small_classes(tmp_path, monkeypatch):
    monkeypatch.setitem(config.REGISTRY["detector"], "fixed", "fakes:FixedDetector")
    monkeypatch.setitem(config.REGISTRY["classifier"], "ma_unrec", "test_outcomes_and_labeled_files:MoroccoUnrecognised")
    rows = ["transaction_id,image_id,image_path,document_type,issuing_country,document_side"]
    for i, label in enumerate([l for l in COLOR_LABELS.values() if l[0] != "none"] * 3):
        path = tmp_path / f"{i}.png"
        path.write_bytes(image_bytes(COLOR_OF[label]))
        rows.append(f"{i + 1},front,{path.as_posix()},{','.join(label)}")
    truth = tmp_path / "gt.csv"
    truth.write_text("\n".join(rows) + "\n", encoding="utf-8")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(yaml.safe_dump({
        "experiment_name": "t", "ground_truth": str(truth), "output_dir": str(tmp_path / "runs"),
        "selection_rule": {"primary_metric": "accepted_accuracy", "tie_breaker": "review_rate", "min_coverage": 0.9},
        "configurations": [{"name": "c", "detector": {"type": "fixed"}, "classifiers": [{"type": "ma_unrec"}]}],
    }), encoding="utf-8")

    result = run_experiment(experiment)
    s = result.summaries[0]
    assert s["unrecognised_rate"] == pytest.approx(0.5)              # MA id + MA licence = 6 of 12 documents
    assert s["accepted_wrong_country_rate"] == 0.0 and s["accepted_accuracy"] == 1.0
    assert s["per_class"]["national_id/MA"] == {"n": 3, "classified_correctly": 0, "unrecognised": 3,
                                                 "classified_as_other_class": 0, "auto_accepted": 0,
                                                 "auto_accepted_correctly": 0}
    assert result.winner is None                                       # coverage 0.5 < min_coverage 0.9
    report = (result.output_dir / "comparison.md").read_text(encoding="utf-8")
    assert "## Small classes (fewer than 10 images)" in report and "coverage 0.500 < 0.9" in report
    assert "UNRECOGNISED" in (result.output_dir / "confusion_c.md").read_text(encoding="utf-8")
    assert (result.output_dir / "per_class.csv").is_file()


def test_min_coverage_is_optional():
    base = {"negative_rejection_rate": None, "parse_error_rate": 0.0, "accepted_accuracy": 0.9, "review_rate": 0.1,
            "coverage": 0.2, "config": "a"}
    assert select_winner([base], SelectionRule.from_dict({"primary_metric": "accepted_accuracy",
                                                          "tie_breaker": "review_rate"}))[0] == "a"


# ---------------------------------------------------------------- the sweep config

def test_knn_sweep_config_is_valid():
    cfg = config.load_yaml("configs/knn_sweep.yaml")
    rule = SelectionRule.from_dict(cfg["selection_rule"])
    assert rule.primary_metric == "accepted_accuracy" and rule.min_coverage == 0.7
    names = [c["name"] for c in cfg["configurations"]]
    assert len(names) == len(set(names)) and names[0] == "A_baseline"
    allowed = set(inspect.signature(EmbeddingKnnClassifier).parameters) | {"type"}
    for spec in cfg["configurations"]:
        [knn] = spec["classifiers"]
        assert set(knn) <= allowed, set(knn) - allowed                # every YAML key is a real setting
        assert knn["references"] == cfg["ground_truth"]
        assert knn.get("nn_fallback", False) == (spec["name"] != "A_baseline")
