"""The "Beyond <biggest country>" report section: what is auto-accepted outside the country that dominates the
ground truth (150 TN of 173 documents), live in every run and backfilled into older runs."""

from id_classifier import config
from id_classifier.evaluate import backfill_beyond_majority, beyond_majority, run_experiment
from test_cloud_source import _experiment


def row(true, pred, needs_review):
    return {"repeat": "1", "true_type": true[0], "true_country": true[1], "true_side": true[2],
            "pred_type": pred[0], "pred_country": pred[1], "pred_side": pred[2], "needs_review": str(needs_review)}


TN = ("national_id", "TN", "front")
MA = ("national_id", "MA", "back")
AE = ("residence_permit", "AE", "back")
BLANK = ("none", "unknown", "n/a")


def test_counts_only_the_other_countries():
    rows = [row(TN, TN, False)] * 3 + [
        row(MA, AE, False),          # auto-accepted with the wrong country: the costliest error
        row(AE, AE, False),          # auto-accepted and right
        row(AE, AE, True),           # reviewed
        row(BLANK, BLANK, False),    # no document: not counted
    ]
    majority, lines, csv_rows = beyond_majority({"c": rows})

    assert majority == "TN"
    assert csv_rows == [["c", 3, 2, 1, 1, 2 / 3, 0.5]]   # documents, accepted, correct, wrong country, coverage, acc
    table = "\n".join(lines)
    assert "| c | 1/1/2 | 0/1/1 |" in table             # per country AE, MA: correct / accepted / documents


def test_run_report_has_the_section_and_backfill_replaces_it(tmp_path, monkeypatch):
    monkeypatch.setitem(config.REGISTRY["detector"], "fixed", "fakes:FixedDetector")
    monkeypatch.setitem(config.REGISTRY["classifier"], "color", "fakes:ColorClassifier")
    result = run_experiment(_experiment(tmp_path, tmp_path / "nfs"))
    report = result.output_dir / "comparison.md"
    written = report.read_text(encoding="utf-8")
    assert written.count("## Beyond ") == 1 and (result.output_dir / "beyond_majority.csv").exists()

    backfill_beyond_majority(result.output_dir)              # an older run: same section, never twice
    assert report.read_text(encoding="utf-8").count("## Beyond ") == 1
