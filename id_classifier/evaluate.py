"""Evaluation harness: run every configuration of an experiment YAML on a labeled set and compare them.

Input
- a ground-truth CSV (GROUND_TRUTH_COLUMNS; image_path relative to the CSV's folder)
- an experiment YAML: routing settings, the selection rule, and a list of configurations
  (detector x classifiers), each run `repeats` times

Output (one new folder per run, under output_dir)
- comparison.md / comparison.csv   one row per configuration + the winner according to the selection rule
- per_country.csv                  accuracy per true country, per configuration
- confusion_<config>.csv/.md(/.png) true country vs predicted country (NO_DOC = no document)
- predictions_<config>.csv         every image's truth and prediction, for error analysis (labels and ids only)
- experiment.yaml                  copy of the config that produced these results

The selection rule is written in the YAML BEFORE running, so the winner is not picked after seeing the numbers.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from id_classifier.config import build_classifiers, build_detector, load_yaml
from id_classifier.pipeline import ImageOutcome, process_image
from id_classifier.routing import RoutingConfig
from id_classifier.sources import read_image_file
from id_classifier.types import GROUND_TRUTH_COLUMNS, NO_DOCUMENT, STATUS_MODEL_ERROR, STATUS_PARSE_ERROR, UNKNOWN, label_problems

log = logging.getLogger(__name__)

# Rate metrics, all between 0 and 1. The two groups say which direction is "better" (used by the selection rule).
HIGHER_IS_BETTER = (
    "type_accuracy", "country_accuracy", "country_accuracy_macro", "side_accuracy",
    "negative_rejection_rate", "coverage", "accepted_accuracy",
)
LOWER_IS_BETTER = ("unknown_rate", "review_rate", "parse_error_rate", "model_error_rate", "latency_mean_ms")
RATE_METRICS = HIGHER_IS_BETTER + ("unknown_rate", "review_rate", "parse_error_rate", "model_error_rate")  # averaged over repeats

METRIC_HELP = {
    "type_accuracy": "documents only: predicted document_type == true type",
    "country_accuracy": "documents only: predicted country == true country",
    "country_accuracy_macro": "country_accuracy computed per true country, then averaged (every country counts equally)",
    "side_accuracy": "documents only: predicted side == true side",
    "negative_rejection_rate": "no-document images predicted as 'none' (or no region detected)",
    "coverage": "documents auto-accepted, i.e. NOT sent to review",
    "accepted_accuracy": "auto-accepted documents with all three labels correct (the quality of what skips review)",
    "unknown_rate": "documents where the predicted country is 'unknown'",
    "review_rate": "all images flagged needs_review",
    "parse_error_rate": "classifier answers that could not be parsed (share of all classifier calls)",
    "model_error_rate": "classifier calls that failed completely (share of all classifier calls)",
    "latency": "time per image (detect + all classifiers + routing), mean ± std over all images and repeats",
}

NO_DOC_LABEL = "NO_DOC"  # confusion-matrix label for "no document"
ERROR_LABEL = "ERROR"    # confusion-matrix label when there is no prediction at all (failed fetch / parse error)


@dataclass
class GroundTruth:
    transaction_id: int
    image_id: str
    image_path: Path
    document_type: str
    issuing_country: str
    document_side: str

    @property
    def is_document(self) -> bool:
        return self.document_type != NO_DOCUMENT


@dataclass
class SelectionRule:
    """How the winning configuration is chosen. Fixed in the experiment YAML before running."""

    min_negative_rejection_rate: float = 0.95
    max_parse_error_rate: float = 0.02
    primary_metric: str = "country_accuracy_macro"  # highest wins...
    tie_breaker: str = "latency_mean_ms"            # ...on a tie, lowest wins

    @classmethod
    def from_dict(cls, values: dict | None) -> "SelectionRule":
        rule = cls(**(values or {}))
        if rule.primary_metric not in HIGHER_IS_BETTER:
            raise ValueError(f"primary_metric must be one of {HIGHER_IS_BETTER}")
        if rule.tie_breaker not in LOWER_IS_BETTER:
            raise ValueError(f"tie_breaker must be one of {LOWER_IS_BETTER}")
        return rule

    def describe(self) -> str:
        return (
            f"negative_rejection_rate >= {self.min_negative_rejection_rate} and "
            f"parse_error_rate <= {self.max_parse_error_rate}; then highest {self.primary_metric}; "
            f"tie -> lowest {self.tie_breaker}"
        )


@dataclass
class ExperimentResult:
    output_dir: Path
    winner: str | None
    summaries: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- ground truth

def read_ground_truth(csv_path: str | Path) -> list[GroundTruth]:
    """Reads and validates the ground-truth CSV. Any bad row stops the run with its row number."""
    csv_path = Path(csv_path)
    truths, seen = [], set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(GROUND_TRUTH_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):  # row 1 is the header
            problems = label_problems(row["document_type"], row["issuing_country"], row["document_side"])
            if problems:
                raise ValueError(f"{csv_path} row {row_number}: {'; '.join(problems)}")
            key = (int(row["transaction_id"]), row["image_id"])
            if key in seen:
                raise ValueError(f"{csv_path} row {row_number}: duplicate key {key}")
            seen.add(key)
            path = Path(row["image_path"])
            truths.append(GroundTruth(
                key[0], key[1], path if path.is_absolute() else csv_path.parent / path,
                row["document_type"], row["issuing_country"], row["document_side"],
            ))
    if not truths:
        raise ValueError(f"{csv_path}: no rows")
    return truths


# ---------------------------------------------------------------- metrics

def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None  # None = "not measurable" (e.g. no negatives in the set)


def _cm_true(truth: GroundTruth) -> str:
    return truth.issuing_country if truth.is_document else NO_DOC_LABEL


def _cm_pred(outcome: ImageOutcome) -> str:
    s = outcome.summary
    if s.document_type == NO_DOCUMENT:
        return NO_DOC_LABEL
    return s.issuing_country or ERROR_LABEL


def compute_metrics(truths: list[GroundTruth], outcomes: list[ImageOutcome]) -> dict:
    """Metrics for ONE repeat of ONE configuration."""
    pairs = list(zip(truths, outcomes))
    docs = [(t, o.summary) for t, o in pairs if t.is_document]
    negatives = [(t, o.summary) for t, o in pairs if not t.is_document]
    accepted = [(t, s) for t, s in docs if not s.needs_review]
    calls = [p for o in outcomes for preds in o.predictions.values() for p in preds]  # every classifier call

    per_country = {}
    by_country = defaultdict(list)
    for t, s in docs:
        by_country[t.issuing_country].append((t, s))
    for country, items in sorted(by_country.items()):
        per_country[country] = {
            "n": len(items),
            "type_accuracy": _rate([s.document_type == t.document_type for t, s in items]),
            "country_accuracy": _rate([s.issuing_country == t.issuing_country for t, s in items]),
            "side_accuracy": _rate([s.document_side == t.document_side for t, s in items]),
        }
    country_scores = [v["country_accuracy"] for v in per_country.values()]

    return {
        "type_accuracy": _rate([s.document_type == t.document_type for t, s in docs]),
        "country_accuracy": _rate([s.issuing_country == t.issuing_country for t, s in docs]),
        "country_accuracy_macro": sum(country_scores) / len(country_scores) if country_scores else None,
        "side_accuracy": _rate([s.document_side == t.document_side for t, s in docs]),
        "negative_rejection_rate": _rate([s.document_type == NO_DOCUMENT for t, s in negatives]),
        "coverage": _rate([not s.needs_review for t, s in docs]),
        "accepted_accuracy": _rate([
            (s.document_type, s.issuing_country, s.document_side) == (t.document_type, t.issuing_country, t.document_side)
            for t, s in accepted
        ]),
        "unknown_rate": _rate([s.issuing_country == UNKNOWN for t, s in docs]),
        "review_rate": _rate([o.summary.needs_review for o in outcomes]),
        "parse_error_rate": _rate([p.status == STATUS_PARSE_ERROR for p in calls]),
        "model_error_rate": _rate([p.status == STATUS_MODEL_ERROR for p in calls]),
        "per_country": per_country,
        "confusion": Counter((_cm_true(t), _cm_pred(o)) for t, o in pairs),
        "latencies": [o.latency_ms for o in outcomes],
    }


def _mean_ignoring_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(name: str, repeats: list[dict]) -> dict:
    """Averages the per-repeat metrics of one configuration."""
    summary = {"config": name, "repeats": len(repeats)}
    for metric in RATE_METRICS:
        summary[metric] = _mean_ignoring_none([r[metric] for r in repeats])
    latencies = [ms for r in repeats for ms in r["latencies"]]
    summary["latency_mean_ms"] = statistics.fmean(latencies) if latencies else None
    summary["latency_std_ms"] = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    per_country = {}
    for country in repeats[0]["per_country"]:
        rows = [r["per_country"][country] for r in repeats]
        per_country[country] = {"n": rows[0]["n"], **{
            k: _mean_ignoring_none([row[k] for row in rows]) for k in ("type_accuracy", "country_accuracy", "side_accuracy")
        }}
    summary["per_country"] = per_country
    summary["confusion"] = sum((r["confusion"] for r in repeats), Counter())  # summed over all repeats
    return summary


def select_winner(summaries: list[dict], rule: SelectionRule) -> tuple[str | None, dict[str, str]]:
    """Applies the rule. Returns (winner name or None, a note per configuration explaining pass/fail)."""
    notes, candidates = {}, []
    for s in summaries:
        problems = []
        rejection = s["negative_rejection_rate"]
        if rejection is not None and rejection < rule.min_negative_rejection_rate:  # None: no negatives to check
            problems.append(f"negative_rejection_rate {rejection:.3f} < {rule.min_negative_rejection_rate}")
        parse_errors = s["parse_error_rate"]
        if parse_errors is not None and parse_errors > rule.max_parse_error_rate:
            problems.append(f"parse_error_rate {parse_errors:.3f} > {rule.max_parse_error_rate}")
        notes[s["config"]] = "; ".join(problems) if problems else "passes"
        if not problems:
            candidates.append(s)
    if not candidates:
        return None, notes

    def sort_key(s: dict):
        primary = s[rule.primary_metric]
        tie = s[rule.tie_breaker]
        return (-(primary if primary is not None else -1.0), tie if tie is not None else float("inf"))

    return min(candidates, key=sort_key)["config"], notes


# ---------------------------------------------------------------- output files

def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows([[_fmt(v) for v in row] for row in rows])


def _confusion_labels(confusion: Counter) -> list[str]:
    labels = {label for pair in confusion for label in pair}
    special = [l for l in (UNKNOWN, NO_DOC_LABEL, ERROR_LABEL) if l in labels]
    return sorted(labels - set(special)) + special  # countries A-Z, then the special labels at the end


def write_confusion(out_dir: Path, name: str, confusion: Counter) -> None:
    labels = _confusion_labels(confusion)
    rows = [[true] + [confusion.get((true, pred), 0) for pred in labels] for true in labels]
    headers = ["true \\ predicted"] + labels
    _write_csv(out_dir / f"confusion_{_slug(name)}.csv", headers, rows)
    (out_dir / f"confusion_{_slug(name)}.md").write_text(
        f"# Country confusion matrix: {name}\n\nRows = true country, columns = predicted. "
        f"Summed over all repeats. {NO_DOC_LABEL} = no document, {ERROR_LABEL} = no prediction.\n\n"
        + _md_table(headers, rows) + "\n",
        encoding="utf-8",
    )
    try:
        import matplotlib
        matplotlib.use("Agg")                 # draw to a file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed: confusion matrix PNG skipped (CSV and Markdown written)")
        return
    matrix = [row[1:] for row in rows]
    size = max(4.0, 0.6 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name)
    peak = max((v for row in matrix for v in row), default=0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > peak / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{_slug(name)}.png", dpi=120)
    plt.close(fig)


def _new_run_dir(base: Path, experiment_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for attempt in range(100):
        path = base / f"{_slug(experiment_name)}_{stamp}" if attempt == 0 else base / f"{_slug(experiment_name)}_{stamp}_{attempt}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a new run folder in {base}")


COMPARISON_COLUMNS = [
    "type_accuracy", "country_accuracy", "country_accuracy_macro", "side_accuracy", "negative_rejection_rate",
    "coverage", "accepted_accuracy", "unknown_rate", "review_rate", "parse_error_rate", "model_error_rate",
]


def write_reports(out_dir: Path, name: str, summaries: list[dict], winner: str | None, notes: dict,
                  rule: SelectionRule, dataset_info: dict) -> None:
    # No no-document images in the ground truth -> that metric cannot be measured: leave it out of the report.
    # (It is still computed, and checked by the selection rule, as soon as such images are added.)
    columns = [m for m in COMPARISON_COLUMNS
               if m != "negative_rejection_rate" or dataset_info["negatives"] > 0]
    headers = ["config", *columns, "latency_ms (mean ± std)", "repeats", "selection"]
    rows = []
    for s in summaries:
        latency = "n/a" if s["latency_mean_ms"] is None else f"{s['latency_mean_ms']:.2f} ± {s['latency_std_ms']:.2f}"
        selection = "WINNER" if s["config"] == winner else notes[s["config"]]
        rows.append([s["config"], *[s[m] for m in columns], latency, s["repeats"], selection])
    _write_csv(out_dir / "comparison.csv", headers, rows)

    country_headers = ["config", "country", "n", "type_accuracy", "country_accuracy", "side_accuracy"]
    country_rows = [
        [s["config"], country, v["n"], v["type_accuracy"], v["country_accuracy"], v["side_accuracy"]]
        for s in summaries for country, v in s["per_country"].items()
    ]
    _write_csv(out_dir / "per_country.csv", country_headers, country_rows)

    parts = [
        f"# Experiment: {name}",
        "",
        f"- Ground truth: `{dataset_info['ground_truth']}`",
        f"- Images: {dataset_info['images']} ({dataset_info['documents']} documents"
        + (f", {dataset_info['negatives']} without a document" if dataset_info["negatives"] else "")
        + f", {dataset_info['fetch_failed']} could not be loaded)",
        f"- Repeats per configuration: {dataset_info['repeats']}",
        f"- Routing: {dataset_info['routing']}",
        f"- Selection rule (fixed before the run): {rule.describe()}",
        "",
        f"## Winner: **{winner}**" if winner else "## Winner: none. No configuration meets the selection rule.",
        "",
        "## Comparison",
        "",
        _md_table(headers, rows),
        "",
        "## Per-country accuracy",
        "",
        _md_table(country_headers, country_rows),
        "",
        "## Metric definitions",
        "",
        *[f"- **{k}**: {v}" for k, v in METRIC_HELP.items() if k in columns or k not in COMPARISON_COLUMNS],
        "",
        "Confusion matrices: `confusion_<config>.md` / `.csv` (and `.png` when matplotlib is installed).",
        "",
    ]
    (out_dir / "comparison.md").write_text("\n".join(parts), encoding="utf-8")


def write_predictions(out_dir: Path, name: str, rows: list[list]) -> None:
    headers = ["repeat", "transaction_id", "image_id", "true_type", "true_country", "true_side",
               "pred_type", "pred_country", "pred_side", "confidence", "needs_review", "reasons", "latency_ms",
               "classifier_outputs"]   # raw answer per classifier for the chosen region (labels/scores only)
    _write_csv(out_dir / f"predictions_{_slug(name)}.csv", headers, rows)


# ---------------------------------------------------------------- main entry point

def _evaluate_one(truth: GroundTruth, detector, classifiers, routing: RoutingConfig) -> ImageOutcome:
    """Loads one image, runs it, then drops the pixels: only labels and timings are kept, so memory stays
    flat however large the ground truth is (images are re-read for every configuration and repeat)."""
    record = read_image_file(truth.transaction_id, truth.image_id, truth.image_path)
    outcome = process_image(record, detector, classifiers, routing)
    record.image = None
    for region in outcome.regions:
        region.crop = None
    return outcome


def _progress_line(number: int, total: int, truth: GroundTruth, outcome: ImageOutcome) -> str:
    """One log line per image so a run can be followed live, e.g.
    'classifying 37/161: predicted JOR (0.91), true JOR [correct]'."""
    predicted, true = _cm_pred(outcome), _cm_true(truth)
    confidence = outcome.summary.confidence
    verdict = "correct" if predicted == true else "WRONG"
    if outcome.summary.needs_review:
        verdict += ", review"
    return (f"classifying {number}/{total}: predicted {predicted} ({_fmt(confidence)}), "
            f"true {true} [{verdict}]")


def run_experiment(config_path: str | Path) -> ExperimentResult:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    name = config.get("experiment_name", config_path.stem)
    repeats = int(config.get("repeats", 1))
    routing = RoutingConfig.from_dict(config.get("routing"))
    rule = SelectionRule.from_dict(config.get("selection_rule"))
    configurations = config.get("configurations") or []
    if not configurations:
        raise ValueError(f"{config_path}: no configurations to run")
    config_names = [c["name"] for c in configurations]
    if len(set(config_names)) != len(config_names):
        raise ValueError(f"{config_path}: configuration names must be unique, got {config_names}")

    truths = read_ground_truth(config["ground_truth"])
    documents = sum(t.is_document for t in truths)
    log.info("Ground truth: %d images (%d documents, %d negatives)", len(truths), documents, len(truths) - documents)

    out_dir = _new_run_dir(Path(config.get("output_dir", "data/runs")), name)
    shutil.copyfile(config_path, out_dir / "experiment.yaml")  # keep the exact config next to its results

    summaries, fetch_failed = [], 0
    for spec in configurations:
        detector = build_detector(spec.get("detector"))
        classifiers = build_classifiers(spec.get("classifiers", []))   # models are loaded once per configuration
        per_repeat, prediction_log = [], []
        for repeat in range(1, repeats + 1):
            outcomes = []
            for number, t in enumerate(truths, start=1):
                outcome = _evaluate_one(t, detector, classifiers, routing)
                outcomes.append(outcome)
                log.info("%s", _progress_line(number, len(truths), t, outcome))
            fetch_failed = sum(o.record.fetch_status != "ok" for o in outcomes)
            per_repeat.append(compute_metrics(truths, outcomes))
            for t, o in zip(truths, outcomes):
                s = o.summary
                prediction_log.append([
                    repeat, t.transaction_id, t.image_id, t.document_type, t.issuing_country, t.document_side,
                    s.document_type, s.issuing_country, s.document_side, s.confidence, s.needs_review,
                    ",".join(s.reasons), round(o.latency_ms, 2),
                    json.dumps({p.model_name: p.raw_output for p in o.predictions.get(s.region_id, [])}),
                ])
            log.info("Configuration %s: repeat %d/%d done", spec["name"], repeat, repeats)
        summary = aggregate(spec["name"], per_repeat)
        summaries.append(summary)
        write_confusion(out_dir, spec["name"], summary["confusion"])
        write_predictions(out_dir, spec["name"], prediction_log)

    winner, notes = select_winner(summaries, rule)
    dataset_info = {
        "ground_truth": config["ground_truth"], "images": len(truths), "documents": documents,
        "negatives": len(truths) - documents, "fetch_failed": fetch_failed, "repeats": repeats,
        "routing": routing.describe(),
    }
    write_reports(out_dir, name, summaries, winner, notes, rule, dataset_info)
    log.info("Results written to %s (winner: %s)", out_dir, winner)
    return ExperimentResult(out_dir, winner, summaries)
