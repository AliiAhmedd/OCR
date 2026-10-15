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
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from id_classifier.config import build_classifiers, build_detector, build_source, load_yaml
from id_classifier.pipeline import ImageOutcome, process_image
from id_classifier.routing import RoutingConfig
from id_classifier.sources import read_image_file
from id_classifier.types import (
    GROUND_TRUTH_COLUMNS,
    NO_DOCUMENT,
    OUTCOME_CLASSIFIED,
    OUTCOME_UNRECOGNISED,
    STATUS_MODEL_ERROR,
    STATUS_PARSE_ERROR,
    UNKNOWN,
    label_problems,
)

log = logging.getLogger(__name__)

# Rate metrics, all between 0 and 1. The two groups say which direction is "better" (used by the selection rule).
HIGHER_IS_BETTER = (
    "type_accuracy", "country_accuracy", "country_accuracy_macro", "side_accuracy",
    "negative_rejection_rate", "coverage", "accepted_accuracy",
)
LOWER_IS_BETTER = ("unknown_rate", "unrecognised_rate", "accepted_wrong_country_rate", "review_rate",
                   "parse_error_rate", "model_error_rate", "latency_mean_ms")
RATE_METRICS = HIGHER_IS_BETTER + tuple(m for m in LOWER_IS_BETTER if m != "latency_mean_ms")  # averaged over repeats
SMALL_CLASS_MAX = 10   # classes with fewer images than this get their own "indicative only" table in the report

METRIC_HELP = {
    "type_accuracy": "documents only: predicted document_type == true type",
    "country_accuracy": "documents only: predicted country == true country",
    "country_accuracy_macro": "country_accuracy computed per true country, then averaged (every country counts equally)",
    "side_accuracy": "documents only: predicted side == true side",
    "negative_rejection_rate": "no-document images predicted as 'none' (or no region detected)",
    "coverage": "documents auto-accepted, i.e. NOT sent to review",
    "accepted_accuracy": "auto-accepted documents with all three labels correct (the quality of what skips review)",
    "unknown_rate": "documents where the predicted country is 'unknown' (should be ~0: 'unknown' is reserved for no document)",
    "unrecognised_rate": "documents the classifier recognised as no known class (sent to review with a best guess)",
    "accepted_wrong_country_rate": "auto-accepted documents with the WRONG country (the costliest error: skips review)",
    "review_rate": "all images flagged needs_review",
    "parse_error_rate": "classifier answers that could not be parsed (share of all classifier calls)",
    "model_error_rate": "classifier calls that failed completely (share of all classifier calls)",
    "latency": "time per image (detect + all classifiers + routing), mean ± std over all images and repeats",
}

NO_DOC_LABEL = "NO_DOC"  # confusion-matrix label for "no document"
ERROR_LABEL = "ERROR"    # confusion-matrix label when there is no prediction at all (failed fetch / parse error)
UNRECOGNISED_LABEL = "UNRECOGNISED"  # confusion-matrix label for a document that matched no known class


@dataclass
class GroundTruth:
    transaction_id: int
    image_id: str
    image_path: Path
    document_type: str
    issuing_country: str
    document_side: str
    reference: str = ""        # image_path exactly as written in the CSV: what a `source` turns into a storage key

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
    min_coverage: float | None = None               # optional: needed when primary_metric is accepted_accuracy,
                                                    # which an over-cautious config could max out by reviewing everything

    @classmethod
    def from_dict(cls, values: dict | None) -> "SelectionRule":
        rule = cls(**(values or {}))
        if rule.primary_metric not in HIGHER_IS_BETTER:
            raise ValueError(f"primary_metric must be one of {HIGHER_IS_BETTER}")
        if rule.tie_breaker not in LOWER_IS_BETTER:
            raise ValueError(f"tie_breaker must be one of {LOWER_IS_BETTER}")
        return rule

    def describe(self) -> str:
        coverage = f"coverage >= {self.min_coverage} and " if self.min_coverage is not None else ""
        return (
            f"negative_rejection_rate >= {self.min_negative_rejection_rate} and "
            f"parse_error_rate <= {self.max_parse_error_rate} and {coverage}"
            f"then highest {self.primary_metric}; tie -> lowest {self.tie_breaker}"
        ).replace(" and then", "; then")


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
                row["document_type"], row["issuing_country"], row["document_side"], row["image_path"],
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
    if s.outcome == OUTCOME_UNRECOGNISED:
        return UNRECOGNISED_LABEL
    return s.issuing_country or ERROR_LABEL


def _class_of(document_type: str, country: str) -> str:
    return f"{document_type}/{country}"   # side is not part of the "dataset" a document goes into


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

    # Per document class (type + country): how each class fares, mainly to see the small classes.
    per_class = {}
    by_class = defaultdict(list)
    for t, s in docs:
        by_class[_class_of(t.document_type, t.issuing_country)].append((t, s))
    for name, items in sorted(by_class.items()):
        right = [s.outcome == OUTCOME_CLASSIFIED and (s.document_type, s.issuing_country) == (t.document_type, t.issuing_country)
                 for t, s in items]
        per_class[name] = {
            "n": len(items),
            "classified_correctly": sum(right),
            "unrecognised": sum(s.outcome == OUTCOME_UNRECOGNISED for t, s in items),
            "classified_as_other_class": sum(s.outcome == OUTCOME_CLASSIFIED and not r for (t, s), r in zip(items, right)),
            "auto_accepted": sum(not s.needs_review for t, s in items),
            "auto_accepted_correctly": sum(not s.needs_review and r for (t, s), r in zip(items, right)),
        }

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
        "unrecognised_rate": _rate([s.outcome == OUTCOME_UNRECOGNISED for t, s in docs]),
        "accepted_wrong_country_rate": _rate([s.issuing_country != t.issuing_country for t, s in accepted]),
        "review_rate": _rate([o.summary.needs_review for o in outcomes]),
        "parse_error_rate": _rate([p.status == STATUS_PARSE_ERROR for p in calls]),
        "model_error_rate": _rate([p.status == STATUS_MODEL_ERROR for p in calls]),
        "per_country": per_country,
        "per_class": per_class,
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
    summary["per_class"] = {
        name: {k: _mean_ignoring_none([r["per_class"][name][k] for r in repeats]) for k in values}
        for name, values in repeats[0]["per_class"].items()
    }
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
        coverage = s["coverage"]
        if rule.min_coverage is not None and (coverage is None or coverage < rule.min_coverage):
            problems.append(f"coverage {_fmt(coverage)} < {rule.min_coverage}")
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
    special = [l for l in (UNKNOWN, UNRECOGNISED_LABEL, NO_DOC_LABEL, ERROR_LABEL) if l in labels]
    return sorted(labels - set(special)) + special  # countries A-Z, then the special labels at the end


def write_confusion(out_dir: Path, name: str, confusion: Counter) -> None:
    labels = _confusion_labels(confusion)
    rows = [[true] + [confusion.get((true, pred), 0) for pred in labels] for true in labels]
    headers = ["true \\ predicted"] + labels
    _write_csv(out_dir / f"confusion_{_slug(name)}.csv", headers, rows)
    (out_dir / f"confusion_{_slug(name)}.md").write_text(
        f"# Country confusion matrix: {name}\n\nRows = true country, columns = predicted. "
        f"Summed over all repeats. {NO_DOC_LABEL} = no document, {UNRECOGNISED_LABEL} = a document matching no known "
        f"class (sent to review), {ERROR_LABEL} = no prediction.\n\n"
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
    "coverage", "accepted_accuracy", "accepted_wrong_country_rate", "unknown_rate", "unrecognised_rate", "review_rate",
    "parse_error_rate", "model_error_rate",
]
CLASS_COLUMNS = ["n", "classified_correctly", "unrecognised", "classified_as_other_class", "auto_accepted",
                 "auto_accepted_correctly"]


BEYOND_HEADERS = ["config", "documents", "auto_accepted", "accepted_correct", "accepted_wrong_country", "coverage",
                  "accepted_accuracy"]


def _accepted(row: dict) -> bool:
    return str(row["needs_review"]).lower() not in ("true", "1")


def _all_correct(row: dict) -> bool:
    return (row["pred_type"], row["pred_country"], row["pred_side"]) == \
           (row["true_type"], row["true_country"], row["true_side"])


def beyond_majority(predictions: dict[str, list[dict]]) -> tuple[str, list[str], list[list]]:
    """What is auto-accepted OUTSIDE the biggest country. With 150 Tunisian cards against 1-7 of every other
    country, coverage and accepted_accuracy over all documents mostly measure Tunisia (progress.md row 26).

    predictions: config -> its per-image rows (the columns of predictions_<config>.csv, one repeat or more).
    Returns the majority country, the markdown lines of the report section and the CSV rows (BEYOND_HEADERS)."""
    rows = {c: [r for r in rs if r["true_type"] != NO_DOCUMENT] for c, rs in predictions.items()}
    first = next(iter(rows.values()), [])
    sizes = Counter(r["true_country"] for r in first if str(r.get("repeat", 1)) == "1")
    if not sizes:
        return "", [], []
    majority = sizes.most_common(1)[0][0]
    countries = sorted(c for c in sizes if c != majority)

    csv_rows, country_cells = [], []
    for config, docs in rows.items():
        others = [r for r in docs if r["true_country"] != majority]
        accepted = [r for r in others if _accepted(r)]
        correct = sum(map(_all_correct, accepted))
        wrong_country = sum(r["pred_country"] != r["true_country"] for r in accepted)
        csv_rows.append([config, len(others), len(accepted), correct, wrong_country,
                         _rate([_accepted(r) for r in others]), _rate([_all_correct(r) for r in accepted])])
        cells = []
        for country in countries:
            items = [r for r in docs if r["true_country"] == country]
            took = [r for r in items if _accepted(r)]
            cells.append(f"{sum(map(_all_correct, took))}/{len(took)}/{len(items)}")
        country_cells.append([config, *cells])

    lines = [
        f"## Beyond {majority} (the biggest country, {sizes[majority]} of {sum(sizes.values())} documents)",
        "",
        f"coverage and accepted_accuracy above are dominated by {majority}. The same numbers over the other "
        f"countries only: what does the classifier dare to auto-accept outside {majority}, and is it right?",
        "",
        _md_table(BEYOND_HEADERS, csv_rows),
        "",
        "Per country: **correct / auto-accepted / documents** (correct = type, country and side all right).",
        "",
        _md_table(["config", *countries], country_cells),
        "",
    ]
    return majority, lines, csv_rows


def write_beyond_majority(out_dir: Path, predictions: dict[str, list[dict]]) -> list[str]:
    """Writes beyond_majority.csv and returns the markdown section for comparison.md."""
    _, lines, csv_rows = beyond_majority(predictions)
    if lines:
        _write_csv(out_dir / "beyond_majority.csv", BEYOND_HEADERS, csv_rows)
    return lines


def backfill_beyond_majority(run_dir: str | Path) -> Path:
    """Adds the "Beyond <biggest country>" section to the comparison.md of a run made before it existed, from
    the run's own predictions_<config>.csv files: nothing is re-run. Replaces the section if it is already there."""
    run_dir = Path(run_dir)
    with (run_dir / "comparison.csv").open(newline="", encoding="utf-8") as f:
        configs = [row["config"] for row in csv.DictReader(f)]
    predictions = {}
    for config in configs:
        with (run_dir / f"predictions_{_slug(config)}.csv").open(newline="", encoding="utf-8") as f:
            predictions[config] = list(csv.DictReader(f))
    section = write_beyond_majority(run_dir, predictions)
    report = run_dir / "comparison.md"
    text = report.read_text(encoding="utf-8")
    text = re.sub(r"## Beyond .*?(?=## Per-country accuracy)", "", text, flags=re.S)   # an older copy
    marker = "## Per-country accuracy"
    if marker not in text:
        raise ValueError(f"{report}: no '{marker}' heading to put the section before")
    report.write_text(text.replace(marker, "\n".join(section) + "\n" + marker, 1), encoding="utf-8")
    return report


def write_reports(out_dir: Path, name: str, summaries: list[dict], winner: str | None, notes: dict,
                  rule: SelectionRule, dataset_info: dict, small_class_max: int = SMALL_CLASS_MAX,
                  status: str | None = None, predictions: dict[str, list[dict]] | None = None) -> None:
    """status: set while the run is still going (the reports are rewritten after every configuration, so they
    can be read live); the winner is then only the best of the configurations finished so far.
    predictions: config -> per-image rows, for the "Beyond <biggest country>" section."""
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

    class_headers = ["config", "class", *CLASS_COLUMNS]
    class_rows = [[s["config"], cls, *[v[c] for c in CLASS_COLUMNS]] for s in summaries for cls, v in s["per_class"].items()]
    _write_csv(out_dir / "per_class.csv", class_headers, class_rows)
    small_rows = [row for row in class_rows if row[2] < small_class_max]

    parts = [
        f"# Experiment: {name}",
        "",
        f"- Ground truth: `{dataset_info['ground_truth']}`",
        f"- Images: {dataset_info['images']} ({dataset_info['documents']} documents"
        + (f", {dataset_info['negatives']} without a document" if dataset_info["negatives"] else "")
        + f", {dataset_info['fetch_failed']} could not be loaded)",
        f"- Repeats per configuration: {dataset_info['repeats']}",
        f"- Routing: {dataset_info['routing']}",
        f"- Images read from: {dataset_info.get('source', 'local files')}",
        *([f"- Evaluation rule `review_single_sample_classes`: the {dataset_info['single_sample_documents']} documents "
           "whose class (type/country/side) has ONE image go to review (`single_sample_class`): with no reference of "
           "their own class they can only match another class. Uses the ground truth: not a production rule."]
          if dataset_info.get("single_sample_documents") is not None else []),
        f"- Selection rule (fixed before the run): {rule.describe()}",
        "",
        *([f"## Status: RUNNING. {status}", ""] if status else []),
        (f"## {'Leading so far (provisional)' if status else 'Winner'}: **{winner}**" if winner
         else "## Winner: none. No configuration meets the selection rule" + (" yet." if status else ".")),
        "",
        "## Comparison",
        "",
        _md_table(headers, rows),
        "",
        *(write_beyond_majority(out_dir, predictions) if predictions else []),
        "## Per-country accuracy",
        "",
        _md_table(country_headers, country_rows),
        "",
        f"## Small classes (fewer than {small_class_max} images): indicative only",
        "",
        "Too few images for a reliable rate, so counts are shown instead. Under leave-one-out a class with n images "
        "has only n-1 references (front and back count separately): a class with 1 image can never be recognised.",
        "",
        _md_table(class_headers, small_rows) if small_rows else "_No small classes._",
        "",
        "All classes: `per_class.csv`.",
        "",
        "## Metric definitions",
        "",
        *[f"- **{k}**: {v}" for k, v in METRIC_HELP.items() if k in columns or k not in COMPARISON_COLUMNS],
        "",
        "Confusion matrices: `confusion_<config>.md` / `.csv` (and `.png` when matplotlib is installed).",
        "",
    ]
    (out_dir / "comparison.md").write_text("\n".join(parts), encoding="utf-8")


PREDICTION_HEADERS = ["repeat", "transaction_id", "image_id", "true_type", "true_country", "true_side",
                      "pred_type", "pred_country", "pred_side", "outcome", "confidence", "needs_review", "reasons",
                      "latency_ms",
                      "classifier_outputs"]   # raw answer per classifier for the chosen region (labels/scores only)


def write_predictions(out_dir: Path, name: str, rows: list[list]) -> None:
    _write_csv(out_dir / f"predictions_{_slug(name)}.csv", PREDICTION_HEADERS, rows)


# ---------------------------------------------------------------- main entry point

SINGLE_SAMPLE_CLASS = "single_sample_class"   # review reason added by the evaluation rule below


def class_sizes(truths) -> Counter:
    """Images per document class (type, country, side) in the ground truth."""
    return Counter((t.document_type, t.issuing_country, t.document_side) for t in truths if t.is_document)


def review_single_sample(truth, summary, sizes: Counter | None) -> None:
    """EVALUATION RULE (YAML `review_single_sample_classes: true`), uses the ground truth: a document whose class
    has only ONE image in the ground truth has no reference of its own class under leave-one-out, so the kNN can
    only match it to another class (passport/NG/front, passport/SY/front, ...). It goes to review, with reason
    "single_sample_class", instead of counting as a wrong guess. A live pipeline cannot do this (it has no ground
    truth); there such a card must be caught by the similarity floor ("unrecognised")."""
    if sizes is None or not truth.is_document:
        return
    if sizes[(truth.document_type, truth.issuing_country, truth.document_side)] < 2:
        summary.needs_review = True
        if SINGLE_SAMPLE_CLASS not in summary.reasons:
            summary.reasons.append(SINGLE_SAMPLE_CLASS)


def _evaluate_one(truth: GroundTruth, detector, classifiers, routing: RoutingConfig, source=None,
                  sizes: Counter | None = None) -> ImageOutcome:
    """Loads one image, runs it, then drops the pixels: only labels and timings are kept, so memory stays
    flat however large the ground truth is (images are re-read for every configuration and repeat).
    Without a source the image is read from disk; with one (YAML `source:`), from that storage.
    sizes: class sizes when the single-sample evaluation rule is on (review_single_sample)."""
    if source is None:
        record = read_image_file(truth.transaction_id, truth.image_id, truth.image_path)
    else:
        record = source.fetch(truth.transaction_id, truth.image_id, truth.reference)
    outcome = process_image(record, detector, classifiers, routing)
    review_single_sample(truth, outcome.summary, sizes)
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

    source = build_source(config.get("source"))   # None = read the ground-truth files from disk
    source_name = getattr(source, "name", None) or (config.get("source") or {}).get("type") or "local files"
    log.info("Images read from: %s", source_name)

    out_dir = _new_run_dir(Path(config.get("output_dir", "data/runs")), name)
    shutil.copyfile(config_path, out_dir / "experiment.yaml")  # keep the exact config next to its results

    sizes = class_sizes(truths) if config.get("review_single_sample_classes") else None
    single = sum(n for n in sizes.values() if n < 2) if sizes else 0

    def info() -> dict:
        return {
            "ground_truth": config["ground_truth"], "images": len(truths), "documents": documents,
            "negatives": len(truths) - documents, "fetch_failed": fetch_failed, "repeats": repeats,
            "routing": routing.describe(), "source": source_name,
            "single_sample_documents": single if sizes else None,
        }

    small_class_max = int(config.get("small_class_max", SMALL_CLASS_MAX))
    summaries, fetch_failed, predictions = [], 0, {}
    started = time.perf_counter()
    for spec in configurations:
        detector = build_detector(spec.get("detector"))
        # models are loaded once per configuration; with a `source:`, kNN references are fetched from it too
        classifiers = build_classifiers(spec.get("classifiers", []), source)
        per_repeat, prediction_log = [], []
        for repeat in range(1, repeats + 1):
            outcomes = []
            for number, t in enumerate(truths, start=1):
                outcome = _evaluate_one(t, detector, classifiers, routing, source, sizes)
                outcomes.append(outcome)
                log.info("%s", _progress_line(number, len(truths), t, outcome))
            fetch_failed = sum(o.record.fetch_status != "ok" for o in outcomes)
            per_repeat.append(compute_metrics(truths, outcomes))
            for t, o in zip(truths, outcomes):
                s = o.summary
                prediction_log.append([
                    repeat, t.transaction_id, t.image_id, t.document_type, t.issuing_country, t.document_side,
                    s.document_type, s.issuing_country, s.document_side, s.outcome, s.confidence, s.needs_review,
                    ",".join(s.reasons), round(o.latency_ms, 2),
                    json.dumps({p.model_name: p.raw_output for p in o.predictions.get(s.region_id, [])}),
                ])
            log.info("Configuration %s: repeat %d/%d done", spec["name"], repeat, repeats)
        summary = aggregate(spec["name"], per_repeat)
        summaries.append(summary)
        write_confusion(out_dir, spec["name"], summary["confusion"])
        write_predictions(out_dir, spec["name"], prediction_log)
        predictions[spec["name"]] = [dict(zip(PREDICTION_HEADERS, row)) for row in prediction_log]

        done, left = len(summaries), len(configurations) - len(summaries)
        if left:                                  # live reports: readable after every finished configuration
            minutes = (time.perf_counter() - started) / 60
            status = (f"{done} of {len(configurations)} configurations done, {left} left "
                      f"(~{minutes / done * left:.0f} min to go at {minutes / done:.1f} min per configuration).")
            winner, notes = select_winner(summaries, rule)
            write_reports(out_dir, name, summaries, winner, notes, rule, info(), small_class_max, status, predictions)
            log.info("Configuration %s done: %s Live results: %s", spec["name"], status, out_dir / "comparison.md")

    winner, notes = select_winner(summaries, rule)
    write_reports(out_dir, name, summaries, winner, notes, rule, info(), small_class_max, predictions=predictions)
    log.info("Results written to %s (winner: %s)", out_dir, winner)
    return ExperimentResult(out_dir, winner, summaries)
