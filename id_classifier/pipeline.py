"""Pipeline: fetch -> detect -> classify -> route (-> store).

process_image() handles one image without touching the database; the evaluation harness uses it directly.
run_classification() loops over a source and stores everything; the CLI `classify` command and
(later) the Airflow DAG use it.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from id_classifier import __version__
from id_classifier.classifiers.base import Classifier
from id_classifier.detectors import Detector
from id_classifier.routing import RoutingConfig, route_image, summarize_image
from id_classifier.sources import ImageKey, ImageSource
from id_classifier.storage import Storage
from id_classifier.types import (
    NO_DOCUMENT,
    STATUS_MODEL_ERROR,
    STATUS_OK,
    ImageRecord,
    Prediction,
    Region,
    RoutedResult,
)

log = logging.getLogger(__name__)

ROUTER_MODEL_NAME = "router"  # model_name of the routing-decision rows in the predictions table


@dataclass
class ImageOutcome:
    record: ImageRecord
    regions: list[Region]
    predictions: dict[str, list[Prediction]]  # region_id -> one prediction per classifier
    routed: list[RoutedResult]                # one per region (or one "no document" / "fetch failed" result)
    summary: RoutedResult                     # one result for the whole image
    latency_ms: float                         # detect + classify + route for this image


def _safe_classify(classifier: Classifier, region: Region) -> Prediction:
    """Classifiers should not raise, but if one does, record it as a model error instead of stopping the run."""
    start = time.perf_counter()
    try:
        return classifier.classify(region.crop)
    except Exception as exc:
        log.warning("Classifier %s failed on region %s: %s", classifier.name, region.region_id, type(exc).__name__)
        return Prediction(
            None, None, None, None, STATUS_MODEL_ERROR, None, classifier.name,
            getattr(classifier, "model_version", "unknown"), (time.perf_counter() - start) * 1000,
            error=type(exc).__name__,
        )


def process_image(record: ImageRecord, detector: Detector, classifiers: list[Classifier], routing: RoutingConfig) -> ImageOutcome:
    start = time.perf_counter()
    regions = detector.detect(record) if record.fetch_status == "ok" else []
    predictions = {r.region_id: [_safe_classify(c, r) for c in classifiers] for r in regions}
    routed = route_image(record, regions, predictions, routing, detector.name)
    latency_ms = (time.perf_counter() - start) * 1000
    return ImageOutcome(record, regions, predictions, routed, summarize_image(routed), latency_ms)


def prediction_rows(outcome: ImageOutcome, run_id: str, routing: RoutingConfig) -> list[dict]:
    """Rows for the predictions table: every classifier answer + the router's decision per region."""
    rows = []
    for region_id, region_predictions in outcome.predictions.items():
        for p in region_predictions:
            rows.append({
                "run_id": run_id, "pipeline_version": __version__,
                "transaction_id": outcome.record.transaction_id, "image_id": outcome.record.image_id, "region_id": region_id,
                "model_name": p.model_name, "model_version": p.model_version,
                "document_type": p.document_type, "issuing_country": p.issuing_country, "document_side": p.document_side,
                "confidence": p.confidence, "status": p.status, "raw_output": p.raw_output, "error": p.error,
                "latency_ms": p.latency_ms,
            })
    for r in outcome.routed:
        rows.append({
            "run_id": run_id, "pipeline_version": __version__,
            "transaction_id": r.transaction_id, "image_id": r.image_id, "region_id": r.region_id,
            "model_name": ROUTER_MODEL_NAME, "model_version": routing.describe(),
            "document_type": r.document_type, "issuing_country": r.issuing_country, "document_side": r.document_side,
            "confidence": r.confidence, "status": STATUS_OK,
            "needs_review": r.needs_review, "review_reasons": ",".join(r.reasons),
        })
    return rows


def count_label(summary: RoutedResult, record: ImageRecord) -> tuple[str, str]:
    """(country bucket, routing status) for the per-run counts in the log. IDs and counts only, no content."""
    if record.fetch_status != "ok":
        return "FETCH_FAILED", "fetch_failed"
    country = "NO_DOC" if summary.document_type == NO_DOCUMENT else (summary.issuing_country or "ERROR")
    return country, "needs_review" if summary.needs_review else "auto_accepted"


def new_run_id(prefix: str = "cli") -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def run_classification(
    source: ImageSource,
    detector: Detector,
    classifiers: list[Classifier],
    routing: RoutingConfig,
    storage: Storage,
    run_id: str,
    keys: list[ImageKey] | None = None,
) -> dict:
    """Processes every key (default: everything the source lists) and stores the results. Returns the counts."""
    storage.create_tables()
    keys = source.list_keys() if keys is None else keys
    per_country, per_status = Counter(), Counter()
    for transaction_id, image_id, reference in keys:
        record = source.fetch(transaction_id, image_id, reference)
        storage.save_image(record)
        outcome = process_image(record, detector, classifiers, routing)
        if record.fetch_status == "ok":  # a failed fetch is recorded in `images` only; there is nothing to predict
            storage.save_regions(outcome.regions)
            storage.save_predictions(prediction_rows(outcome, run_id, routing))
            storage.ensure_reviews(outcome.routed)
        country, status = count_label(outcome.summary, record)
        per_country[country] += 1
        per_status[status] += 1
    log.info("Run %s: %d images processed", run_id, len(keys))
    log.info("Per country: %s", dict(sorted(per_country.items())))
    log.info("Per routing status: %s", dict(sorted(per_status.items())))
    return {"run_id": run_id, "images": len(keys), "per_country": dict(per_country), "per_status": dict(per_status)}
