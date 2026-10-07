"""Pipeline for one image: detect -> classify -> route.

process_image() handles one image without touching any database; the evaluation harness calls it for every
ground-truth image.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from id_classifier.classifiers.base import Classifier
from id_classifier.detectors import Detector
from id_classifier.routing import RoutingConfig, route_image, summarize_image
from id_classifier.types import STATUS_MODEL_ERROR, ImageRecord, Prediction, Region, RoutedResult

log = logging.getLogger(__name__)


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
