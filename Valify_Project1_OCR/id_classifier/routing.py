"""Routing: combine the classifiers' predictions and decide "auto-accept" or "needs review".

Principle: never force a label. When in doubt the result keeps its best-guess labels (or None) and gets
needs_review=True plus the list of reasons, so a reviewer can see WHY it was flagged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from id_classifier.types import (
    NO_DOCUMENT,
    STATUS_OK,
    UNKNOWN,
    ImageRecord,
    Prediction,
    Region,
    RoutedResult,
    make_region_id,
)

# Review reasons (stored in the database, so keep the names stable)
LOW_CONFIDENCE = "low_confidence"          # combined confidence below the threshold (or missing)
UNKNOWN_COUNTRY = "unknown_country"        # a document, but the country could not be decided
DISAGREEMENT = "disagreement"              # classifiers gave different labels
MULTIPLE_DOCUMENTS = "multiple_documents"  # the detector found 2+ documents in one image
NO_DOCUMENT_FOUND = "no_document"          # detector found nothing, or the classifier says "none"
FETCH_FAILED = "fetch_failed"              # the image could not be loaded
# plus the prediction statuses themselves ("parse_error", "model_error") when a classifier failed

LABEL_FIELDS = ("document_type", "issuing_country", "document_side")


@dataclass
class RoutingConfig:
    confidence_threshold: float = 0.8  # below this the result goes to review

    @classmethod
    def from_dict(cls, values: dict | None) -> "RoutingConfig":
        return cls(**(values or {}))   # an unknown key (typo in YAML) raises a TypeError, on purpose

    def describe(self) -> str:
        return f"threshold={self.confidence_threshold}"  # stored as model_version of the router


def _vote(values: list[str | None]) -> str | None:
    """Majority vote for one field. A tie returns None: we do not pick a winner at random."""
    ranked = Counter(v for v in values if v is not None).most_common()
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def combine(region: Region, predictions: list[Prediction], config: RoutingConfig, multiple_documents: bool = False) -> RoutedResult:
    """One region + all classifiers' predictions for it -> one RoutedResult."""
    reasons = []
    valid = [p for p in predictions if p.status == STATUS_OK]
    for p in predictions:
        if p.status != STATUS_OK and p.status not in reasons:
            reasons.append(p.status)                          # e.g. "parse_error": a classifier failed

    labels = {f: _vote([getattr(p, f) for p in valid]) for f in LABEL_FIELDS}
    if any(len({getattr(p, f) for p in valid}) > 1 for f in LABEL_FIELDS):
        reasons.append(DISAGREEMENT)

    confidences = [p.confidence for p in valid]
    confidence = sum(confidences) / len(confidences) if confidences and None not in confidences else None
    if confidence is None or confidence < config.confidence_threshold:
        reasons.append(LOW_CONFIDENCE)

    if labels["document_type"] == NO_DOCUMENT:
        reasons.append(NO_DOCUMENT_FOUND)                     # "none" is never auto-accepted: a human confirms it
    elif valid and labels["issuing_country"] in (None, UNKNOWN):   # (no valid prediction: already flagged above)
        reasons.append(UNKNOWN_COUNTRY)
    if multiple_documents:
        reasons.append(MULTIPLE_DOCUMENTS)

    return RoutedResult(
        transaction_id=region.transaction_id,
        image_id=region.image_id,
        region_id=region.region_id,
        confidence=confidence,
        needs_review=bool(reasons),
        reasons=reasons,
        **labels,
    )


def route_image(
    record: ImageRecord,
    regions: list[Region],
    predictions: dict[str, list[Prediction]],
    config: RoutingConfig,
    detector_name: str,
) -> list[RoutedResult]:
    """All routed results for one image: one per region, or a single one when there is no region."""
    if record.fetch_status != "ok":
        region_id = make_region_id(record.transaction_id, record.image_id, detector_name, FETCH_FAILED)
        return [RoutedResult(record.transaction_id, record.image_id, region_id, None, None, None, None, True, [FETCH_FAILED])]
    if not regions:
        region_id = make_region_id(record.transaction_id, record.image_id, detector_name, "none")
        return [RoutedResult(record.transaction_id, record.image_id, region_id, NO_DOCUMENT, UNKNOWN, "n/a", None, True, [NO_DOCUMENT_FOUND])]
    multiple = len(regions) > 1
    return [combine(r, predictions.get(r.region_id, []), config, multiple_documents=multiple) for r in regions]


def summarize_image(routed: list[RoutedResult]) -> RoutedResult:
    """One result per image (used for evaluation and per-country counts).
    With several regions the most confident one is shown, and it is already marked multiple_documents."""
    if len(routed) == 1:
        return routed[0]
    return max(routed, key=lambda r: r.confidence if r.confidence is not None else -1.0)
