"""Routing: combine the classifiers' predictions and decide "auto-accept" or "needs review".

Principle: never force a label. When in doubt the result keeps its best-guess labels (or None) and gets
needs_review=True plus the list of reasons, so a reviewer can see WHY it was flagged.

Every result also gets exactly one outcome:
- classified:   a document of a known class (auto-accepted, or reviewed when confidence is low etc.)
- unrecognised: a document, but it matches no known class well enough. Keeps the best guess, always reviewed.
- no_document:  a blank photo / nothing that looks like a document. Labels "none" / "unknown" / "n/a".
                Also: an image where the detector found NO card (whole-image fallback region) and the
                classifier recognised nothing: there is no evidence of a document at all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from id_classifier.detectors import FULL_IMAGE_INDEX
from id_classifier.types import (
    NO_DOCUMENT,
    OUTCOME_CLASSIFIED,
    OUTCOME_FETCH_FAILED,
    OUTCOME_NO_DOCUMENT,
    OUTCOME_UNRECOGNISED,
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
UNRECOGNISED = "unrecognised"              # a document, but it matches no known class well enough
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

    outcome = _vote([p.outcome for p in valid]) or OUTCOME_CLASSIFIED
    if outcome == OUTCOME_UNRECOGNISED and is_whole_image_fallback(region):
        outcome = OUTCOME_NO_DOCUMENT                         # no card found AND nothing recognised: no document
    if labels["document_type"] == NO_DOCUMENT or outcome == OUTCOME_NO_DOCUMENT:
        outcome = OUTCOME_NO_DOCUMENT
        labels = {"document_type": NO_DOCUMENT, "issuing_country": UNKNOWN, "document_side": "n/a"}
        reasons.append(NO_DOCUMENT_FOUND)                     # "none" is never auto-accepted: a human confirms it
    elif outcome == OUTCOME_UNRECOGNISED:
        reasons.append(UNRECOGNISED)                          # best guess kept, a human decides
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
        outcome=outcome,
        **labels,
    )


def is_whole_image_fallback(region: Region) -> bool:
    """True for YoloDetector's "no box found" region: the whole photo, sent on only as a fallback."""
    return region.region_id.endswith(f":{FULL_IMAGE_INDEX}")


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
        return [RoutedResult(record.transaction_id, record.image_id, region_id, None, None, None, None, True, [FETCH_FAILED],
                             OUTCOME_FETCH_FAILED)]
    if not regions:
        region_id = make_region_id(record.transaction_id, record.image_id, detector_name, "none")
        return [RoutedResult(record.transaction_id, record.image_id, region_id, NO_DOCUMENT, UNKNOWN, "n/a", None, True,
                             [NO_DOCUMENT_FOUND], OUTCOME_NO_DOCUMENT)]
    multiple = len(regions) > 1
    return [combine(r, predictions.get(r.region_id, []), config, multiple_documents=multiple) for r in regions]


def summarize_image(routed: list[RoutedResult]) -> RoutedResult:
    """One result per image (used for evaluation and per-country counts).
    With several regions the most confident one is shown, and it is already marked multiple_documents."""
    if len(routed) == 1:
        return routed[0]
    return max(routed, key=lambda r: r.confidence if r.confidence is not None else -1.0)
