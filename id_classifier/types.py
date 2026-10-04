"""Shared labels and the small records that move between the pipeline steps.

Every step (source -> detector -> classifier -> routing -> storage) passes these records along,
so this file is the one place that defines what a "label" is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image

# --- Label taxonomy. Changing it means changing these tuples (and the VLM prompt later). ---
DOCUMENT_TYPES = ("national_id", "passport", "residence_permit", "driving_license", "other", "none")
DOCUMENT_SIDES = ("front", "back", "n/a", "unknown")  # "n/a" for single-page documents (passport) and for "none"
UNKNOWN = "unknown"    # country (or side) the classifier could not decide
NO_DOCUMENT = "none"   # document_type when there is no document in the image at all

_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")  # shape of an ISO 3166-1 alpha-2 code, e.g. TR, TN

# --- Prediction status: an error is recorded as an error, never as a label. ---
STATUS_OK = "ok"                    # the classifier returned a valid answer
STATUS_PARSE_ERROR = "parse_error"  # the model answered, but not in the expected JSON / label set
STATUS_MODEL_ERROR = "model_error"  # the model could not be called at all (crash, timeout, not installed)

# Columns of a ground-truth CSV (used by the synthetic generator and the evaluation harness).
GROUND_TRUTH_COLUMNS = ("transaction_id", "image_id", "image_path", "document_type", "issuing_country", "document_side")


def is_valid_country(code: str | None) -> bool:
    """True for 'unknown' or a two-letter upper-case code. Only the shape is checked, not the ISO list."""
    return code == UNKNOWN or bool(_COUNTRY_CODE.match(code or ""))


def label_problems(document_type: str | None, issuing_country: str | None, document_side: str | None) -> list[str]:
    """Returns what is wrong with a label triple (empty list = valid)."""
    problems = []
    if document_type not in DOCUMENT_TYPES:
        problems.append(f"document_type {document_type!r} not in {DOCUMENT_TYPES}")
    if not is_valid_country(issuing_country):
        problems.append(f"issuing_country {issuing_country!r} is not a 2-letter code or 'unknown'")
    if document_side not in DOCUMENT_SIDES:
        problems.append(f"document_side {document_side!r} not in {DOCUMENT_SIDES}")
    return problems


def make_region_id(transaction_id: int, image_id: str, detector_name: str, index: int | str) -> str:
    """Region ids look like '900000001:front:full_image:0'. The detector name is part of the id,
    so regions from two different detectors never collide in the database."""
    return f"{transaction_id}:{image_id}:{detector_name}:{index}"


@dataclass
class ImageRecord:
    """One image of one transaction. Key = (transaction_id, image_id)."""

    transaction_id: int
    image_id: str                        # e.g. "front", "back", "page"
    image_reference: str                 # file path or blob key; safe to log (no personal data)
    image: Image.Image | None = None     # decoded image; None when the fetch failed
    image_hash: str | None = None        # sha256 of the original bytes, to spot duplicates
    fetch_status: str = "ok"             # "ok" | "failed"
    failure_reason: str | None = None    # short machine-readable reason, e.g. "decode_error: OSError"

    @property
    def key(self) -> str:
        return f"{self.transaction_id}:{self.image_id}"


@dataclass
class Region:
    """One document found inside an image by a detector."""

    region_id: str
    transaction_id: int                  # link back to the parent image
    image_id: str
    bbox: tuple[int, int, int, int]      # (x0, y0, x1, y1) in pixels of the parent image
    confidence: float                    # detector confidence, 1.0 for FullImageDetector
    detector_name: str
    crop: Image.Image | None = None      # the cut-out region that goes to the classifier (not stored in the DB)


@dataclass
class Prediction:
    """What one classifier said about one region."""

    document_type: str | None            # None only when status != "ok"
    issuing_country: str | None
    document_side: str | None
    confidence: float | None             # 0..1, or None when the model gives no confidence
    status: str                          # STATUS_OK / STATUS_PARSE_ERROR / STATUS_MODEL_ERROR
    raw_output: str | None               # exactly what the model returned (stored in the DB, never logged)
    model_name: str
    model_version: str
    latency_ms: float
    error: str | None = None             # short error description when status != "ok"


@dataclass
class RoutedResult:
    """Final decision for one region (or for an image with no region): labels + whether a human must look."""

    transaction_id: int
    image_id: str
    region_id: str
    document_type: str | None
    issuing_country: str | None
    document_side: str | None
    confidence: float | None
    needs_review: bool
    reasons: list[str] = field(default_factory=list)  # why it needs review, e.g. ["low_confidence"]
