"""Detectors: find the document(s) inside an image.

A detector returns zero or more Regions (bounding box + confidence + the cropped pixels).
- 0 regions  -> no document found (the image goes to review, never gets a forced label)
- 1 region   -> the normal case
- 2+ regions -> e.g. front and back photographed together (goes to review)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from id_classifier.types import ImageRecord, Region, make_region_id


class Detector(ABC):
    """Interface every detector implements."""

    name: str

    @abstractmethod
    def detect(self, record: ImageRecord) -> list[Region]:
        """Returns the document regions found in record.image ([] when none)."""


class FullImageDetector(Detector):
    """Baseline without a model: treats the whole image as one document region.

    Deciding "is there a document at all?" is then left to the classifier (document_type "none").
    """

    def __init__(self, name: str = "full_image"):
        self.name = name

    def detect(self, record: ImageRecord) -> list[Region]:
        if record.image is None:                 # the fetch failed: nothing to look at
            return []
        width, height = record.image.size
        return [
            Region(
                region_id=make_region_id(record.transaction_id, record.image_id, self.name, 0),
                transaction_id=record.transaction_id,
                image_id=record.image_id,
                bbox=(0, 0, width, height),
                confidence=1.0,
                detector_name=self.name,
                crop=record.image,               # no copy needed: nothing modifies the image
            )
        ]
