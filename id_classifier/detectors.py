"""Detectors: find the document(s) inside an image.

A detector returns zero or more Regions (bounding box + confidence + the cropped pixels).
- 0 regions  -> no document found (the image goes to review, never gets a forced label)
- 1 region   -> the normal case
- 2+ regions -> e.g. front and back photographed together (goes to review)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from id_classifier.types import ImageRecord, Region, make_region_id

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels, x1/y1 exclusive (same as PIL's crop)


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


class YoloDetector(Detector):
    """Finds documents with an Ultralytics YOLO model and crops them out of the image.

    Two kinds of weights:
    - open vocabulary (YOLOE, e.g. "yoloe-26s-seg.pt"): pass `prompts`, the text classes to look for,
      e.g. ["identity card", "passport", "driving licence"]. Needs no training.
    - fixed classes (YOLO26 trained on our boxes, e.g. "models/id_yolo26n.pt"): leave `prompts` empty.

    Every box becomes one Region, most confident first (index 0). Boxes are rounded outward to whole
    pixels, clipped to the image, and near-duplicates are merged (YOLOE can mark the same card as both
    "identity card" and "driving licence", which must not count as two documents).
    """

    def __init__(
        self,
        weights: str = "yoloe-26s-seg.pt",
        prompts: list[str] | None = None,
        confidence: float = 0.25,     # boxes below this detector confidence are ignored
        duplicate_iou: float = 0.7,   # two boxes overlapping more than this are the same document
        image_size: int = 640,        # YOLO resizes the image to this (longest side) before detection
        device: str = "cpu",
        name: str = "yolo",
        model=None,                   # an already-loaded model; tests pass a fake here
    ):
        self.name = name
        self.weights = weights
        self.prompts = list(prompts or [])
        self.confidence = confidence
        self.duplicate_iou = duplicate_iou
        self.image_size = image_size
        self.device = device
        self.model = model if model is not None else self._load_model()

    def _load_model(self):
        # Imported here, not at the top of the file: only configurations that use YOLO need ultralytics/torch.
        from ultralytics import YOLO, YOLOE

        if not self.prompts:
            return YOLO(self.weights)
        model = YOLOE(self.weights)
        model.set_classes(self.prompts)   # turns the text prompts into the classes the model looks for
        return model

    def detect(self, record: ImageRecord) -> list[Region]:
        if record.image is None:                 # the fetch failed: nothing to look at
            return []
        [result] = self.model.predict(
            record.image, conf=self.confidence, imgsz=self.image_size, device=self.device, verbose=False
        )
        raw_boxes = result.boxes.xyxy.tolist()   # [[x0, y0, x1, y1], ...] as floats, in original image pixels
        confidences = result.boxes.conf.tolist()
        width, height = record.image.size

        candidates = []
        for raw_box, confidence in zip(raw_boxes, confidences):
            box = clip_box(raw_box, width, height)
            if box is not None:                  # None = nothing left inside the image
                candidates.append((box, float(confidence)))
        kept = drop_duplicates(candidates, self.duplicate_iou)

        return [
            Region(
                region_id=make_region_id(record.transaction_id, record.image_id, self.name, index),
                transaction_id=record.transaction_id,
                image_id=record.image_id,
                bbox=box,
                confidence=confidence,
                detector_name=self.name,
                crop=record.image.crop(box),     # exactly the pixels inside bbox
            )
            for index, (box, confidence) in enumerate(kept)
        ]


def clip_box(raw_box: list[float], width: int, height: int) -> Box | None:
    """Rounds a float box OUTWARD to whole pixels (never cuts off part of the document) and clips it to
    the image. Returns None when nothing of the box is left inside the image."""
    x0, y0, x1, y1 = raw_box
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union: 0 = no overlap, 1 = identical boxes."""
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    overlap = overlap_w * overlap_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def drop_duplicates(candidates: list[tuple[Box, float]], max_iou: float) -> list[tuple[Box, float]]:
    """Sorts boxes most confident first and drops any box that overlaps an already kept box by more
    than max_iou. Ties keep the original order, so the result is the same on every run."""
    kept: list[tuple[Box, float]] = []
    for box, confidence in sorted(candidates, key=lambda item: -item[1]):
        if all(box_iou(box, kept_box) <= max_iou for kept_box, _ in kept):
            kept.append((box, confidence))
    return kept
