"""The Classifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from id_classifier.types import Prediction


class Classifier(ABC):
    """Interface every classifier implements.

    Rules for implementations:
    - never raise for a bad model answer: return a Prediction with status "parse_error" or "model_error"
    - labels must come from id_classifier.types (DOCUMENT_TYPES, DOCUMENT_SIDES, 2-letter country or "unknown")
    - measure latency_ms around the model call
    """

    name: str            # unique name inside one experiment, e.g. "qwen_vl_7b" or "mock_noisy"
    model_version: str   # what exactly ran (weights tag, settings), stored with every prediction

    @abstractmethod
    def classify(self, crop: Image.Image) -> Prediction:
        """Labels one document region."""
