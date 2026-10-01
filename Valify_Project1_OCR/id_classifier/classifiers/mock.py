"""MockClassifier: a fake model for tests and demos. Deterministic, no ML libraries.

It "cheats": it reads the true label hidden inside synthetic PNGs (see synthetic.LABEL_KEY), then
deliberately gets a configurable share of images wrong. That lets the evaluation harness show
realistic-looking differences between configurations. Its scores say nothing about real models.
"""

from __future__ import annotations

import json
import random
import time

from PIL import Image

from id_classifier.classifiers.base import Classifier
from id_classifier.synthetic import LABEL_KEY
from id_classifier.types import (
    DOCUMENT_TYPES,
    NO_DOCUMENT,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    UNKNOWN,
    Prediction,
)

DEFAULT_COUNTRY_POOL = ("TR", "TN", "SA", "JO", "SD", "EG")  # countries a wrong answer is picked from


class MockClassifier(Classifier):
    def __init__(
        self,
        name: str = "mock",
        error_rate: float = 0.0,        # share of images that get one wrong label
        parse_error_rate: float = 0.0,  # share of images that pretend the model returned broken JSON
        seed: int = 0,                  # different seeds = different (but repeatable) mistakes
        country_pool: list[str] | None = None,
    ):
        self.name = name
        self.error_rate = error_rate
        self.parse_error_rate = parse_error_rate
        self.seed = seed
        self.country_pool = list(country_pool or DEFAULT_COUNTRY_POOL)
        self.model_version = f"error_rate={error_rate},parse_error_rate={parse_error_rate},seed={seed}"

    def classify(self, crop: Image.Image) -> Prediction:
        start = time.perf_counter()
        hidden = crop.info.get(LABEL_KEY)
        if hidden is None:  # not a synthetic image: behave like an unsure model
            answer = {"document_type": "other", "issuing_country": UNKNOWN, "document_side": UNKNOWN, "confidence": 0.0}
            return self._prediction(answer, start)

        truth = json.loads(hidden)
        rng = random.Random(f"{self.seed}|{truth['sample_id']}")  # same image + same seed -> same answer
        if rng.random() < self.parse_error_rate:
            return Prediction(
                None, None, None, None, STATUS_PARSE_ERROR, "<<simulated broken output>>",
                self.name, self.model_version, self._elapsed_ms(start), error="simulated parse error",
            )

        answer = {k: truth[k] for k in ("document_type", "issuing_country", "document_side")}
        if rng.random() < self.error_rate:
            answer = self._make_wrong(answer, rng)
            answer["confidence"] = round(rng.uniform(0.4, 0.9), 3)    # wrong answers are often less sure...
        else:
            answer["confidence"] = round(rng.uniform(0.85, 0.99), 3)  # ...right answers usually confident
        return self._prediction(answer, start)

    def _make_wrong(self, answer: dict, rng: random.Random) -> dict:
        wrong = dict(answer)
        if answer["document_type"] == NO_DOCUMENT:  # a mistake on an empty image = "seeing" a document
            wrong.update(document_type="national_id", issuing_country=rng.choice(self.country_pool), document_side="front")
            return wrong
        field = rng.choice(["document_type", "issuing_country", "document_side"])  # break one field
        if field == "document_type":
            choices = [t for t in DOCUMENT_TYPES if t != answer[field]]
        elif field == "issuing_country":
            choices = [c for c in self.country_pool if c != answer[field]] + [UNKNOWN]
        else:
            choices = [s for s in ("front", "back", UNKNOWN) if s != answer[field]]
        wrong[field] = rng.choice(choices)
        return wrong

    def _prediction(self, answer: dict, start: float) -> Prediction:
        return Prediction(
            document_type=answer["document_type"],
            issuing_country=answer["issuing_country"],
            document_side=answer["document_side"],
            confidence=answer["confidence"],
            status=STATUS_OK,
            raw_output=json.dumps(answer),
            model_name=self.name,
            model_version=self.model_version,
            latency_ms=self._elapsed_ms(start),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000
