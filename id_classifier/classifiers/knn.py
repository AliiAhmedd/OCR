"""EmbeddingKnnClassifier: label a crop by the labeled reference crops it looks most like.

1. Every reference image (a ground-truth CSV) is cropped by `reference_detector` (the same detector the
   pipeline uses, so references and queries are cropped alike) and turned into an embedding: a vector
   that describes how the image looks (DINOv2, a self-supervised vision model). Done once, then cached.
2. A query crop is embedded the same way and compared to every reference by cosine similarity
   (1 = looks identical, 0 = unrelated).
3. The k most similar references vote on the label triple (type, country, side), each vote weighted by
   its similarity. Confidence = the winning label's share of the votes.
4. 1-NN fallback (`nn_fallback`, off by default): a class with only a few references loses the k-vote to
   a big class even when its own reference is clearly the closest (1 Algerian card among 150 Tunisian
   ones). When the vote is diluted (the winner is not the nearest reference's label, or its share is
   below `nn_trigger_share`) the NEAREST reference decides instead, but only if it is similar enough
   (`nn_min_similarity`) AND clearly closer than the best reference of any other document class
   (`nn_margin`). Its confidence is then `nn_confidence`: >= the routing threshold auto-accepts such
   decisions, below it always sends them to review.
5. Three outcomes, never a forced label:
   - classified:   steps 3/4 gave a label.
   - unrecognised: even the nearest reference is below `min_similarity`. The nearest reference's labels
                   are kept as the reviewer's best guess, and routing sends the image to review.
   - no_document:  the chosen label is document type "none" (references can include blank photos).
   "unknown" country is reserved for "no document"; an unrecognised card keeps its best guess.

Fair evaluation: a reference with similarity >= `exclude_self_above` is the query image itself (the
ground truth is both reference set and test set), so it is skipped: leave-one-out.

Needs the [models] extra (torch, transformers). Tests pass their own `embedder`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from id_classifier.classifiers.base import Classifier
from id_classifier.sources import read_image_file
from id_classifier.types import (
    NO_DOCUMENT,
    OUTCOME_CLASSIFIED,
    OUTCOME_NO_DOCUMENT,
    OUTCOME_UNRECOGNISED,
    STATUS_OK,
    UNKNOWN,
    Prediction,
)

log = logging.getLogger(__name__)

Label = tuple[str, str, str]   # (document_type, issuing_country, document_side)


class EmbeddingKnnClassifier(Classifier):
    def __init__(
        self,
        references: str,                          # ground-truth CSV whose images are the labeled references
        name: str = "dinov2_knn",
        model: str = "facebook/dinov2-small",
        reference_detector: dict | None = None,   # detector spec used to crop the references (required)
        k: int = 5,
        min_similarity: float = 0.5,              # nearest reference below this -> "unrecognised" (review)
        exclude_self_above: float = 0.995,        # leave-one-out: skip the query's own picture
        nn_fallback: bool = False,                # step 4 of the module docstring
        nn_min_similarity: float = 0.8,           # 1-NN: the nearest reference must be at least this similar...
        nn_margin: float = 0.05,                  # ...and this much closer than any other document class
        nn_trigger_share: float = 0.8,            # 1-NN is tried when the vote winner's share is below this
        nn_confidence: float = 0.9,               # confidence reported for a 1-NN decision
        cache_dir: str | None = "data/cache",     # reference embeddings are saved here (None = no cache)
        embedder: Callable[[Image.Image], np.ndarray] | None = None,   # tests pass a fake
    ):
        self.name = name
        self.k = k
        self.min_similarity = min_similarity
        self.exclude_self_above = exclude_self_above
        self.nn_fallback = nn_fallback
        self.nn_min_similarity = nn_min_similarity
        self.nn_margin = nn_margin
        self.nn_trigger_share = nn_trigger_share
        self.nn_confidence = nn_confidence
        self.model_version = f"{model},k={k},min_sim={min_similarity}" + (
            f",nn_sim={nn_min_similarity},nn_margin={nn_margin},nn_share={nn_trigger_share},nn_conf={nn_confidence}"
            if nn_fallback else "")
        self._embed = embedder or _DinoEmbedder(model)
        self.labels, self.vectors = self._load_references(references, reference_detector, model, cache_dir)
        log.info("kNN %s: %d reference crops", name, len(self.labels))

    def classify(self, crop: Image.Image) -> Prediction:
        start = time.perf_counter()
        query = _normalize(self._embed(crop))
        similarities = self.vectors @ query                        # cosine similarity to every reference
        candidates = [i for i in np.argsort(-similarities) if similarities[i] < self.exclude_self_above]
        neighbours = [(self.labels[i], float(similarities[i])) for i in candidates[: self.k]]
        decision = self.decide(neighbours, self._best_other_class(candidates, similarities))
        label, confidence, outcome = decision.pop("label"), decision.pop("confidence"), decision.pop("outcome")
        raw = json.dumps({**decision, "neighbours": [["/".join(l), round(s, 4)] for l, s in neighbours]})
        return Prediction(*label, confidence=round(confidence, 4), status=STATUS_OK, raw_output=raw,
                          model_name=self.name, model_version=self.model_version,
                          latency_ms=(time.perf_counter() - start) * 1000, outcome=outcome)

    def _best_other_class(self, candidates: list[int], similarities: np.ndarray) -> float | None:
        """Similarity of the most similar reference whose document class (type + country, side ignored)
        differs from the nearest reference's. None when every reference is of that class."""
        if not candidates:
            return None
        nearest_class = self.labels[candidates[0]][:2]
        for i in candidates[1:]:                                   # candidates are sorted, most similar first
            if self.labels[i][:2] != nearest_class:
                return float(similarities[i])
        return None

    def decide(self, neighbours: list[tuple[Label, float]], best_other_class: float | None) -> dict:
        """The decision rule on its own (no embedding), so it can be tested and explained step by step.
        neighbours: the k nearest references, most similar first. Returns label, confidence, outcome and the
        numbers that led there (stored in raw_output so every decision can be audited)."""
        if not neighbours:                                         # no reference left after leave-one-out
            return {"label": ("other", UNKNOWN, UNKNOWN), "confidence": 0.0, "outcome": OUTCOME_UNRECOGNISED,
                    "decision": "no_reference", "best_similarity": 0.0}
        nearest_label, best = neighbours[0]
        info = {"best_similarity": round(best, 4)}
        if best < self.min_similarity:                             # looks like nothing we know
            outcome = OUTCOME_NO_DOCUMENT if nearest_label[0] == NO_DOCUMENT else OUTCOME_UNRECOGNISED
            return {"label": nearest_label, "confidence": best, "outcome": outcome,
                    "decision": "below_min_similarity", **info}

        votes: dict[Label, float] = {}
        for neighbour_label, similarity in neighbours:
            votes[neighbour_label] = votes.get(neighbour_label, 0.0) + similarity
        label = max(votes, key=votes.get)
        share = votes[label] / sum(votes.values())
        info["vote_share"] = round(share, 4)
        confidence, decision = share, "vote"

        diluted = label != nearest_label or share < self.nn_trigger_share
        if self.nn_fallback and diluted:
            margin = None if best_other_class is None else best - best_other_class
            info["margin"] = None if margin is None else round(margin, 4)
            if best >= self.nn_min_similarity and (margin is None or margin >= self.nn_margin):
                label, confidence, decision = nearest_label, self.nn_confidence, "1nn"
            else:
                decision = "vote (1nn rejected)"
        outcome = OUTCOME_NO_DOCUMENT if label[0] == NO_DOCUMENT else OUTCOME_CLASSIFIED
        return {"label": label, "confidence": confidence, "outcome": outcome, "decision": decision, **info}

    def _load_references(self, csv_path, detector_spec, model, cache_dir) -> tuple[list[Label], np.ndarray]:
        with Path(csv_path).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cache = None
        if cache_dir is not None and isinstance(self._embed, _DinoEmbedder):
            key = hashlib.sha256(json.dumps([Path(csv_path).read_text(encoding="utf-8"), detector_spec, model],
                                            sort_keys=True).encode()).hexdigest()[:16]
            cache = Path(cache_dir) / f"knn_refs_{key}.npz"
            if cache.exists():
                data = np.load(cache, allow_pickle=False)
                return [tuple(l) for l in data["labels"].tolist()], data["vectors"]

        from id_classifier.config import build_detector      # here, not at the top: config imports this module
        detector = build_detector(detector_spec)
        labels, vectors = [], []
        for row in rows:
            path = Path(row["image_path"])                    # relative paths are relative to the CSV (as in evaluate)
            path = path if path.is_absolute() else Path(csv_path).parent / path
            record = read_image_file(int(row["transaction_id"]), row["image_id"], path)
            regions = detector.detect(record)
            if not regions:                                   # unreadable image: no reference
                continue
            labels.append((row["document_type"], row["issuing_country"], row["document_side"]))
            vectors.append(_normalize(self._embed(regions[0].crop)))   # the most confident box
            record.image = None
        if not vectors:
            raise ValueError(f"No reference image could be read from {csv_path}")
        matrix = np.stack(vectors)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, labels=np.array(labels), vectors=matrix)
        return labels, matrix


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class _DinoEmbedder:
    """DINOv2 image embedding: the CLS token of the last layer (one vector per image)."""

    def __init__(self, model: str):
        import torch                                           # heavy imports only when this classifier is used
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()

    def __call__(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self._torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state[0, 0].numpy()
