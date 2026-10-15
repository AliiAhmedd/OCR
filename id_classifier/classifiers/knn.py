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
from collections import Counter
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

ROTATIONS = (0, 90, 180, 270)  # degrees, counter-clockwise: the views a query crop can be embedded in
QUERY_ROTATION_MODES = ("none", "portrait", "all")


def rotations_for(width: int, height: int, mode: str) -> tuple[int, ...]:
    """Which turns of a query crop are compared with the references (the best match per reference counts).
    DINOv2 is not rotation invariant: a card photographed sideways lands next to other sideways cards.
      none:     the crop as it is (0)
      portrait: a crop taller than wide is also tried turned 90 and 270 (cards are landscape; 0 is kept, so an
                upright portrait passport page still matches as it is); a landscape crop stays one view
      all:      0, 90, 180, 270 for every crop (also catches upside-down cards; 4x the embedding time)"""
    if mode not in QUERY_ROTATION_MODES:
        raise ValueError(f"query_rotations {mode!r} not in {QUERY_ROTATION_MODES}")
    if mode == "all":
        return ROTATIONS
    if mode == "portrait" and height > width:
        return (0, 90, 270)
    return (0,)


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
        nn_max_class_refs: int | None = None,     # 1-NN only for a nearest label with at most this many references
        nn_big_class_min_similarity: float | None = None,  # ...or, for a bigger class, only from this similarity on
        nn_side_check: bool = False,              # big class: 1-NN only if its class's neighbours vote the same side
        cache_dir: str | None = "data/cache",     # reference embeddings are saved here (None = no cache)
        embedder: Callable[[Image.Image], np.ndarray] | None = None,   # tests pass a fake
        source=None,                              # the experiment's YAML `source:`: references are fetched from it
        query_rotations: str = "none",            # none | portrait | all: see rotations_for()
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
        self.nn_max_class_refs = nn_max_class_refs
        self.nn_big_class_min_similarity = nn_big_class_min_similarity
        self.nn_side_check = nn_side_check
        rotations_for(1, 1, query_rotations)              # unknown mode -> error now, not on the first image
        self.query_rotations = query_rotations
        self.model_version = (
            f"{model},k={k},min_sim={min_similarity}"
            + (f",nn_sim={nn_min_similarity},nn_margin={nn_margin},nn_share={nn_trigger_share},nn_conf={nn_confidence}"
               if nn_fallback else "")
            + (f",nn_max_refs={nn_max_class_refs}" if nn_max_class_refs is not None else "")
            + (f",nn_big_sim={nn_big_class_min_similarity}" if nn_big_class_min_similarity is not None else "")
            + (",nn_side_check" if nn_side_check else "")
            + (f",rot={query_rotations}" if query_rotations != "none" else ""))
        self._embed = embedder or _DinoEmbedder(model)
        self.labels, self.vectors = self._load_references(references, reference_detector, model, cache_dir, source)
        self.class_refs = Counter(self.labels)            # references per label: big vs small classes
        log.info("kNN %s: %d reference crops", name, len(self.labels))

    def classify(self, crop: Image.Image) -> Prediction:
        start = time.perf_counter()
        angles = rotations_for(crop.width, crop.height, self.query_rotations)
        views = [self._embed(crop.rotate(a, expand=True) if a else crop) for a in angles]
        return self.classify_vector(np.stack(views) if len(views) > 1 else views[0], start, angles)

    def embed(self, crop: Image.Image) -> np.ndarray:
        """The normalised embedding of a crop, as classify() uses it (routing_tune stores these)."""
        return _normalize(self._embed(crop))

    def classify_vector(self, vector: np.ndarray, start: float | None = None,
                        angles: tuple[int, ...] = (0,)) -> Prediction:
        """classify() from an already computed embedding: the decision and routing settings can then be
        tuned without running the detector or the embedding model again (routing_tune.py).
        vector: one embedding, or one row per view (angles): each reference then keeps its best view."""
        start = time.perf_counter() if start is None else start
        views = np.atleast_2d(np.asarray(vector, dtype=np.float32))
        views = np.stack([_normalize(v) for v in views])
        per_view = self.vectors @ views.T                          # cosine similarity: references x views
        similarities = per_view.max(axis=1)                        # best view per reference
        candidates = [i for i in np.argsort(-similarities) if similarities[i] < self.exclude_self_above]
        neighbours = [(self.labels[i], float(similarities[i])) for i in candidates[: self.k]]
        decision = self.decide(neighbours, self._best_other_class(candidates, similarities))
        label, confidence, outcome = decision.pop("label"), decision.pop("confidence"), decision.pop("outcome")
        if len(views) > 1 and candidates:                         # which turn matched the nearest reference
            decision["rotation"] = int(angles[int(per_view[candidates[0]].argmax())])
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
        big = self.nn_max_class_refs is not None and self.class_refs.get(nearest_label, 0) > self.nn_max_class_refs
        if self.nn_fallback and diluted and big and self.nn_big_class_min_similarity is None:
            decision = "vote (1nn off: big class)"          # a big class (TN) keeps the k-vote: baseline accuracy
        elif self.nn_fallback and diluted:
            margin = None if best_other_class is None else best - best_other_class
            info["margin"] = None if margin is None else round(margin, 4)
            needed = self.nn_big_class_min_similarity if big else self.nn_min_similarity   # big class: stricter
            side = self._side_vote(neighbours, nearest_label) if big and self.nn_side_check else None
            if side is not None:
                info["side_vote"] = side
            if best >= needed and (margin is None or margin >= self.nn_margin):
                if side is not None and side != nearest_label[2]:
                    decision = "vote (1nn side mismatch)"   # TN front <-> back: the sides of its class disagree
                else:
                    label, confidence, decision = nearest_label, self.nn_confidence, "1nn"
            else:
                decision = "vote (1nn rejected)"
        outcome = OUTCOME_NO_DOCUMENT if label[0] == NO_DOCUMENT else OUTCOME_CLASSIFIED
        return {"label": label, "confidence": confidence, "outcome": outcome, "decision": decision, **info}

    @staticmethod
    def _side_vote(neighbours: list[tuple[Label, float]], nearest_label: Label) -> str | None:
        """The side that the k neighbours of the nearest reference's own type + country vote for (weighted by
        similarity), or None when there is no clear winner. The front and back of one card type look alike to
        DINOv2, so for a big class the nearest reference alone is not trusted on the side (progress.md row 38)."""
        votes: dict[str, float] = {}
        for (document_type, country, side), similarity in neighbours:
            if (document_type, country) == nearest_label[:2]:
                votes[side] = votes.get(side, 0.0) + similarity
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
            return None
        return ranked[0][0]

    def _load_references(self, csv_path, detector_spec, model, cache_dir, source=None) -> tuple[list[Label], np.ndarray]:
        """Without a source the reference images are read from disk; with one (YAML `source:`), every byte is
        fetched from that storage, exactly like the query images. A reference that cannot be read is an error,
        never skipped: a silently smaller reference set gives wrong scores that look fine."""
        with Path(csv_path).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cache = None
        if cache_dir is not None and isinstance(self._embed, _DinoEmbedder):
            parts = [Path(csv_path).read_text(encoding="utf-8"), detector_spec, model]
            if source is not None:                            # own cache per storage: each run reads its own bytes
                parts.append(getattr(source, "name", type(source).__name__))
            key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
            cache = Path(cache_dir) / f"knn_refs_{key}.npz"
            if cache.exists():
                data = np.load(cache, allow_pickle=False)
                return [tuple(l) for l in data["labels"].tolist()], data["vectors"]

        from id_classifier.config import build_detector      # here, not at the top: config imports this module
        detector = build_detector(detector_spec)
        labels, vectors, failed = [], [], []
        for row in rows:
            transaction_id, image_id = int(row["transaction_id"]), row["image_id"]
            if source is None:
                path = Path(row["image_path"])                # relative paths are relative to the CSV (as in evaluate)
                path = path if path.is_absolute() else Path(csv_path).parent / path
                record = read_image_file(transaction_id, image_id, path)
            else:
                record = source.fetch(transaction_id, image_id, row["image_path"])   # same reference as evaluate
            if record.fetch_status != "ok":
                failed.append(f"{row['image_path']} ({record.failure_reason})")
                continue
            regions = detector.detect(record)
            if not regions:                                   # no region (detector without fallback): no reference
                continue
            labels.append((row["document_type"], row["issuing_country"], row["document_side"]))
            vectors.append(_normalize(self._embed(regions[0].crop)))   # the most confident box
            record.image = None
        if failed:
            where = getattr(source, "name", "disk") if source is not None else "disk"
            raise ValueError(f"{len(failed)} of {len(rows)} reference images in {csv_path} could not be read from "
                             f"{where}, e.g. {failed[0]}")
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
