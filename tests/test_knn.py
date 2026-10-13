"""EmbeddingKnnClassifier decision rule: k-vote, 1-NN margin fallback, and the three outcomes.

Embeddings are faked: every solid-colour test image maps to a hand-made vector, so the cosine
similarities (and therefore the decisions) are exact and known in advance."""

import json

import numpy as np
import pytest
from PIL import Image

from fakes import image_bytes
from id_classifier import config
from id_classifier.classifiers.knn import EmbeddingKnnClassifier
from id_classifier.types import OUTCOME_CLASSIFIED, OUTCOME_NO_DOCUMENT, OUTCOME_UNRECOGNISED

TN_F = ("national_id", "TN", "front")
TN_B = ("national_id", "TN", "back")
DZ_F = ("national_id", "DZ", "front")
DZ_B = ("national_id", "DZ", "back")
NONE = ("none", "unknown", "n/a")


def unit(angle_deg: float, axis: int = 1, dims: int = 4) -> np.ndarray:
    """A unit vector at `angle_deg` from the x axis, towards `axis`: cosine to x = cos(angle)."""
    v = np.zeros(dims)
    v[0], v[axis] = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return v


def make_knn(tmp_path, monkeypatch, references: list[tuple[tuple, np.ndarray]], **settings) -> EmbeddingKnnClassifier:
    """A classifier whose references are the given (label, vector) pairs. Each reference is a solid-colour
    PNG; the fake embedder maps its colour back to the vector."""
    monkeypatch.setitem(config.REGISTRY["detector"], "fixed", "fakes:FixedDetector")
    tmp_path.mkdir(parents=True, exist_ok=True)
    vectors, rows = {}, ["transaction_id,image_id,image_path,document_type,issuing_country,document_side"]
    for i, (label, vector) in enumerate(references):
        color = (i, 10, 20)
        vectors[color] = vector
        path = tmp_path / f"ref{i}.png"
        path.write_bytes(image_bytes(color))
        rows.append(f"{i + 1},front,{path.as_posix()},{','.join(label)}")
    csv_path = tmp_path / "refs.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def embed(image: Image.Image) -> np.ndarray:
        color = image.convert("RGB").getpixel((0, 0))
        return vectors.get(color, image.info.get("vector"))

    return EmbeddingKnnClassifier(str(csv_path), reference_detector={"type": "fixed"}, cache_dir=None,
                                  embedder=embed, **settings)


def query(vector: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (4, 4), (250, 250, 250))
    image.info["vector"] = vector                     # read back by the fake embedder
    return image


@pytest.fixture
def knn(tmp_path, monkeypatch):
    """Only one reference is needed for decide() tests; the rule is fed neighbours directly."""
    return lambda **settings: make_knn(tmp_path, monkeypatch, [(TN_F, unit(0))], **settings)


# ---------------------------------------------------------------- decide(): the rule itself

DILUTED = [(DZ_F, 0.86), (TN_F, 0.80), (TN_F, 0.79), (TN_F, 0.79), (TN_F, 0.78)]   # 1 Algerian vs 4 Tunisian


def test_vote_wins_without_fallback(knn):
    d = knn(min_similarity=0.72).decide(DILUTED, best_other_class=0.80)
    assert d["label"] == TN_F and d["outcome"] == OUTCOME_CLASSIFIED and d["decision"] == "vote"
    assert d["confidence"] == pytest.approx((0.80 + 0.79 + 0.79 + 0.78) / sum(s for _, s in DILUTED))


def test_fallback_trusts_a_clearly_closer_nearest_reference(knn):
    d = knn(min_similarity=0.72, nn_fallback=True, nn_min_similarity=0.80, nn_margin=0.05,
            nn_confidence=0.9).decide(DILUTED, best_other_class=0.80)
    assert d["label"] == DZ_F and d["decision"] == "1nn" and d["confidence"] == 0.9
    assert d["margin"] == pytest.approx(0.06)


def test_fallback_rejected_when_margin_too_small(knn):
    d = knn(min_similarity=0.72, nn_fallback=True, nn_min_similarity=0.80, nn_margin=0.05).decide(
        DILUTED, best_other_class=0.84)
    assert d["label"] == TN_F and d["decision"] == "vote (1nn rejected)"


def test_fallback_rejected_when_nearest_not_similar_enough(knn):
    d = knn(min_similarity=0.72, nn_fallback=True, nn_min_similarity=0.90, nn_margin=0.0).decide(
        DILUTED, best_other_class=0.80)
    assert d["label"] == TN_F and d["decision"] == "vote (1nn rejected)"


def test_unanimous_neighbours_never_use_the_fallback(knn):
    tunisian = [(TN_F, 0.95), (TN_F, 0.94), (TN_F, 0.93), (TN_F, 0.93), (TN_F, 0.92)]
    d = knn(nn_fallback=True).decide(tunisian, best_other_class=0.60)
    assert d["label"] == TN_F and d["decision"] == "vote" and d["confidence"] == pytest.approx(1.0)


def test_below_floor_is_unrecognised_with_a_best_guess_not_unknown(knn):
    d = knn(min_similarity=0.72).decide([(DZ_F, 0.60), (TN_F, 0.55)], best_other_class=0.55)
    assert d["outcome"] == OUTCOME_UNRECOGNISED and d["label"] == DZ_F   # "unknown" is reserved for no document
    assert d["confidence"] == pytest.approx(0.60)


def test_blank_reference_nearest_means_no_document(knn):
    below = knn(min_similarity=0.72).decide([(NONE, 0.60)], best_other_class=None)
    voted = knn(min_similarity=0.72).decide([(NONE, 0.90), (NONE, 0.88), (TN_F, 0.70)], best_other_class=0.70)
    assert below["outcome"] == voted["outcome"] == OUTCOME_NO_DOCUMENT


def test_no_reference_left_is_unrecognised(knn):
    assert knn().decide([], best_other_class=None)["outcome"] == OUTCOME_UNRECOGNISED


# ---------------------------------------------------------------- classify(): embedding + margin + leave-one-out

def test_small_class_recognised_end_to_end(tmp_path, monkeypatch):
    """1 Algerian reference among 4 Tunisian ones: the vote says TN, the 1-NN fallback says DZ."""
    refs = [(DZ_F, unit(10)), (TN_F, unit(28)), (TN_F, unit(29)), (TN_F, unit(30)), (TN_F, unit(31))]
    q = query(unit(0))                                              # cos to DZ = 0.985, to the TN ones ~0.87-0.88
    plain = make_knn(tmp_path / "a", monkeypatch, refs, min_similarity=0.5).classify(q)
    fallback = make_knn(tmp_path / "b", monkeypatch, refs, min_similarity=0.5, nn_fallback=True,
                        nn_min_similarity=0.9, nn_margin=0.05).classify(q)
    assert plain.issuing_country == "TN"
    assert fallback.issuing_country == "DZ" and fallback.confidence == 0.9
    raw = json.loads(fallback.raw_output)
    assert raw["decision"] == "1nn" and raw["margin"] == pytest.approx(np.cos(np.radians(10)) - np.cos(np.radians(28)), abs=1e-3)


def test_margin_compares_against_another_class_not_the_other_side(tmp_path, monkeypatch):
    """The Algerian BACK being close does not count against an Algerian front: same document class."""
    refs = [(DZ_F, unit(10)), (DZ_B, unit(11)), (TN_F, unit(28)), (TN_F, unit(29)), (TN_F, unit(30))]
    p = make_knn(tmp_path, monkeypatch, refs, min_similarity=0.5, nn_fallback=True, nn_min_similarity=0.9,
                 nn_margin=0.05).classify(query(unit(0)))
    assert json.loads(p.raw_output)["margin"] > 0.05 and p.issuing_country == "DZ"


def test_leave_one_out_skips_the_query_itself(tmp_path, monkeypatch):
    refs = [(DZ_F, unit(0)), (TN_F, unit(40))]
    p = make_knn(tmp_path, monkeypatch, refs, min_similarity=0.5).classify(query(unit(0)))
    assert json.loads(p.raw_output)["neighbours"][0][0] == "national_id/TN/front"   # its own picture is skipped


def test_fallback_off_by_default_keeps_old_behaviour(tmp_path, monkeypatch):
    clf = make_knn(tmp_path, monkeypatch, [(TN_F, unit(0))])
    assert clf.nn_fallback is False and "nn_" not in clf.model_version
