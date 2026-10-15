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


def test_big_class_keeps_the_vote_small_class_may_use_1nn(tmp_path, monkeypatch):
    """nn_max_class_refs: TN (many references) keeps the baseline k-vote; a small class may still use 1-NN."""
    refs = [(TN_F, unit(0)), (TN_F, unit(3)), (TN_F, unit(6)), (TN_B, unit(9)), (TN_B, unit(12)), (TN_B, unit(15)),
            (DZ_F, unit(40, 2))]
    knn = make_knn(tmp_path, monkeypatch, refs, nn_fallback=True, nn_min_similarity=0.5, nn_margin=0.0,
                   nn_max_class_refs=2)
    assert knn.class_refs[TN_F] == 3 and knn.class_refs[DZ_F] == 1

    tn_nearest = [(TN_B, 0.95), (TN_F, 0.94), (TN_F, 0.93), (TN_F, 0.92)]       # vote TN_F, nearest TN_B
    big = knn.decide(tn_nearest, best_other_class=None)
    assert big["decision"] == "vote (1nn off: big class)" and big["label"] == TN_F

    dz_nearest = [(DZ_F, 0.95), (TN_F, 0.80), (TN_F, 0.79), (TN_F, 0.78)]       # vote TN_F, nearest DZ_F
    small = knn.decide(dz_nearest, best_other_class=0.80)
    assert small["decision"] == "1nn" and small["label"] == DZ_F

    knn.nn_max_class_refs = None                                                # gate off: as before
    assert knn.decide(tn_nearest, best_other_class=None)["decision"] != "vote (1nn off: big class)"


def test_big_class_may_use_1nn_only_above_its_stricter_similarity(tmp_path, monkeypatch):
    """nn_big_class_min_similarity: TN may still use 1-NN, but only when its nearest reference is very similar."""
    refs = [(TN_F, unit(0)), (TN_F, unit(3)), (TN_F, unit(6)), (TN_B, unit(9)), (TN_B, unit(12)), (TN_B, unit(15)),
            (DZ_F, unit(40, 2))]
    knn = make_knn(tmp_path, monkeypatch, refs, nn_fallback=True, nn_min_similarity=0.5, nn_margin=0.0,
                   nn_max_class_refs=2, nn_big_class_min_similarity=0.96)
    tn_nearest = [(TN_B, 0.95), (TN_F, 0.94), (TN_F, 0.93), (TN_F, 0.92)]       # vote TN_F, nearest TN_B at 0.95

    assert knn.decide(tn_nearest, None)["decision"] == "vote (1nn rejected)"    # 0.95 < 0.96: big-class bar
    knn.nn_big_class_min_similarity = 0.94
    assert knn.decide(tn_nearest, None)["label"] == TN_B                        # 0.95 >= 0.94: 1-NN allowed
    dz_nearest = [(DZ_F, 0.60), (TN_F, 0.55), (TN_F, 0.54), (TN_F, 0.53)]       # small class: normal bar 0.5
    assert knn.decide(dz_nearest, 0.55)["decision"] == "1nn"


def test_side_check_blocks_a_big_class_1nn_whose_side_its_neighbours_contradict(tmp_path, monkeypatch):
    """nn_side_check: TN front and back look alike, so a TN 1-NN must agree with the side its TN neighbours vote."""
    refs = [(TN_F, unit(0)), (TN_F, unit(3)), (TN_F, unit(6)), (TN_B, unit(9)), (TN_B, unit(12)), (TN_B, unit(15)),
            (DZ_F, unit(40, 2))]
    knn = make_knn(tmp_path, monkeypatch, refs, nn_fallback=True, nn_min_similarity=0.5, nn_margin=0.0,
                   nn_max_class_refs=2, nn_big_class_min_similarity=0.9, nn_side_check=True)

    front_wins = [(TN_B, 0.95), (TN_F, 0.94), (TN_F, 0.93), (TN_F, 0.92), (DZ_F, 0.5)]   # nearest back, TN votes front
    blocked = knn.decide(front_wins, None)
    assert blocked["decision"] == "vote (1nn side mismatch)" and blocked["side_vote"] == "front"

    back_wins = [(TN_B, 0.95), (TN_B, 0.94), (TN_F, 0.93), (DZ_F, 0.92), (DZ_F, 0.91)]   # diluted, TN votes back
    allowed = knn.decide(back_wins, 0.92)
    assert allowed["decision"] == "1nn" and allowed["label"] == TN_B

    knn.nn_side_check = False                                                     # check off: as row 37
    assert knn.decide(front_wins, None)["decision"] == "1nn"


def test_side_mismatch_accepts_the_vote_when_the_side_vote_agrees_strongly_enough(tmp_path, monkeypatch):
    """nn_side_vote_min_share: the 1-NN is blocked (wrong side), but the k-vote already has the side its TN
    neighbours vote: accept the vote when that side vote is one-sided enough."""
    refs = [(TN_F, unit(0)), (TN_F, unit(3)), (TN_F, unit(6)), (TN_B, unit(9)), (TN_B, unit(12)), (TN_B, unit(15)),
            (DZ_F, unit(40, 2))]
    knn = make_knn(tmp_path, monkeypatch, refs, nn_fallback=True, nn_min_similarity=0.5, nn_margin=0.0,
                   nn_max_class_refs=2, nn_big_class_min_similarity=0.9, nn_side_check=True,
                   nn_side_vote_min_share=0.7)
    front_wins = [(TN_B, 0.95), (TN_F, 0.94), (TN_F, 0.93), (TN_F, 0.92), (DZ_F, 0.5)]   # front share 2.79/3.74 = 0.746

    accepted = knn.decide(front_wins, None)
    assert accepted["decision"] == "vote (side vote agrees)" and accepted["label"] == TN_F
    assert accepted["confidence"] == knn.nn_confidence and accepted["side_vote_share"] == 0.746

    knn.nn_side_vote_min_share = 0.8                                             # not one-sided enough
    assert knn.decide(front_wins, None)["decision"] == "vote (1nn side mismatch)"
    knn.nn_side_vote_min_share = None                                            # off: as configuration 14
    assert knn.decide(front_wins, None)["decision"] == "vote (1nn side mismatch)"
