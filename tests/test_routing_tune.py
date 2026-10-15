"""routing_tune: routing on stored embeddings must give exactly what the pipeline gives, for every setting."""

import numpy as np
import pytest
from PIL import Image

from fakes import FixedDetector
from id_classifier.classifiers.knn import ROTATIONS, rotations_for
from id_classifier.pipeline import process_image
from id_classifier.routing import RoutingConfig
from id_classifier.routing_tune import StoredImage, grid_settings, route_all, score
from id_classifier.types import ImageRecord
from test_knn import DZ_F, TN_B, TN_F, make_knn, unit


def query_record(i: int, size: tuple[int, int]) -> ImageRecord:
    """A query image with a unique pixel pattern, so each of its turns embeds differently."""
    rng = np.random.default_rng(i)
    image = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    return ImageRecord(i, "front", f"q{i}", image=image)


def pattern_embedder(knn, dims: int = 4):
    """Wraps make_knn's fake: solid reference images keep their vector; any other image (a query, in any turn)
    gets a fixed pseudo-random vector from its pixels, so pipeline and stored views see identical embeddings."""
    reference_embed = knn._embed

    def embed(image: Image.Image) -> np.ndarray:
        if all(low == high for low, high in image.convert("RGB").getextrema()):   # solid = a reference
            return reference_embed(image)
        seed = int.from_bytes(image.tobytes()[:8].ljust(8, b"0"), "little") ^ image.width
        v = np.random.default_rng(seed).normal(size=dims)
        return unit(0) * 0.9 + v * 0.3              # near the TN references: decisions vary per turn
    knn._embed = embed


@pytest.mark.parametrize("settings", grid_settings({"nn_fallback": [False, True], "min_similarity": [0.5, 0.9],
                                                    "query_rotations": ["none", "portrait", "all"]}))
@pytest.mark.parametrize("detector", [FixedDetector(), FixedDetector([0.9, 0.4])])   # 1 box / 2 boxes
def test_stored_embeddings_route_exactly_like_the_pipeline(tmp_path, monkeypatch, settings, detector):
    knn = make_knn(tmp_path, monkeypatch, [(TN_F, unit(0)), (TN_F, unit(5)), (TN_B, unit(10)), (DZ_F, unit(30, 2))])
    pattern_embedder(knn)
    for name, value in settings.items():
        setattr(knn, name, value)
    routing = RoutingConfig()
    sizes = [(8, 5), (5, 8), (6, 9), (9, 6)]          # landscape and portrait crops

    expected, images = [], []
    for i, size in enumerate(sizes):
        record = query_record(i, size)
        regions = detector.detect(record)
        views = [np.stack([knn.embed(r.crop.rotate(a, expand=True) if a else r.crop) for a in ROTATIONS])
                 for r in regions]                    # what the embedding pass stores
        expected.append(process_image(record, detector, [knn], routing).summary)
        record.image = None
        images.append(StoredImage(record, regions, views))

    got = route_all(images, knn, routing, detector.name)
    fields = lambda s: (s.document_type, s.issuing_country, s.document_side, s.confidence, s.needs_review,
                        s.reasons, s.outcome)
    assert [fields(s) for s in got] == [fields(s) for s in expected]


def test_rotation_modes_choose_the_turns():
    assert rotations_for(8, 5, "portrait") == (0,)              # landscape: as it is
    assert rotations_for(5, 8, "portrait") == (0, 90, 270)      # portrait: also turned sideways
    assert rotations_for(8, 5, "all") == ROTATIONS
    assert rotations_for(5, 8, "none") == (0,)
    with pytest.raises(ValueError):
        rotations_for(5, 8, "sideways")


def test_a_sideways_card_matches_its_upright_reference_with_rotation_max(tmp_path, monkeypatch):
    """The point of query_rotations: a portrait crop of a landscape card is compared in its upright turn too."""
    knn = make_knn(tmp_path, monkeypatch, [(TN_F, unit(0)), (DZ_F, unit(90))])
    upright = unit(0)
    knn._embed = lambda image: upright * 0.99 + unit(90) * 0.1 if image.width > image.height else unit(80)
    # sideways it looks like the DZ reference (0.985, below the 0.995 leave-one-out cut), upright like TN
    sideways = Image.new("RGB", (5, 8), (1, 2, 3))

    knn.query_rotations = "none"
    assert knn.classify(sideways).issuing_country == "DZ"
    knn.query_rotations = "portrait"
    prediction = knn.classify(sideways)
    assert prediction.issuing_country == "TN" and '"rotation": 90' in prediction.raw_output


def test_score_goal_needs_no_multi_sample_review_and_no_wrong_country():
    from types import SimpleNamespace as S

    t = lambda c, side="front": S(document_type="national_id", issuing_country=c, document_side=side, is_document=True)
    s = lambda c, review, side="front": S(document_type="national_id", issuing_country=c, document_side=side,
                                         needs_review=review, reasons=["low_confidence"] if review else [])
    truths = [t("TN"), t("TN"), t("TN"), t("DZ"), t("NG")]            # DZ and NG: one image each
    good = score(truths, [s("TN", False)] * 3 + [s("DZ", True), s("NG", True)])
    assert good["goal"] == "YES" and good["reviewed_single_sample"] == 2

    multi = score(truths, [s("TN", True), s("TN", False), s("TN", False), s("DZ", True), s("NG", True)])
    assert multi["reviewed_multi_sample"] == 1 and multi["goal"] == ""

    wrong = score(truths, [s("TN", False)] * 3 + [s("TN", False), s("NG", True)])   # DZ accepted as TN
    assert wrong["accepted_wrong_country"] == 1 and wrong["goal"] == "" and wrong["beyond_accepted"] == 1


def test_single_sample_rule_sends_one_image_classes_to_review():
    """review_single_sample_classes: a class with one image has no reference of its own under leave-one-out."""
    from types import SimpleNamespace as S

    from id_classifier.evaluate import SINGLE_SAMPLE_CLASS, class_sizes, review_single_sample

    t = lambda c: S(document_type="passport", issuing_country=c, document_side="front", is_document=True)
    truths = [t("NG"), t("EG"), t("EG"), S(document_type="none", issuing_country="unknown", document_side="n/a",
                                           is_document=False)]
    sizes = class_sizes(truths)
    accepted = lambda: S(needs_review=False, reasons=[])
    ng, eg, blank = accepted(), accepted(), accepted()
    for truth, summary in zip(truths, (ng, eg, accepted(), blank)):
        review_single_sample(truth, summary, sizes)

    assert ng.needs_review and ng.reasons == [SINGLE_SAMPLE_CLASS]     # NG: one image -> review
    assert not eg.needs_review                                          # EG: two images -> routed as before
    assert not blank.needs_review                                       # no document: not a class
    review_single_sample(truths[0], accepted(), None)                   # rule off: nothing changes


def test_portrait_box_turns_only_a_real_card_box():
    assert rotations_for(5, 8, "portrait_box") == (0, 90, 270)                     # portrait card box: turned
    assert rotations_for(5, 8, "portrait_box", whole_image=True) == (0,)           # whole portrait photo: as is
    assert rotations_for(8, 5, "portrait_box") == (0,)                             # landscape: as is
    assert rotations_for(5, 8, "portrait", whole_image=True) == (0, 90, 270)       # old mode: unchanged


class WholeImageDetector(FixedDetector):
    """Like YoloDetector finding no box: one region, id ':full', the crop tagged as the whole image."""

    def detect(self, record):
        from id_classifier.detectors import FULL_IMAGE_INDEX
        from id_classifier.types import WHOLE_IMAGE_INFO, Region, make_region_id

        record.image.info[WHOLE_IMAGE_INFO] = True
        box = (0, 0, *record.image.size)
        return [Region(make_region_id(record.transaction_id, record.image_id, self.name, FULL_IMAGE_INDEX),
                       record.transaction_id, record.image_id, box, 0.0, self.name, crop=record.image)]


@pytest.mark.parametrize("mode", ["none", "portrait", "portrait_box", "all"])
@pytest.mark.parametrize("detector", [FixedDetector(), WholeImageDetector()])        # card box / no box found
def test_portrait_box_routes_like_the_pipeline(tmp_path, monkeypatch, mode, detector):
    knn = make_knn(tmp_path, monkeypatch, [(TN_F, unit(0)), (TN_F, unit(5)), (TN_B, unit(10)), (DZ_F, unit(30, 2))],
                   nn_fallback=True)
    pattern_embedder(knn)
    knn.query_rotations = mode
    routing = RoutingConfig()
    expected, images = [], []
    for i, size in enumerate([(8, 5), (5, 8), (6, 9)]):
        record = query_record(i, size)
        regions = detector.detect(record)
        views = [np.stack([knn.embed(r.crop.rotate(a, expand=True) if a else r.crop) for a in ROTATIONS])
                 for r in regions]
        expected.append(process_image(record, detector, [knn], routing).summary)
        record.image = None
        images.append(StoredImage(record, regions, views))
    got = route_all(images, knn, routing, detector.name)
    fields = lambda s: (s.document_type, s.issuing_country, s.document_side, s.confidence, s.needs_review, s.reasons)
    assert [fields(s) for s in got] == [fields(s) for s in expected]
