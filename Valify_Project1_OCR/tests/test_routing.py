from id_classifier import routing
from id_classifier.routing import RoutingConfig, combine, route_image, summarize_image
from id_classifier.types import ImageRecord, Prediction, Region

CONFIG = RoutingConfig(confidence_threshold=0.8)


def region(index=0):
    return Region(f"1:front:full_image:{index}", 1, "front", (0, 0, 10, 10), 1.0, "full_image")


def pred(doc="national_id", country="TR", side="front", conf=0.95, status="ok", name="m"):
    if status != "ok":
        doc = country = side = conf = None
    return Prediction(doc, country, side, conf, status, "{}", name, "v1", 1.0)


def test_confident_single_prediction_is_auto_accepted():
    r = combine(region(), [pred()], CONFIG)
    assert not r.needs_review and r.reasons == []
    assert (r.document_type, r.issuing_country, r.document_side) == ("national_id", "TR", "front")


def test_low_confidence_goes_to_review():
    r = combine(region(), [pred(conf=0.5)], CONFIG)
    assert r.needs_review and r.reasons == [routing.LOW_CONFIDENCE]


def test_missing_confidence_counts_as_low():
    r = combine(region(), [pred(conf=None)], CONFIG)
    assert routing.LOW_CONFIDENCE in r.reasons


def test_unknown_country_goes_to_review():
    r = combine(region(), [pred(country="unknown")], CONFIG)
    assert r.reasons == [routing.UNKNOWN_COUNTRY]


def test_disagreement_goes_to_review_and_tie_is_not_forced():
    r = combine(region(), [pred(country="TR", name="a"), pred(country="TN", name="b")], CONFIG)
    assert routing.DISAGREEMENT in r.reasons
    assert r.issuing_country is None           # 1 vs 1: no winner picked
    assert r.document_type == "national_id"    # fields they agree on are kept


def test_majority_vote_with_three_classifiers():
    r = combine(region(), [pred(country="TR", name="a"), pred(country="TR", name="b"), pred(country="TN", name="c")], CONFIG)
    assert r.issuing_country == "TR" and routing.DISAGREEMENT in r.reasons


def test_parse_error_never_becomes_a_label():
    r = combine(region(), [pred(status="parse_error")], CONFIG)
    assert r.needs_review and "parse_error" in r.reasons
    assert (r.document_type, r.issuing_country, r.document_side) == (None, None, None)


def test_one_failed_classifier_in_ensemble_is_flagged():
    r = combine(region(), [pred(name="a"), pred(status="model_error", name="b")], CONFIG)
    assert r.issuing_country == "TR" and r.reasons == ["model_error"]


def test_classifier_says_no_document():
    r = combine(region(), [pred(doc="none", country="unknown", side="n/a")], CONFIG)
    assert r.reasons == [routing.NO_DOCUMENT_FOUND]   # not also "unknown_country"


def test_no_region_detected():
    record = ImageRecord(1, "front", "ref", image=object())
    [r] = route_image(record, [], {}, CONFIG, "yolo")
    assert r.document_type == "none" and r.reasons == [routing.NO_DOCUMENT_FOUND]
    assert r.region_id == "1:front:yolo:none"


def test_multiple_documents_in_one_image():
    regions = [region(0), region(1)]
    preds = {regions[0].region_id: [pred(conf=0.9)], regions[1].region_id: [pred(side="back", conf=0.99)]}
    routed = route_image(ImageRecord(1, "front", "ref"), regions, preds, CONFIG, "full_image")
    assert all(routing.MULTIPLE_DOCUMENTS in r.reasons for r in routed)
    assert summarize_image(routed).document_side == "back"   # the most confident region represents the image


def test_failed_fetch():
    [r] = route_image(ImageRecord(1, "front", "ref", fetch_status="failed"), [], {}, CONFIG, "full_image")
    assert r.needs_review and r.reasons == [routing.FETCH_FAILED] and r.document_type is None
