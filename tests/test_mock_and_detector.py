from PIL import Image

from id_classifier.classifiers.mock import MockClassifier
from id_classifier.detectors import FullImageDetector
from id_classifier.synthetic import SyntheticSource
from id_classifier.types import STATUS_OK, STATUS_PARSE_ERROR, ImageRecord


def _records(source):
    return [(s, source.fetch(s.transaction_id, s.image_id, s.reference)) for s in source.samples]


def test_full_image_detector_returns_whole_image():
    record = ImageRecord(5, "front", "ref", image=Image.new("RGB", (40, 30)))
    [region] = FullImageDetector().detect(record)
    assert region.bbox == (0, 0, 40, 30)
    assert region.region_id == "5:front:full_image:0"
    assert (region.transaction_id, region.image_id) == (5, "front")


def test_full_image_detector_on_failed_fetch():
    assert FullImageDetector().detect(ImageRecord(5, "front", "ref", fetch_status="failed")) == []


def test_perfect_mock_returns_the_truth():
    mock = MockClassifier(error_rate=0.0)
    for sample, record in _records(SyntheticSource(per_country=1, negatives=3, seed=0)):
        p = mock.classify(record.image)
        assert p.status == STATUS_OK
        assert (p.document_type, p.issuing_country, p.document_side) == (
            sample.document_type, sample.issuing_country, sample.document_side)
        assert p.confidence >= 0.85


def test_mock_is_deterministic_and_errors_happen():
    source = SyntheticSource(per_country=2, negatives=3, seed=0)
    a, b = MockClassifier(error_rate=1.0, seed=5), MockClassifier(error_rate=1.0, seed=5)
    for sample, record in _records(source):
        pa, pb = a.classify(record.image), b.classify(record.image)
        assert (pa.document_type, pa.issuing_country, pa.document_side) == (pb.document_type, pb.issuing_country, pb.document_side)
        assert (pa.document_type, pa.issuing_country, pa.document_side) != (
            sample.document_type, sample.issuing_country, sample.document_side)   # error_rate=1: always wrong


def test_parse_errors_are_errors_not_labels():
    mock = MockClassifier(parse_error_rate=1.0)
    _, record = _records(SyntheticSource(per_country=1, negatives=0, countries=["TR"]))[0]
    p = mock.classify(record.image)
    assert p.status == STATUS_PARSE_ERROR
    assert (p.document_type, p.issuing_country, p.document_side, p.confidence) == (None, None, None, None)


def test_mock_on_non_synthetic_image_is_unsure():
    p = MockClassifier().classify(Image.new("RGB", (10, 10)))
    assert p.issuing_country == "unknown" and p.confidence == 0.0
