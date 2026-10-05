from PIL import Image

from id_classifier.detectors import FullImageDetector
from id_classifier.types import ImageRecord


def test_full_image_detector_returns_whole_image():
    record = ImageRecord(5, "front", "ref", image=Image.new("RGB", (40, 30)))
    [region] = FullImageDetector().detect(record)
    assert region.bbox == (0, 0, 40, 30)
    assert region.region_id == "5:front:full_image:0"
    assert (region.transaction_id, region.image_id) == (5, "front")


def test_full_image_detector_on_failed_fetch():
    assert FullImageDetector().detect(ImageRecord(5, "front", "ref", fetch_status="failed")) == []
