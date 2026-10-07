from PIL import Image

from fakes import FakeYoloModel, position_image
from id_classifier.config import REGISTRY
from id_classifier.detectors import FullImageDetector, YoloDetector, box_iou, clip_box
from id_classifier.types import ImageRecord


def test_full_image_detector_returns_whole_image():
    record = ImageRecord(5, "front", "ref", image=Image.new("RGB", (40, 30)))
    [region] = FullImageDetector().detect(record)
    assert region.bbox == (0, 0, 40, 30)
    assert region.region_id == "5:front:full_image:0"
    assert (region.transaction_id, region.image_id) == (5, "front")


def test_full_image_detector_on_failed_fetch():
    assert FullImageDetector().detect(ImageRecord(5, "front", "ref", fetch_status="failed")) == []


# --- YoloDetector, with a fake model that returns boxes we choose ---

def yolo(boxes, confidences, **settings) -> YoloDetector:
    return YoloDetector(model=FakeYoloModel(boxes, confidences), **settings)


def test_yolo_crop_is_exactly_the_box():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    [region] = yolo([(10, 20, 60, 50)], [0.9]).detect(record)
    assert region.bbox == (10, 20, 60, 50)
    assert region.crop.size == (50, 30)                         # width = x1 - x0, height = y1 - y0
    assert region.crop.getpixel((0, 0)) == (10, 20, 0)          # crop starts at the box's top-left pixel
    assert region.crop.getpixel((49, 29)) == (59, 49, 0)        # ... and ends just inside x1, y1
    assert region.confidence == 0.9
    assert (region.transaction_id, region.image_id, region.detector_name) == (7, "front", "yolo")


def test_yolo_box_rounded_outward_and_clipped_to_image():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    [region] = yolo([(-5.2, 10.7, 120.3, 79.9)], [0.8]).detect(record)
    assert region.bbox == (0, 10, 100, 80)                      # never larger than the image
    assert region.crop.size == (100, 70)


def test_yolo_regions_most_confident_first_with_stable_ids():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    regions = yolo([(0, 0, 40, 40), (50, 0, 90, 40)], [0.6, 0.9], name="yoloe").detect(record)
    assert [r.confidence for r in regions] == [0.9, 0.6]
    assert [r.region_id for r in regions] == ["7:front:yoloe:0", "7:front:yoloe:1"]
    assert regions[0].bbox == (50, 0, 90, 40)


def test_yolo_same_card_found_twice_counts_once():
    """YOLOE can find one card as both "identity card" and "driving licence": keep only the best box."""
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    regions = yolo([(10, 10, 90, 70), (12, 11, 91, 70)], [0.7, 0.85]).detect(record)
    assert len(regions) == 1
    assert regions[0].confidence == 0.85


def test_yolo_two_separate_documents_are_both_kept():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    regions = yolo([(0, 0, 45, 80), (55, 0, 100, 80)], [0.9, 0.8]).detect(record)
    assert len(regions) == 2                                    # routing will flag multiple_documents


def test_yolo_nothing_found_and_box_outside_image_without_fallback():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    assert yolo([], [], fallback_full_image=False).detect(record) == []
    assert yolo([(150, 10, 200, 50)], [0.9], fallback_full_image=False).detect(record) == []   # outside: dropped


def test_yolo_fallback_whole_image_when_nothing_found():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    [region] = yolo([], []).detect(record)                          # fallback is on by default
    assert region.bbox == (0, 0, 100, 80)
    assert region.crop.size == (100, 80)
    assert region.region_id == "7:front:yolo:full"                  # never mistaken for a real box
    assert region.confidence == 0.0
    [outside] = yolo([(150, 10, 200, 50)], [0.9]).detect(record)    # only a box outside the image
    assert outside.region_id == "7:front:yolo:full"


def test_yolo_fallback_not_used_when_a_box_is_found():
    record = ImageRecord(7, "front", "ref", image=position_image(100, 80))
    [region] = yolo([(10, 20, 60, 50)], [0.9]).detect(record)
    assert region.region_id == "7:front:yolo:0"


def test_yolo_failed_fetch_does_not_call_the_model():   # and gets no fallback region either
    detector = yolo([(0, 0, 10, 10)], [0.9])
    assert detector.detect(ImageRecord(7, "front", "ref", fetch_status="failed")) == []
    assert detector.model.calls == []


def test_yolo_passes_its_settings_to_the_model():
    detector = yolo([], [], confidence=0.4, image_size=960)
    detector.detect(ImageRecord(7, "front", "ref", image=position_image(30, 20)))
    [call] = detector.model.calls
    assert (call["conf"], call["imgsz"], call["device"], call["image_size"]) == (0.4, 960, "cpu", (30, 20))


def test_box_helpers():
    assert clip_box([10.5, 10.5, 20.5, 20.5], 100, 100) == (10, 10, 21, 21)
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0      # touching edges do not overlap
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


def test_yolo_is_registered_for_yaml_configs():
    assert REGISTRY["detector"]["yolo"] == "id_classifier.detectors:YoloDetector"
