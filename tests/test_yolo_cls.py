"""YoloClsClassifier and its k-fold dataset, with a fake model (no training, no ultralytics)."""

import json
from types import SimpleNamespace

from PIL import Image

from fakes import picture
from id_classifier.classifiers.yolo_cls import (YoloClsClassifier, build_cls_dataset, class_name, crop_hash,
                                                parse_class_name)
from id_classifier.config import REGISTRY


class FakeClsModel:
    """Always answers `name` with `probability`; remembers it was used."""

    def __init__(self, name: str, probability: float):
        self.name, self.probability, self.calls = name, probability, 0

    def predict(self, image, **settings):
        self.calls += 1
        return [SimpleNamespace(names={0: self.name}, probs=SimpleNamespace(top1=0, top1conf=self.probability))]


def test_class_names_round_trip():
    assert class_name("none", "unknown", "n/a") == "none__unknown__na"         # no "/" in a folder name
    assert parse_class_name("none__unknown__na") == ("none", "unknown", "n/a")
    assert parse_class_name(class_name("national_id", "TN", "front")) == ("national_id", "TN", "front")


def test_dataset_holds_each_crop_out_of_exactly_one_fold(tmp_path):
    crops = [(picture(n), "national_id__TN__front") for n in range(6)] + [(picture(10), "national_id__MA__back")]
    manifest = build_cls_dataset(crops, tmp_path / "ds", folds=3)
    held = manifest["held_out"]
    assert sorted(h for hashes in held.values() for h in hashes) == sorted(crop_hash(c) for c, _ in crops)
    assert [len(held[f"fold{f}"]) for f in range(3)] == [3, 2, 2]               # TN round-robin 2/2/2 + MA in fold0
    assert len(list((tmp_path / "ds" / "fold0" / "train" / "national_id__TN__front").iterdir())) == 4
    assert not (tmp_path / "ds" / "fold0" / "train" / "national_id__MA__back").exists()   # held out of fold0
    assert len(list((tmp_path / "ds" / "full" / "train").rglob("*.jpg"))) == 7


def test_classifier_uses_the_model_that_never_saw_the_crop():
    crop = picture(1)
    manifest = {"folds": 2, "held_out": {"fold0": [crop_hash(crop)], "fold1": []}, "weights": {}}
    models = {"fold0": FakeClsModel("national_id__TN__back", 0.9), "fold1": FakeClsModel("x__XX__front", 0.9),
              "full": FakeClsModel("national_id__MA__front", 0.9)}
    classifier = YoloClsClassifier("unused", models=models, manifest=manifest)
    prediction = classifier.classify(crop)
    assert (prediction.document_type, prediction.issuing_country, prediction.document_side) == ("national_id", "TN", "back")
    assert json.loads(prediction.raw_output)["model"] == "fold0"
    new = classifier.classify(picture(2))                                      # not in any fold: production model
    assert (new.issuing_country, json.loads(new.raw_output)["model"]) == ("MA", "full")


def test_low_probability_is_unknown():
    classifier = YoloClsClassifier("unused", min_probability=0.6, manifest={"held_out": {}, "weights": {}},
                                   models={"full": FakeClsModel("national_id__TN__front", 0.4)})
    prediction = classifier.classify(picture(3))
    assert (prediction.document_type, prediction.issuing_country, prediction.document_side) == ("other", "unknown", "unknown")
    assert prediction.confidence == 0.4


def test_yolo_cls_is_registered():
    assert REGISTRY["classifier"]["yolo_cls"] == "id_classifier.classifiers.yolo_cls:YoloClsClassifier"
