"""YoloClsClassifier: a small Ultralytics YOLO26 classification model trained on our own document crops.

Classes are label triples, written as folder names like "national_id__TN__front" (side "n/a" -> "na").

Honest evaluation with k-fold cross-validation: the ground-truth crops are split into `folds` parts and one
model is trained per part WITHOUT that part. At evaluation, a crop is classified by the model that never saw
it, found by the hash of the crop's pixels (the same detector crops the same image identically). Crops that
belong to no fold (new images, production) use the model trained on everything ("full").

Below `min_probability` the answer is "unknown" (routing sends it to review): the model has no "none of
these" class, so a low top probability is its only way to say "I don't know this document".

Training writes the crops to `dataset_dir` (REAL ID images, gitignored) and the models to `output_dir`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from id_classifier.classifiers.base import Classifier
from id_classifier.types import STATUS_OK, UNKNOWN, Prediction

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


def class_name(document_type: str, country: str, side: str) -> str:
    return "__".join((document_type, country, "na" if side == "n/a" else side))


def parse_class_name(name: str) -> tuple[str, str, str]:
    document_type, country, side = name.split("__")
    return document_type, country, "n/a" if side == "na" else side


def crop_hash(crop: Image.Image) -> str:
    """Fingerprint of the crop's pixels: the same image cropped by the same detector gives the same hash."""
    return hashlib.sha256(crop.convert("RGB").tobytes()).hexdigest()


def build_cls_dataset(crops: list[tuple[Image.Image, str]], dataset_dir: Path, folds: int, seed: int = 42) -> dict:
    """Writes one Ultralytics classification dataset per fold (train/<class>/*.jpg; val = a copy of train,
    only to satisfy Ultralytics: the honest score comes from `evaluate`) plus one for "full".
    Each label's crops are spread round-robin over the folds, so rare labels land in different folds.
    Returns the manifest: fold -> hashes of the crops held out of that fold's training."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    by_label = defaultdict(list)
    for crop, label in crops:
        by_label[label].append(crop)
    rng = random.Random(seed)
    fold_of = []                                              # (crop, label, fold)
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        fold_of += [(crop, label, i % folds) for i, crop in enumerate(items)]

    held_out = {f"fold{f}": [] for f in range(folds)}
    for index, (crop, label, fold) in enumerate(fold_of):
        held_out[f"fold{fold}"].append(crop_hash(crop))
        for run in [f"fold{f}" for f in range(folds) if f != fold] + ["full"]:
            for split in ("train", "val"):
                folder = dataset_dir / run / split / label
                folder.mkdir(parents=True, exist_ok=True)
                crop.convert("RGB").save(folder / f"{index:05d}.jpg", quality=90)
    return {"folds": folds, "held_out": held_out}


def train_cls_models(dataset_dir: Path, output_dir: Path, manifest: dict, base_weights: str = "yolo26n-cls.pt",
                     epochs: int = 15, image_size: int = 224) -> Path:
    """Trains one model per fold plus "full" and writes output_dir/manifest.json. Returns output_dir."""
    from ultralytics import YOLO                              # only when training

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [f"fold{f}" for f in range(manifest["folds"])] + ["full"]
    for run in runs:
        start = time.perf_counter()
        YOLO(base_weights).train(data=str((dataset_dir / run).resolve()), epochs=epochs, imgsz=image_size,
                                 batch=16, device="cpu", workers=0, project=str(output_dir.resolve()), name=run,
                                 exist_ok=True, plots=False, verbose=False, seed=42, deterministic=True)
        log.info("YOLO-cls: %s trained in %.0f s", run, time.perf_counter() - start)
    manifest = {**manifest, "weights": {run: f"{run}/weights/best.pt" for run in runs}}
    (output_dir / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return output_dir


class YoloClsClassifier(Classifier):
    def __init__(self, models_dir: str, name: str = "yolo26_cls", min_probability: float = 0.5,
                 models: dict | None = None, manifest: dict | None = None):   # tests pass fakes
        self.name = name
        self.min_probability = min_probability
        self.model_version = f"{models_dir},min_p={min_probability}"
        root = Path(models_dir)
        self.manifest = manifest or json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if models is None:
            from ultralytics import YOLO
            models = {run: YOLO(str(root / path)) for run, path in self.manifest["weights"].items()}
        self.models = models
        self.fold_of_hash = {h: run for run, hashes in self.manifest["held_out"].items() for h in hashes}

    def classify(self, crop: Image.Image) -> Prediction:
        start = time.perf_counter()
        run = self.fold_of_hash.get(crop_hash(crop), "full")   # the model that never saw this crop
        [result] = self.models[run].predict(crop.convert("RGB"), device="cpu", verbose=False)
        top_name = result.names[int(result.probs.top1)]
        probability = float(result.probs.top1conf)
        label = parse_class_name(top_name) if probability >= self.min_probability else ("other", UNKNOWN, UNKNOWN)
        raw = json.dumps({"model": run, "top1": top_name, "probability": round(probability, 4)})
        return Prediction(*label, confidence=round(probability, 4), status=STATUS_OK, raw_output=raw,
                          model_name=self.name, model_version=self.model_version,
                          latency_ms=(time.perf_counter() - start) * 1000)
