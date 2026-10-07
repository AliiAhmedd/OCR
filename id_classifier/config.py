"""Config: read YAML files and build components from them.

A component is written in YAML as a small dict with a `type` and its settings, for example
    classifier: {type: knn, name: dinov2_knn, references: data/ground_truth.csv, ...}
`type` picks the class from the table below; every other key is passed to the class constructor.

Classes are listed as "module:ClassName" strings and imported only when used, so the heavy libraries
(torch, ultralytics, transformers) load only when a configuration needs them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

REGISTRY = {
    "detector": {
        "yolo": "id_classifier.detectors:YoloDetector",          # needs the [yolo] extra (ultralytics)
    },
    "classifier": {
        "knn": "id_classifier.classifiers.knn:EmbeddingKnnClassifier",   # needs the [embeddings] extra
    },
}


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build(kind: str, spec: dict) -> Any:
    """Creates one component (kind = "detector" | "classifier") from its YAML dict."""
    settings = dict(spec)                     # copy, so the caller's dict is not changed
    type_name = settings.pop("type", None)
    options = REGISTRY[kind]
    if type_name not in options:
        raise ValueError(f"Unknown {kind} type {type_name!r}. Options: {sorted(options)}")
    module_name, class_name = options[type_name].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**settings)


def build_detector(spec: dict):
    if not spec:
        raise ValueError("A detector is needed, e.g. detector: {type: yolo, ...}")
    return build("detector", spec)


def build_classifiers(specs: list[dict]) -> list:
    if not specs:
        raise ValueError("At least one classifier is needed")
    classifiers = [build("classifier", s) for s in specs]
    names = [c.name for c in classifiers]
    if len(set(names)) != len(names):         # names are stored per prediction, so they must be unique
        raise ValueError(f"Classifier names must be unique within one configuration, got {names}")
    return classifiers
