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
        "yolo": "id_classifier.detectors:YoloDetector",          # needs the [models] extra (ultralytics)
    },
    "classifier": {
        "knn": "id_classifier.classifiers.knn:EmbeddingKnnClassifier",   # needs the [models] extra (torch, transformers)
    },
    # Where the images are read from. Settings (bucket, prefix, endpoint, strip_prefixes) go in YAML;
    # credentials never do: each SDK reads them from the environment (see fetchers.py).
    "source": {
        "nfs": "id_classifier.sources:NfsZipSource",       # files on disk / the NFS share (the default)
        "s3": "id_classifier.fetchers:s3_source",          # needs the [s3] extra
        "azure": "id_classifier.fetchers:azure_source",    # needs the [azure] extra
        "gcs": "id_classifier.fetchers:gcs_source",        # needs the [gcs] extra
    },
}


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build(kind: str, spec: dict) -> Any:
    """Creates one component (kind = "detector" | "classifier" | "source") from its YAML dict."""
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


def build_source(spec: dict | None):
    """The image source of an experiment, or None when the YAML has no `source:` (= read files from disk)."""
    return build("source", spec) if spec else None
