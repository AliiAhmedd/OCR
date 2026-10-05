"""Shared test fixtures: a few colour-coded images (see fakes.py) and the fake classifier registered as "fake"."""

from pathlib import Path

import pytest
import yaml

from fakes import COLOR_OF, image_bytes
from id_classifier.config import REGISTRY
from id_classifier.types import GROUND_TRUTH_COLUMNS

# file stem -> label: 4 documents (TN front+back, MA ID, MA licence) + 1 image without a document
IMAGES = {
    "101_front": ("national_id", "TN", "front"),
    "101_back": ("national_id", "TN", "back"),
    "102_front": ("national_id", "MA", "front"),
    "103_front": ("driving_license", "MA", "front"),
    "104_front": ("none", "unknown", "n/a"),
}


@pytest.fixture(autouse=True)
def fake_classifier(monkeypatch):
    """Makes `type: fake` available in YAML configs (the package itself ships no test classifier)."""
    monkeypatch.setitem(REGISTRY["classifier"], "fake", "fakes:ColorClassifier")


@pytest.fixture
def images_dir(tmp_path) -> Path:
    folder = tmp_path / "images"
    folder.mkdir()
    for stem, label in IMAGES.items():
        (folder / f"{stem}.png").write_bytes(image_bytes(COLOR_OF[label]))
    return folder


@pytest.fixture
def ground_truth_csv(images_dir) -> Path:
    lines = [",".join(GROUND_TRUTH_COLUMNS)]
    for stem, (document_type, country, side) in IMAGES.items():
        transaction_id, image_id = stem.split("_")
        lines.append(f"{transaction_id},{image_id},images/{stem}.png,{document_type},{country},{side}")
    path = images_dir.parent / "ground_truth.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def write_experiment(tmp_path):
    """Writes an experiment YAML into tmp_path and returns its path (safe_dump handles Windows paths)."""

    def _write(ground_truth: Path, configurations: list[dict], **extra) -> Path:
        config = {
            "experiment_name": "test",
            "ground_truth": str(ground_truth),
            "output_dir": str(tmp_path / "runs"),
            "repeats": 2,
            "routing": {"confidence_threshold": 0.8},
            "configurations": configurations,
            **extra,
        }
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    return _write
