"""Shared test fixtures: one small synthetic dataset for the whole test session."""

from pathlib import Path

import pytest
import yaml

from id_classifier.synthetic import SyntheticSource, write_dataset


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    """2 ID cards (front+back) + 1 passport per country, 3 negatives -> 5*5 + 3 = 28 images."""
    out = tmp_path_factory.mktemp("synthetic")
    return write_dataset(SyntheticSource(per_country=2, negatives=3, seed=7), out)


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
