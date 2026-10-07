"""EmbeddingKnnClassifier with a fake embedder: each known colour points in its own direction, so two images
are either identical (similarity 1) or unrelated (0). References = the colour images of conftest.py."""

import json

import numpy as np
from PIL import Image

from fakes import COLOR_LABELS
from id_classifier.classifiers.knn import EmbeddingKnnClassifier
from id_classifier.config import REGISTRY

COLOURS = list(COLOR_LABELS)


def one_hot_embedder(image: Image.Image) -> np.ndarray:
    """Nearest known colour -> its own axis; anything far from every known colour -> an extra axis."""
    pixel = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
    vector = np.zeros(len(COLOURS) + 1)
    distances = [sum((a - b) ** 2 for a, b in zip(c, pixel)) for c in COLOURS]
    vector[distances.index(min(distances)) if min(distances) < 3000 else len(COLOURS)] = 1.0
    return vector


def knn(ground_truth_csv, **settings) -> EmbeddingKnnClassifier:
    return EmbeddingKnnClassifier(references=str(ground_truth_csv), embedder=one_hot_embedder, cache_dir=None,
                                  **settings)


def test_references_are_loaded_with_labels(ground_truth_csv):
    classifier = knn(ground_truth_csv)
    assert len(classifier.labels) == 5
    assert ("national_id", "TN", "front") in classifier.labels


def test_nearest_reference_gives_the_label(ground_truth_csv, monkeypatch):
    classifier = knn(ground_truth_csv, exclude_self_above=1.01)   # nothing excluded: the image is its own neighbour
    prediction = classifier.classify(Image.new("RGB", (32, 24), (250, 5, 5)))
    assert (prediction.document_type, prediction.issuing_country, prediction.document_side) == ("national_id", "TN", "front")
    assert prediction.status == "ok"
    assert prediction.confidence == 1.0                              # the only neighbour with any similarity
    assert json.loads(prediction.raw_output)["best_similarity"] == 1.0


def test_leave_one_out_the_image_itself_is_skipped(ground_truth_csv):
    """With the default exclude_self_above, a query identical to a reference cannot match itself; here no
    other reference looks like it, so the answer is 'unknown' (routing -> review)."""
    prediction = knn(ground_truth_csv).classify(Image.new("RGB", (32, 24), (255, 0, 0)))
    assert (prediction.document_type, prediction.issuing_country, prediction.document_side) == ("other", "unknown", "unknown")


def test_unfamiliar_crop_is_unknown(ground_truth_csv):
    prediction = knn(ground_truth_csv, exclude_self_above=1.01).classify(Image.new("RGB", (32, 24), (128, 64, 200)))
    assert (prediction.document_type, prediction.issuing_country) == ("other", "unknown")
    assert prediction.confidence == 0.0


def test_votes_are_weighted_and_confidence_is_the_share(ground_truth_csv, tmp_path):
    """Two TN-front references and one MA-front reference all look the same: TN wins with 2/3 of the votes."""
    rows = ["transaction_id,image_id,image_path,document_type,issuing_country,document_side"]
    for txn, (country) in ((1, "TN"), (2, "TN"), (3, "MA")):
        path = tmp_path / f"{txn}.png"
        Image.new("RGB", (32, 24), (255, 0, 0)).save(path)
        rows.append(f"{txn},front,{path},national_id,{country},front")
    csv_path = tmp_path / "refs.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    prediction = knn(csv_path, exclude_self_above=1.01, k=3).classify(Image.new("RGB", (32, 24), (255, 0, 0)))
    assert prediction.issuing_country == "TN"
    assert abs(prediction.confidence - 2 / 3) < 1e-3


def test_knn_is_registered_for_yaml_configs():
    assert REGISTRY["classifier"]["knn"] == "id_classifier.classifiers.knn:EmbeddingKnnClassifier"
