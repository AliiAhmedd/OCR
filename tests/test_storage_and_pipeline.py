from sqlalchemy import select, update

from fakes import ColorClassifier
from id_classifier.detectors import FullImageDetector
from id_classifier.pipeline import run_classification
from id_classifier.routing import RoutingConfig
from id_classifier.sources import LocalFolderSource
from id_classifier.storage import Storage, images, predictions, reviews
from id_classifier.types import ImageRecord


def make_storage(tmp_path) -> Storage:
    storage = Storage(f"sqlite:///{(tmp_path / 'db' / 'pipeline.db').as_posix()}")  # also checks the folder is created
    storage.create_tables()
    return storage


def run(storage, run_id, folder):
    return run_classification(
        LocalFolderSource(folder), FullImageDetector(), [ColorClassifier()], RoutingConfig(), storage, run_id,
    )


def test_image_row_is_upserted_not_duplicated(tmp_path):
    storage = make_storage(tmp_path)
    storage.save_image(ImageRecord(1, "front", "a.png", fetch_status="failed", failure_reason="read_error: OSError"))
    storage.save_image(ImageRecord(1, "front", "a.png", image_hash="ab" * 32))   # the re-fetch worked
    with storage.engine.connect() as conn:
        rows = conn.execute(select(images.c.fetch_status, images.c.failure_reason)).all()
    assert rows == [("ok", None)]


def test_pipeline_stores_everything_and_reruns_are_safe(tmp_path, images_dir):
    storage = make_storage(tmp_path)
    counts = run(storage, "run-1", images_dir)
    assert counts["images"] == 5                          # 4 documents + 1 without a document
    assert counts["per_status"] == {"auto_accepted": 4, "needs_review": 1}
    assert counts["per_country"] == {"TN": 2, "MA": 2, "NO_DOC": 1}
    assert storage.count_rows("images") == 5
    assert storage.count_rows("regions") == 5
    assert storage.count_rows("predictions") == 10       # 5 classifier rows + 5 router rows
    assert storage.count_rows("reviews") == 5

    run(storage, "run-1", images_dir)                    # same run again: nothing new
    assert storage.count_rows("predictions") == 10
    assert storage.count_rows("images") == 5

    run(storage, "run-2", images_dir)                    # a new run appends history, never overwrites
    assert storage.count_rows("predictions") == 20
    assert storage.count_rows("reviews") == 5


def test_human_review_is_never_overwritten(tmp_path, images_dir):
    storage = make_storage(tmp_path)
    run(storage, "run-1", images_dir)
    with storage.engine.begin() as conn:
        conn.execute(update(reviews).values(review_status="corrected", issuing_country="JO", reviewer="tester"))
    run(storage, "run-2", images_dir)
    with storage.engine.connect() as conn:
        statuses = {row.review_status for row in conn.execute(select(reviews.c.review_status))}
    assert statuses == {"corrected"}


def test_router_rows_explain_review(tmp_path, images_dir):
    storage = make_storage(tmp_path)
    run(storage, "run-1", images_dir)
    with storage.engine.connect() as conn:
        router = conn.execute(
            select(predictions.c.needs_review, predictions.c.review_reasons).where(predictions.c.model_name == "router")
        ).all()
    assert len(router) == 5
    assert all(reasons for needs_review, reasons in router if needs_review)   # every flagged row says why


def test_failed_fetch_is_recorded_without_predictions(tmp_path, images_dir):
    storage = make_storage(tmp_path)
    counts = run_classification(LocalFolderSource(images_dir), FullImageDetector(), [ColorClassifier()],
                                RoutingConfig(), storage, "r", keys=[(1, "front", "missing")])
    assert counts["per_status"] == {"fetch_failed": 1}
    assert storage.count_rows("images") == 1 and storage.count_rows("predictions") == 0
