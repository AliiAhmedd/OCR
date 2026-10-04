"""Storage: the pipeline's own tables, via SQLAlchemy Core.

The same code runs on SQLite (local demo, tests: database_url "sqlite:///data/pipeline.db") and on
Postgres (Airflow: the engine comes from the pipeline_postgres connection). Credentials never live here.

Tables
- images:      one row per (transaction_id, image_id); fetch status is updated on a re-fetch
- regions:     one row per detected document region (bbox), linked to its image
- predictions: APPEND-ONLY history: every classifier answer and every routing decision, per run
- reviews:     the current final label per region (auto-accepted, or waiting for / done by a human)

Safe reruns: inserts use ON CONFLICT DO NOTHING (same idea as ocr_error_extract.py), so running the
same run_id twice adds nothing, and a new run_id appends a new set of predictions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine, make_url

from id_classifier.types import ImageRecord, Region, RoutedResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


metadata = MetaData()

images = Table(
    "images", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", BigInteger, nullable=False),      # Core's transaction id (ocr_error_4201.transaction_id)
    Column("image_id", String(64), nullable=False),            # e.g. "front", "back"
    Column("image_reference", Text, nullable=False),           # path / blob key, never the image itself
    Column("image_hash", String(64)),                          # sha256 of the bytes (NULL when the fetch failed)
    Column("fetch_status", String(16), nullable=False),        # "ok" | "failed"
    Column("failure_reason", Text),
    Column("ingested_at", DateTime(timezone=True), nullable=False, default=_now),
    UniqueConstraint("transaction_id", "image_id", name="uq_images_transaction_image"),
)

regions = Table(
    "regions", metadata,
    Column("region_id", String(200), primary_key=True),        # "<transaction_id>:<image_id>:<detector>:<n>"
    Column("transaction_id", BigInteger, nullable=False),
    Column("image_id", String(64), nullable=False),
    Column("detector_name", String(100), nullable=False),
    Column("x0", Integer), Column("y0", Integer), Column("x1", Integer), Column("y1", Integer),  # bbox in pixels
    Column("detector_confidence", Float),
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(200), nullable=False),             # Airflow run_id, or "cli-<timestamp>"
    Column("pipeline_version", String(32), nullable=False),    # id_classifier.__version__
    Column("transaction_id", BigInteger, nullable=False),
    Column("image_id", String(64), nullable=False),
    Column("region_id", String(200), nullable=False),
    Column("model_name", String(100), nullable=False),         # classifier name, or "router" for the routing decision
    Column("model_version", String(200), nullable=False),
    Column("document_type", String(32)),
    Column("issuing_country", String(16)),
    Column("document_side", String(16)),
    Column("confidence", Float),
    Column("status", String(16), nullable=False),              # "ok" | "parse_error" | "model_error" (router: "ok")
    Column("needs_review", Boolean),                           # router rows only
    Column("review_reasons", Text),                            # router rows only, comma separated
    Column("raw_output", Text),                                # model's raw answer: stored here, NEVER logged
    Column("error", Text),
    Column("latency_ms", Float),
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
    UniqueConstraint("run_id", "region_id", "model_name", name="uq_predictions_run_region_model"),
)

reviews = Table(
    "reviews", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("region_id", String(200), nullable=False, unique=True),
    Column("transaction_id", BigInteger, nullable=False),
    Column("image_id", String(64), nullable=False),
    Column("document_type", String(32)),                       # final labels (router's suggestion until reviewed)
    Column("issuing_country", String(16)),
    Column("document_side", String(16)),
    Column("review_status", String(16), nullable=False),       # pending | auto_accepted | accepted | corrected | excluded
    Column("reviewer", String(100)),
    Column("reviewed_at", DateTime(timezone=True)),
    Column("exclusion_reason", Text),                          # why a record is left out of the datasets
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
)


class Storage:
    def __init__(self, database_url: str | None = None, engine: Engine | None = None):
        if engine is None:
            if not database_url:
                raise ValueError("Storage needs a database_url or an engine")
            url = make_url(database_url)
            if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)  # create data/ for the SQLite file
            engine = create_engine(database_url)
        self.engine = engine

    def create_tables(self) -> None:
        metadata.create_all(self.engine)  # CREATE TABLE IF NOT EXISTS for all four tables

    def _insert(self, table: Table):
        """INSERT statement that supports ON CONFLICT on both Postgres and SQLite."""
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            raise ValueError(f"Unsupported database: {dialect} (use Postgres or SQLite)")
        return insert(table)

    def save_image(self, record: ImageRecord) -> None:
        """Insert the image row, or refresh its fetch fields if it exists (e.g. a failed fetch that now works)."""
        values = {
            "transaction_id": record.transaction_id,
            "image_id": record.image_id,
            "image_reference": record.image_reference,
            "image_hash": record.image_hash,
            "fetch_status": record.fetch_status,
            "failure_reason": record.failure_reason,
        }
        stmt = self._insert(images).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["transaction_id", "image_id"],
            set_={k: stmt.excluded[k] for k in ("image_reference", "image_hash", "fetch_status", "failure_reason")},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def save_regions(self, found: list[Region]) -> None:
        rows = [
            {
                "region_id": r.region_id, "transaction_id": r.transaction_id, "image_id": r.image_id,
                "detector_name": r.detector_name, "x0": r.bbox[0], "y0": r.bbox[1], "x1": r.bbox[2], "y1": r.bbox[3],
                "detector_confidence": r.confidence, "created_at": _now(),
            }
            for r in found
        ]
        self._insert_ignore(regions, rows, ["region_id"])

    def save_predictions(self, rows: list[dict]) -> None:
        for row in rows:
            row.setdefault("created_at", _now())
        self._insert_ignore(predictions, rows, ["run_id", "region_id", "model_name"])

    def ensure_reviews(self, routed: list[RoutedResult]) -> None:
        """Create a review row per region if there is none yet. An existing row (maybe already reviewed
        by a human) is never touched."""
        rows = [
            {
                "region_id": r.region_id, "transaction_id": r.transaction_id, "image_id": r.image_id,
                "document_type": r.document_type, "issuing_country": r.issuing_country, "document_side": r.document_side,
                "review_status": "pending" if r.needs_review else "auto_accepted", "created_at": _now(),
            }
            for r in routed
        ]
        self._insert_ignore(reviews, rows, ["region_id"])

    def _insert_ignore(self, table: Table, rows: list[dict], conflict_columns: list[str]) -> None:
        if not rows:
            return
        columns = {key for row in rows for key in row}             # a batch insert needs the same keys in every row
        rows = [{key: row.get(key) for key in columns} for row in rows]
        stmt =self._insert(table).on_conflict_do_nothing(index_elements=conflict_columns)
        with self.engine.begin() as conn:  # one transaction: all rows or none
            conn.execute(stmt, rows)

    def count_rows(self, table_name: str) -> int:
        table = metadata.tables[table_name]
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(table)).scalar_one()
