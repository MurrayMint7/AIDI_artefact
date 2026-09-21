"""Prepare Amazon Reviews 2023 data through one testable interface."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd
import yaml

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only in an incomplete environment
    duckdb = None


LOGGER = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when source data cannot satisfy the declared experiment schema."""


@dataclass(frozen=True)
class CategoryFiles:
    name: str
    reviews_path: Path
    metadata_path: Path
    expected_review_rows: int | None
    expected_metadata_rows: int | None


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved experiment settings and local filesystem locations."""

    config_path: Path
    raw_dir: Path
    output_dir: Path
    settings: Mapping[str, Any]
    categories: tuple[CategoryFiles, ...]

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        raw_dir: str | Path,
        output_dir: str | Path,
    ) -> "PipelineConfig":
        config_path = Path(config_path)
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Experiment configuration must be a YAML mapping")

        resolved_raw_dir = Path(raw_dir)
        categories = tuple(
            CategoryFiles(
                name=category["name"],
                reviews_path=resolved_raw_dir / _source_filename(category, "reviews"),
                metadata_path=resolved_raw_dir / _source_filename(category, "metadata"),
                expected_review_rows=_optional_int(category.get("expected_review_rows")),
                expected_metadata_rows=_optional_int(category.get("expected_metadata_rows")),
            )
            for category in settings.get("data", {}).get("categories", [])
        )
        if not categories:
            raise SchemaError("Experiment configuration declares no data categories")

        return cls(
            config_path=config_path,
            raw_dir=resolved_raw_dir,
            output_dir=Path(output_dir),
            settings=settings,
            categories=categories,
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Observable outputs of a completed preparation run."""

    dataset_path: Path
    data_manifest_path: Path
    data_quality_path: Path
    join_report_path: Path
    split_manifest_path: Path
    row_counts: Mapping[str, int]


def prepare(config: PipelineConfig) -> DatasetBundle:
    """Validate, minimise, deduplicate, join, label and split local source data."""

    mode = config.settings.get("execution", {}).get("preparation_mode", "in_memory")
    if mode == "out_of_core":
        return _prepare_out_of_core(config)
    if mode != "in_memory":
        raise SchemaError(f"Unknown preparation mode: {mode}")
    return _prepare_in_memory(config)


def _prepare_in_memory(config: PipelineConfig) -> DatasetBundle:
    """Prepare small datasets in memory; retained as a simple reference path."""

    output_data_dir = config.output_dir / "processed"
    manifest_dir = config.output_dir / "manifests"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    source_records: list[dict[str, Any]] = []
    review_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    source_review_count = 0

    for category_order, category in enumerate(config.categories):
        review_frame = _read_reviews(category, config.settings, category_order)
        metadata_frame = _read_metadata(category, config.settings)
        _validate_expected_rows(category, len(review_frame), len(metadata_frame))
        source_review_count += len(review_frame)
        review_frames.append(review_frame)
        metadata_frames.append(metadata_frame)
        source_records.extend(
            (
                _source_record(category.name, "reviews", category.reviews_path, len(review_frame)),
                _source_record(
                    category.name,
                    "metadata",
                    category.metadata_path,
                    len(metadata_frame),
                ),
            )
        )

    reviews = pd.concat(review_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)

    reviews, cleaning_counts = _clean_and_label_reviews(reviews, config.settings)
    metadata = _validate_metadata_cardinality(metadata)
    joined, join_report = _join_metadata(reviews, metadata)
    split_dataset, split_report = _split_and_sample(joined, config.settings)

    prohibited = set(config.settings["data"].get("prohibited_curated_fields", []))
    leaked = prohibited.intersection(split_dataset.columns)
    if leaked:
        raise SchemaError(f"Prohibited fields reached curated data: {sorted(leaked)}")

    split_dataset = split_dataset.drop(
        columns=["asin", "_source_order", "_normalised_text"],
        errors="ignore",
    ).reset_index(drop=True)

    dataset_path = output_data_dir / "training_dataset.parquet"
    split_dataset.to_parquet(
        dataset_path,
        index=False,
        **_parquet_write_options(config.settings),
    )

    row_counts = {
        "source_reviews": source_review_count,
        **cleaning_counts,
        "curated_reviews": len(split_dataset),
    }
    config_hash = _sha256(config.config_path)

    data_manifest_path = manifest_dir / "data_manifest.json"
    data_quality_path = manifest_dir / "data_quality.json"
    join_report_path = manifest_dir / "join_report.json"
    split_manifest_path = manifest_dir / "split_manifest.json"

    _write_json(
        data_manifest_path,
        {
            "config_file": config.config_path.name,
            "config_sha256": config_hash,
            "sources": source_records,
            "output": {
                "filename": dataset_path.name,
                "rows": len(split_dataset),
                "sha256": _sha256(dataset_path),
            },
        },
    )
    _write_json(
        data_quality_path,
        {
            "row_counts": row_counts,
            "columns": list(split_dataset.columns),
            "null_counts": {
                column: int(count)
                for column, count in split_dataset.isna().sum().items()
            },
        },
    )
    _write_json(join_report_path, join_report)
    _write_json(split_manifest_path, split_report)

    return DatasetBundle(
        dataset_path=dataset_path,
        data_manifest_path=data_manifest_path,
        data_quality_path=data_quality_path,
        join_report_path=join_report_path,
        split_manifest_path=split_manifest_path,
        row_counts=row_counts,
    )


def _prepare_out_of_core(config: PipelineConfig) -> DatasetBundle:
    """Prepare the full corpus with bounded RAM and a disk-backed query engine."""

    if duckdb is None:
        raise RuntimeError(
            "Out-of-core preparation requires DuckDB; install the project dependencies"
        )

    output_data_dir = config.output_dir / "processed"
    manifest_dir = config.output_dir / "manifests"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    execution = config.settings.get("execution", {})
    chunk_rows = int(execution.get("review_chunk_rows", 100_000))
    if chunk_rows < 1:
        raise SchemaError("execution.review_chunk_rows must be positive")
    memory_limit = str(execution.get("duckdb_memory_limit", "1GB"))
    threads = int(execution.get("duckdb_threads", 1))
    if threads < 1:
        raise SchemaError("execution.duckdb_threads must be positive")

    source_records: list[dict[str, Any]] = []
    source_review_count = 0
    empty_text_removed = 0

    with tempfile.TemporaryDirectory(prefix="prepare-", dir=config.output_dir) as workspace:
        workspace_path = Path(workspace)
        connection = duckdb.connect(str(workspace_path / "preparation.duckdb"))
        try:
            connection.execute("SET memory_limit = ?", [memory_limit])
            connection.execute("SET threads = ?", [threads])
            connection.execute(
                "SET temp_directory = ?", [str(workspace_path / "spill")]
            )
            connection.execute("SET preserve_insertion_order = false")
            _create_staging_tables(connection)

            for category_order, category in enumerate(config.categories):
                LOGGER.info("Staging %s reviews", category.name)
                review_rows, removed = _stage_review_source(
                    connection,
                    category,
                    config.settings,
                    category_order,
                    chunk_rows,
                )
                LOGGER.info(
                    "Staged %s review rows for %s (%s empty removed)",
                    f"{review_rows:,}",
                    category.name,
                    f"{removed:,}",
                )
                LOGGER.info("Staging %s metadata", category.name)
                metadata_rows = _stage_metadata_source(
                    connection,
                    category,
                    config.settings,
                    chunk_rows,
                )
                LOGGER.info(
                    "Staged %s metadata rows for %s",
                    f"{metadata_rows:,}",
                    category.name,
                )
                _validate_expected_rows(category, review_rows, metadata_rows)
                source_review_count += review_rows
                empty_text_removed += removed
                source_records.extend(
                    (
                        _source_record(
                            category.name, "reviews", category.reviews_path, review_rows
                        ),
                        _source_record(
                            category.name,
                            "metadata",
                            category.metadata_path,
                            metadata_rows,
                        ),
                    )
                )

            _validate_staged_metadata(connection)
            LOGGER.info("Deduplicating staged reviews")
            cleaning_counts = _deduplicate_staged_reviews(connection)
            LOGGER.info("Joining product metadata")
            join_report = _join_staged_metadata(connection)
            LOGGER.info("Applying chronological splits and deterministic caps")
            split_dataset, split_report = _split_and_sample_staged(
                connection, config.settings
            )
        finally:
            connection.close()

    prohibited = set(config.settings["data"].get("prohibited_curated_fields", []))
    leaked = prohibited.intersection(split_dataset.columns)
    if leaked:
        raise SchemaError(f"Prohibited fields reached curated data: {sorted(leaked)}")
    if "asin" in split_dataset.columns:
        raise SchemaError("Diagnostic asin field reached the curated output")

    dataset_path = output_data_dir / "training_dataset.parquet"
    split_dataset.to_parquet(
        dataset_path,
        index=False,
        **_parquet_write_options(config.settings),
    )

    row_counts = {
        "source_reviews": source_review_count,
        "empty_text_removed": empty_text_removed,
        **cleaning_counts,
        "curated_reviews": len(split_dataset),
    }
    data_manifest_path = manifest_dir / "data_manifest.json"
    data_quality_path = manifest_dir / "data_quality.json"
    join_report_path = manifest_dir / "join_report.json"
    split_manifest_path = manifest_dir / "split_manifest.json"

    _write_json(
        data_manifest_path,
        {
            "config_file": config.config_path.name,
            "config_sha256": _sha256(config.config_path),
            "sources": source_records,
            "output": {
                "filename": dataset_path.name,
                "rows": len(split_dataset),
                "sha256": _sha256(dataset_path),
            },
        },
    )
    _write_json(
        data_quality_path,
        {
            "row_counts": row_counts,
            "columns": list(split_dataset.columns),
            "null_counts": {
                column: int(count)
                for column, count in split_dataset.isna().sum().items()
            },
        },
    )
    _write_json(join_report_path, join_report)
    _write_json(split_manifest_path, split_report)

    return DatasetBundle(
        dataset_path=dataset_path,
        data_manifest_path=data_manifest_path,
        data_quality_path=data_quality_path,
        join_report_path=join_report_path,
        split_manifest_path=split_manifest_path,
        row_counts=row_counts,
    )


def _create_staging_tables(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE reviews_raw (
            rating TINYINT,
            text VARCHAR,
            title VARCHAR,
            timestamp TIMESTAMPTZ,
            verified_purchase BOOLEAN,
            helpful_vote BIGINT,
            asin VARCHAR,
            parent_asin VARCHAR,
            source_category VARCHAR,
            _source_order BIGINT,
            _normalised_text VARCHAR,
            normalised_text_sha256 VARCHAR,
            sentiment_id TINYINT,
            sentiment_label VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE metadata_raw (
            parent_asin VARCHAR,
            main_category VARCHAR,
            price DOUBLE,
            rating_number BIGINT
        )
        """
    )


def _json_chunks(path: Path, chunk_rows: int) -> Any:
    if not path.is_file():
        raise SchemaError(f"Required source file does not exist: {path}")
    try:
        yield from pd.read_json(
            path,
            lines=True,
            compression="infer",
            chunksize=chunk_rows,
        )
    except ValueError as exc:
        raise SchemaError(f"Could not parse JSON Lines source {path}: {exc}") from exc


def _rating_to_label(settings: Mapping[str, Any]) -> dict[int, tuple[int, str]]:
    mapping: dict[int, tuple[int, str]] = {}
    for label, label_config in settings["data"]["labels"].items():
        for rating in label_config["ratings"]:
            mapping[int(rating)] = (int(label_config["id"]), label)
    if set(mapping) != {1, 2, 3, 4, 5}:
        raise SchemaError("Rating-to-label mapping must cover each integer rating from 1 to 5")
    return mapping


def _stage_review_source(
    connection: Any,
    category: CategoryFiles,
    settings: Mapping[str, Any],
    category_order: int,
    chunk_rows: int,
) -> tuple[int, int]:
    aliases = settings["data"].get("accepted_field_aliases", {})
    required = set(settings["data"]["canonical_review_fields"])
    label_mapping = _rating_to_label(settings)
    source_rows = 0
    empty_rows = 0

    for chunk_number, frame in enumerate(
        _json_chunks(category.reviews_path, chunk_rows), start=1
    ):
        source_rows += len(frame)
        for alias, canonical in aliases.items():
            if canonical not in frame.columns and alias in frame.columns:
                frame = frame.rename(columns={alias: canonical})
        missing = required.difference(frame.columns)
        if missing:
            raise SchemaError(
                f"Review source {category.reviews_path.name} is missing required fields: "
                f"{sorted(missing)}"
            )

        selected = frame.loc[:, sorted(required)].copy()
        ratings = pd.to_numeric(selected["rating"], errors="coerce")
        valid_ratings = ratings.notna() & ratings.between(1, 5) & (ratings % 1 == 0)
        if not valid_ratings.all():
            raise SchemaError(
                f"Found {int((~valid_ratings).sum())} invalid or non-integral ratings"
            )
        selected["rating"] = ratings.astype("int8")

        if pd.api.types.is_datetime64_any_dtype(selected["timestamp"]):
            timestamps = pd.to_datetime(selected["timestamp"], utc=True, errors="coerce")
        else:
            timestamps = pd.to_datetime(
                pd.to_numeric(selected["timestamp"], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
        if timestamps.isna().any():
            raise SchemaError(f"Found {int(timestamps.isna().sum())} invalid timestamps")
        selected["timestamp"] = timestamps

        selected["text"] = selected["text"].astype("string")
        non_empty = selected["text"].notna() & selected["text"].str.strip().ne("")
        empty_rows += int((~non_empty).sum())
        selected = selected.loc[non_empty].copy()
        selected["text"] = selected["text"].str.strip()
        selected["title"] = selected["title"].astype("string")
        selected["asin"] = selected["asin"].astype("string")
        selected["parent_asin"] = selected["parent_asin"].astype("string")
        selected["helpful_vote"] = pd.to_numeric(
            selected["helpful_vote"], errors="coerce"
        ).astype("Int64")
        selected["verified_purchase"] = selected["verified_purchase"].astype(
            "boolean"
        )
        selected["source_category"] = category.name
        selected["_source_order"] = (
            category_order * 1_000_000_000 + selected.index
        ).astype("int64")
        selected["sentiment_id"] = selected["rating"].map(
            lambda rating: label_mapping[int(rating)][0]
        ).astype("int8")
        selected["sentiment_label"] = selected["rating"].map(
            lambda rating: label_mapping[int(rating)][1]
        ).astype("string")
        selected["_normalised_text"] = selected["text"].map(_normalise_text)
        selected["normalised_text_sha256"] = selected["_normalised_text"].map(
            _sha256_text
        )

        staged_columns = [
            "rating",
            "text",
            "title",
            "timestamp",
            "verified_purchase",
            "helpful_vote",
            "asin",
            "parent_asin",
            "source_category",
            "_source_order",
            "_normalised_text",
            "normalised_text_sha256",
            "sentiment_id",
            "sentiment_label",
        ]
        connection.register("review_chunk", selected.loc[:, staged_columns])
        try:
            connection.execute("INSERT INTO reviews_raw SELECT * FROM review_chunk")
        finally:
            connection.unregister("review_chunk")
        if chunk_number % 10 == 0:
            LOGGER.info(
                "Read %s review rows from %s",
                f"{source_rows:,}",
                category.name,
            )

    return source_rows, empty_rows


def _stage_metadata_source(
    connection: Any,
    category: CategoryFiles,
    settings: Mapping[str, Any],
    chunk_rows: int,
) -> int:
    retained = set(settings["data"]["retained_metadata_fields"])
    source_rows = 0
    staged_columns = ["parent_asin", "main_category", "price", "rating_number"]
    if retained != set(staged_columns):
        raise SchemaError(
            "Out-of-core preparation expects retained metadata fields "
            f"{sorted(staged_columns)}"
        )

    for chunk_number, frame in enumerate(
        _json_chunks(category.metadata_path, chunk_rows), start=1
    ):
        source_rows += len(frame)
        missing = retained.difference(frame.columns)
        if missing:
            raise SchemaError(
                f"Metadata source {category.metadata_path.name} is missing required fields: "
                f"{sorted(missing)}"
            )
        selected = frame.loc[:, staged_columns].copy()
        selected["parent_asin"] = selected["parent_asin"].astype("string")
        selected["main_category"] = selected["main_category"].astype("string")
        selected["price"] = pd.to_numeric(selected["price"], errors="coerce").astype(
            "Float64"
        )
        selected["rating_number"] = pd.to_numeric(
            selected["rating_number"], errors="coerce"
        ).astype("Int64")
        connection.register("metadata_chunk", selected)
        try:
            connection.execute("INSERT INTO metadata_raw SELECT * FROM metadata_chunk")
        finally:
            connection.unregister("metadata_chunk")
        if chunk_number % 10 == 0:
            LOGGER.info(
                "Read %s metadata rows from %s",
                f"{source_rows:,}",
                category.name,
            )
    return source_rows


def _validate_staged_metadata(connection: Any) -> None:
    duplicates = connection.execute(
        """
        SELECT parent_asin
        FROM metadata_raw
        GROUP BY parent_asin
        HAVING count(*) > 1
        ORDER BY parent_asin
        LIMIT 5
        """
    ).fetchall()
    if duplicates:
        examples = [str(row[0]) for row in duplicates]
        raise SchemaError(
            "Metadata parent_asin must be unique for a many-to-one join; "
            f"duplicate examples: {examples}"
        )


def _deduplicate_staged_reviews(connection: Any) -> dict[str, int]:
    connection.execute(
        """
        CREATE OR REPLACE TABLE text_hash_stats AS
        SELECT
            normalised_text_sha256,
            count(*) AS occurrences,
            count(DISTINCT sentiment_id) AS label_variants
        FROM reviews_raw
        GROUP BY normalised_text_sha256
        """
    )
    conflicting_rows, duplicate_rows = connection.execute(
        """
        SELECT
            coalesce(sum(CASE WHEN label_variants > 1 THEN occurrences ELSE 0 END), 0),
            coalesce(sum(CASE WHEN label_variants = 1 THEN occurrences - 1 ELSE 0 END), 0)
        FROM text_hash_stats
        """
    ).fetchone()
    connection.execute(
        """
        CREATE OR REPLACE TABLE winning_review_rows AS
        SELECT
            r.normalised_text_sha256,
            arg_min(r._source_order, (r.timestamp, r._source_order)) AS _source_order
        FROM reviews_raw AS r
        JOIN text_hash_stats AS stats USING (normalised_text_sha256)
        WHERE stats.label_variants = 1
        GROUP BY r.normalised_text_sha256
        """
    )
    connection.execute(
        """
        CREATE TABLE curated_reviews AS
        SELECT r.*, r.normalised_text_sha256 AS record_id
        FROM reviews_raw AS r
        JOIN winning_review_rows AS winners
            USING (normalised_text_sha256, _source_order)
        """
    )
    connection.execute("DROP TABLE reviews_raw")
    connection.execute("DROP TABLE text_hash_stats")
    connection.execute("DROP TABLE winning_review_rows")
    return {
        "exact_duplicates_removed": int(duplicate_rows),
        "conflicting_label_rows_removed": int(conflicting_rows),
    }


def _join_staged_metadata(connection: Any) -> dict[str, Any]:
    wrong_key_matches = int(
        connection.execute(
            """
            SELECT count(*)
            FROM curated_reviews AS reviews
            WHERE EXISTS (
                SELECT 1 FROM metadata_raw AS metadata
                WHERE metadata.parent_asin = reviews.asin
            )
            """
        ).fetchone()[0]
    )
    connection.execute(
        """
        CREATE TABLE joined_reviews AS
        SELECT
            reviews.*,
            metadata.main_category,
            metadata.price,
            metadata.rating_number,
            metadata.parent_asin IS NOT NULL AS metadata_matched
        FROM curated_reviews AS reviews
        LEFT JOIN metadata_raw AS metadata USING (parent_asin)
        """
    )
    reviews, matches = connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE metadata_matched) FROM joined_reviews"
    ).fetchone()
    connection.execute("DROP TABLE curated_reviews")
    connection.execute("DROP TABLE metadata_raw")
    reviews = int(reviews)
    matches = int(matches)
    return {
        "reviews": reviews,
        "correct_key": "parent_asin",
        "correct_key_matches": matches,
        "correct_key_unmatched": reviews - matches,
        "correct_key_match_rate": matches / reviews if reviews else 0.0,
        "diagnostic_wrong_key": "asin",
        "diagnostic_wrong_key_matches": wrong_key_matches,
        "diagnostic_wrong_key_match_rate": wrong_key_matches / reviews if reviews else 0.0,
    }


def _split_and_sample_staged(
    connection: Any,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_config = settings["split"]
    preferred = split_config["preferred"]
    train_end = pd.Timestamp(preferred["train_end"])
    validation_start = pd.Timestamp(preferred["validation_start"])
    validation_end = pd.Timestamp(preferred["validation_end"])
    test_start = pd.Timestamp(preferred["test_start"])
    if not train_end < validation_start <= validation_end < test_start:
        raise SchemaError("Chronological split boundaries overlap or are out of order")

    unassigned = int(
        connection.execute(
            """
            SELECT count(*) FROM joined_reviews
            WHERE NOT (
                timestamp <= ? OR timestamp BETWEEN ? AND ? OR timestamp >= ?
            )
            """,
            [train_end, validation_start, validation_end, test_start],
        ).fetchone()[0]
    )
    if unassigned:
        raise SchemaError(f"Chronological rules left {unassigned} rows unassigned")

    preferred_ok = _staged_preferred_split_meets_gate(
        connection,
        settings,
        train_end,
        validation_start,
        validation_end,
        test_start,
    )
    if preferred_ok:
        connection.execute(
            """
            CREATE TABLE split_base AS
            SELECT *, CASE
                WHEN timestamp <= ? THEN 'train'
                WHEN timestamp BETWEEN ? AND ? THEN 'validation'
                WHEN timestamp >= ? THEN 'test'
            END AS base_split
            FROM joined_reviews
            """,
            [train_end, validation_start, validation_end, test_start],
        )
        strategy = "preferred_chronological"
        boundaries: Mapping[str, Any] = preferred
    else:
        too_small = connection.execute(
            """
            SELECT source_category, count(*) AS rows
            FROM joined_reviews
            GROUP BY source_category
            HAVING count(*) < 3
            ORDER BY source_category
            LIMIT 1
            """
        ).fetchone()
        if too_small:
            raise SchemaError(
                "Fallback chronological split needs at least three curated reviews per "
                f"category; {too_small[0]} has {too_small[1]}"
            )
        connection.execute(
            """
            CREATE TABLE split_base AS
            WITH ranked AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY source_category
                        ORDER BY timestamp, _source_order
                    ) AS chronological_rank,
                    count(*) OVER (PARTITION BY source_category) AS category_rows
                FROM joined_reviews
            )
            SELECT * EXCLUDE (chronological_rank, category_rows), CASE
                WHEN chronological_rank <= category_rows
                    - greatest(1, floor(category_rows * 0.1))
                    - greatest(1, floor(category_rows * 0.1)) THEN 'train'
                WHEN chronological_rank <= category_rows
                    - greatest(1, floor(category_rows * 0.1)) THEN 'validation'
                ELSE 'test'
            END AS base_split
            FROM ranked
            """
        )
        strategy = "per_category_chronological_80_10_10"
        boundaries = {"train": 0.8, "validation": 0.1, "test": 0.1}

    seed = int(settings.get("project", {}).get("seed", 42))
    caps = split_config["caps"]
    training_cap = int(caps["training_total"])
    validation_cap = int(caps["validation_total"])
    test_cap = int(caps["test_total"])
    if min(training_cap, validation_cap, test_cap) < 1:
        raise SchemaError("All split caps must be positive")

    pre_sampling_counts = {
        str(split): int(count)
        for split, count in connection.execute(
            "SELECT base_split, count(*) FROM split_base GROUP BY base_split"
        ).fetchall()
    }
    training_class_counts = [
        {
            "source_category": str(category),
            "sentiment_label": str(label),
            "rows": int(count),
        }
        for category, label, count in connection.execute(
            """
            SELECT source_category, sentiment_label, count(*)
            FROM split_base
            WHERE base_split = 'train'
            GROUP BY source_category, sentiment_label
            ORDER BY source_category, sentiment_label
            """
        ).fetchall()
    ]

    if split_config.get("balance_training_across_category_class_cells", False):
        category_count = len(settings["data"]["categories"])
        label_count = len(settings["data"]["labels"])
        per_cell_cap = training_cap // (category_count * label_count)
        if per_cell_cap < 1:
            raise SchemaError("Training cap is too small for category/class balancing")
        connection.execute(
            """
            CREATE TABLE sampled_train AS
            SELECT * EXCLUDE (sample_rank) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY source_category, sentiment_label
                    ORDER BY md5(record_id || ?)
                ) AS sample_rank
                FROM split_base WHERE base_split = 'train'
            ) WHERE sample_rank <= ?
            """,
            [f":{seed}", per_cell_cap],
        )
    else:
        connection.execute(
            """
            CREATE TABLE sampled_train AS
            SELECT * EXCLUDE (sample_rank) FROM (
                SELECT *, row_number() OVER (ORDER BY md5(record_id || ?)) AS sample_rank
                FROM split_base WHERE base_split = 'train'
            ) WHERE sample_rank <= ?
            """,
            [f":{seed}", training_cap],
        )

    connection.execute(
        """
        CREATE TABLE sampled_validation AS
        WITH capped AS (
            SELECT * EXCLUDE (sample_rank) FROM (
                SELECT *, row_number() OVER (ORDER BY md5(record_id || ?)) AS sample_rank
                FROM split_base WHERE base_split = 'validation'
            ) WHERE sample_rank <= ?
        ), ordered AS (
            SELECT *,
                row_number() OVER (ORDER BY timestamp, _source_order) AS validation_rank,
                count(*) OVER () AS validation_rows
            FROM capped
        )
        SELECT * EXCLUDE (validation_rank, validation_rows), CASE
            WHEN validation_rank <= ceil(validation_rows / 2.0)
                THEN 'validation_model_selection'
            ELSE 'validation_policy_calibration'
        END AS validation_split
        FROM ordered
        """,
        [f":{seed}", validation_cap],
    )
    connection.execute(
        """
        CREATE TABLE sampled_test AS
        SELECT * EXCLUDE (sample_rank) FROM (
            SELECT *, row_number() OVER (ORDER BY md5(record_id || ?)) AS sample_rank
            FROM split_base WHERE base_split = 'test'
        ) WHERE sample_rank <= ?
        """,
        [f":{seed}", test_cap],
    )
    connection.execute(
        """
        CREATE TABLE final_dataset AS
        SELECT * EXCLUDE (asin, _source_order, _normalised_text, base_split),
            'train' AS split
        FROM sampled_train
        UNION ALL BY NAME
        SELECT * EXCLUDE (
            asin, _source_order, _normalised_text, base_split, validation_split
        ), validation_split AS split
        FROM sampled_validation
        UNION ALL BY NAME
        SELECT * EXCLUDE (asin, _source_order, _normalised_text, base_split),
            'test' AS split
        FROM sampled_test
        """
    )
    duplicate_splits = int(
        connection.execute(
            """
            SELECT count(*) FROM (
                SELECT normalised_text_sha256
                FROM final_dataset
                GROUP BY normalised_text_sha256
                HAVING count(DISTINCT split) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_splits:
        raise SchemaError("A normalised review hash occurs in more than one split")

    sampled = connection.execute(
        "SELECT * FROM final_dataset ORDER BY timestamp, record_id"
    ).fetchdf()
    counts = {
        str(split): int(count)
        for split, count in connection.execute(
            "SELECT split, count(*) FROM final_dataset GROUP BY split"
        ).fetchall()
    }
    record_ids_by_split: dict[str, list[str]] = {}
    for split, record_id in connection.execute(
        "SELECT split, record_id FROM final_dataset ORDER BY split, record_id"
    ).fetchall():
        record_ids_by_split.setdefault(str(split), []).append(str(record_id))

    return sampled, {
        "strategy": strategy,
        "boundaries": boundaries,
        "seed": seed,
        "pre_sampling_counts": pre_sampling_counts,
        "pre_sampling_training_class_counts": training_class_counts,
        "counts": counts,
        "record_ids_by_split": record_ids_by_split,
    }


def _staged_preferred_split_meets_gate(
    connection: Any,
    settings: Mapping[str, Any],
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> bool:
    del train_end  # Included in the signature to keep all declared boundaries together.
    minimums = settings["split"].get("minimum_rows_per_category_class", {})
    categories = [str(category["name"]) for category in settings["data"]["categories"]]
    labels = [str(label) for label in settings["data"]["labels"]]
    for split_name, start, end in (
        ("validation", validation_start, validation_end),
        ("test", test_start, None),
    ):
        minimum = int(minimums.get(split_name, 0))
        if minimum == 0:
            continue
        if end is None:
            rows = connection.execute(
                """
                SELECT source_category, sentiment_label, count(*)
                FROM joined_reviews WHERE timestamp >= ?
                GROUP BY source_category, sentiment_label
                """,
                [start],
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT source_category, sentiment_label, count(*)
                FROM joined_reviews WHERE timestamp BETWEEN ? AND ?
                GROUP BY source_category, sentiment_label
                """,
                [start, end],
            ).fetchall()
        counts = {(str(category), str(label)): int(count) for category, label, count in rows}
        if any(
            counts.get((category, label), 0) < minimum
            for category in categories
            for label in labels
        ):
            return False
    return True


def _source_filename(category: Mapping[str, Any], kind: str) -> str:
    file_key = f"{kind}_file"
    url_key = f"{kind}_url"
    if file_key in category:
        return str(category[file_key])
    if url_key in category:
        filename = Path(urlparse(str(category[url_key])).path).name
        if filename:
            return filename
    raise SchemaError(f"Category {category.get('name', '<unknown>')} has no {kind} source")


def _parquet_write_options(settings: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_settings = settings.get("benchmark", {})
    selected = benchmark_settings.get("selected_output", {})
    codec = str(selected.get("codec", benchmark_settings.get("output_codec", "snappy")))
    if codec not in {"snappy", "zstd"}:
        raise SchemaError(f"Unsupported production Parquet codec: {codec}")
    options: dict[str, Any] = {"compression": codec}
    row_group_size = selected.get("row_group_size")
    if row_group_size is not None:
        row_group_size = int(row_group_size)
        if row_group_size < 1:
            raise SchemaError("Production Parquet row-group size must be positive")
        options["row_group_size"] = row_group_size
    return options


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _validate_expected_rows(
    category: CategoryFiles,
    review_rows: int,
    metadata_rows: int,
) -> None:
    if category.expected_review_rows is not None and review_rows != category.expected_review_rows:
        raise SchemaError(
            f"{category.name} expected {category.expected_review_rows} review rows "
            f"but found {review_rows}"
        )
    if (
        category.expected_metadata_rows is not None
        and metadata_rows != category.expected_metadata_rows
    ):
        raise SchemaError(
            f"{category.name} expected {category.expected_metadata_rows} metadata rows "
            f"but found {metadata_rows}"
        )


def _read_json_lines(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SchemaError(f"Required source file does not exist: {path}")
    try:
        return pd.read_json(path, lines=True, compression="infer")
    except ValueError as exc:
        raise SchemaError(f"Could not parse JSON Lines source {path}: {exc}") from exc


def _read_reviews(
    category: CategoryFiles,
    settings: Mapping[str, Any],
    category_order: int,
) -> pd.DataFrame:
    frame = _read_json_lines(category.reviews_path)
    aliases = settings["data"].get("accepted_field_aliases", {})
    for alias, canonical in aliases.items():
        if canonical not in frame.columns and alias in frame.columns:
            frame = frame.rename(columns={alias: canonical})

    required = set(settings["data"]["canonical_review_fields"])
    missing = required.difference(frame.columns)
    if missing:
        raise SchemaError(
            f"Review source {category.reviews_path.name} is missing required fields: "
            f"{sorted(missing)}"
        )

    selected = frame.loc[:, sorted(required)].copy()
    selected["source_category"] = category.name
    selected["_source_order"] = [
        category_order * 1_000_000_000 + row_number
        for row_number in range(len(selected))
    ]
    return selected


def _read_metadata(
    category: CategoryFiles,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    frame = _read_json_lines(category.metadata_path)
    retained = set(settings["data"]["retained_metadata_fields"])
    missing = retained.difference(frame.columns)
    if missing:
        raise SchemaError(
            f"Metadata source {category.metadata_path.name} is missing required fields: "
            f"{sorted(missing)}"
        )
    return frame.loc[:, sorted(retained)].copy()


def _clean_and_label_reviews(
    reviews: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    ratings = pd.to_numeric(reviews["rating"], errors="coerce")
    valid_ratings = ratings.notna() & ratings.between(1, 5) & (ratings % 1 == 0)
    if not valid_ratings.all():
        invalid_rows = int((~valid_ratings).sum())
        raise SchemaError(f"Found {invalid_rows} invalid or non-integral ratings")
    reviews["rating"] = ratings.astype("int8")

    if pd.api.types.is_datetime64_any_dtype(reviews["timestamp"]):
        parsed_timestamps = pd.to_datetime(reviews["timestamp"], utc=True, errors="coerce")
    else:
        numeric_timestamps = pd.to_numeric(reviews["timestamp"], errors="coerce")
        parsed_timestamps = pd.to_datetime(
            numeric_timestamps,
            unit="ms",
            utc=True,
            errors="coerce",
        )
    if parsed_timestamps.isna().any():
        raise SchemaError(f"Found {int(parsed_timestamps.isna().sum())} invalid timestamps")
    reviews["timestamp"] = parsed_timestamps

    reviews["text"] = reviews["text"].astype("string")
    non_empty = reviews["text"].notna() & reviews["text"].str.strip().ne("")
    empty_text_removed = int((~non_empty).sum())
    reviews = reviews.loc[non_empty].copy()
    reviews["text"] = reviews["text"].str.strip()

    rating_to_label: dict[int, tuple[int, str]] = {}
    for label, label_config in settings["data"]["labels"].items():
        for rating in label_config["ratings"]:
            rating_to_label[int(rating)] = (int(label_config["id"]), label)
    if set(rating_to_label) != {1, 2, 3, 4, 5}:
        raise SchemaError("Rating-to-label mapping must cover each integer rating from 1 to 5")

    reviews["sentiment_id"] = reviews["rating"].map(
        lambda rating: rating_to_label[int(rating)][0]
    ).astype("int8")
    reviews["sentiment_label"] = reviews["rating"].map(
        lambda rating: rating_to_label[int(rating)][1]
    ).astype("string")

    reviews["_normalised_text"] = reviews["text"].map(_normalise_text)
    reviews["normalised_text_sha256"] = reviews["_normalised_text"].map(_sha256_text)

    label_counts_by_hash = reviews.groupby("normalised_text_sha256")[
        "sentiment_id"
    ].nunique()
    conflicting_hashes = set(label_counts_by_hash[label_counts_by_hash > 1].index)
    conflicting_mask = reviews["normalised_text_sha256"].isin(conflicting_hashes)
    conflicting_label_rows_removed = int(conflicting_mask.sum())
    reviews = reviews.loc[~conflicting_mask].copy()

    before_deduplication = len(reviews)
    reviews = reviews.sort_values(["timestamp", "_source_order"]).drop_duplicates(
        subset="normalised_text_sha256",
        keep="first",
    )
    exact_duplicates_removed = before_deduplication - len(reviews)
    reviews["record_id"] = reviews["normalised_text_sha256"]
    return reviews, {
        "empty_text_removed": empty_text_removed,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_label_rows_removed": conflicting_label_rows_removed,
    }


def _validate_metadata_cardinality(metadata: pd.DataFrame) -> pd.DataFrame:
    duplicate_keys = metadata[metadata["parent_asin"].duplicated(keep=False)]["parent_asin"]
    if not duplicate_keys.empty:
        examples = sorted(str(value) for value in duplicate_keys.dropna().unique())[:5]
        raise SchemaError(
            "Metadata parent_asin must be unique for a many-to-one join; "
            f"duplicate examples: {examples}"
        )
    return metadata


def _join_metadata(
    reviews: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_keys = set(metadata["parent_asin"].dropna().astype(str))
    wrong_key_matches = int(reviews["asin"].astype(str).isin(metadata_keys).sum())

    joined = reviews.merge(
        metadata,
        how="left",
        on="parent_asin",
        validate="many_to_one",
        indicator=True,
    )
    joined["metadata_matched"] = joined["_merge"].eq("both")
    joined = joined.drop(columns="_merge")
    correct_key_matches = int(joined["metadata_matched"].sum())
    return joined, {
        "reviews": len(joined),
        "correct_key": "parent_asin",
        "correct_key_matches": correct_key_matches,
        "correct_key_unmatched": len(joined) - correct_key_matches,
        "correct_key_match_rate": correct_key_matches / len(joined) if len(joined) else 0.0,
        "diagnostic_wrong_key": "asin",
        "diagnostic_wrong_key_matches": wrong_key_matches,
        "diagnostic_wrong_key_match_rate": wrong_key_matches / len(joined) if len(joined) else 0.0,
    }


def _split_and_sample(
    dataset: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_config = settings["split"]
    preferred = split_config["preferred"]
    train_end = pd.Timestamp(preferred["train_end"])
    validation_start = pd.Timestamp(preferred["validation_start"])
    validation_end = pd.Timestamp(preferred["validation_end"])
    test_start = pd.Timestamp(preferred["test_start"])

    if not train_end < validation_start <= validation_end < test_start:
        raise SchemaError("Chronological split boundaries overlap or are out of order")

    dataset = dataset.copy()
    preferred_split = pd.Series(pd.NA, index=dataset.index, dtype="string")
    preferred_split.loc[dataset["timestamp"] <= train_end] = "train"
    preferred_split.loc[
        dataset["timestamp"].between(validation_start, validation_end)
    ] = "validation"
    preferred_split.loc[dataset["timestamp"] >= test_start] = "test"
    if preferred_split.isna().any():
        raise SchemaError(
            f"Chronological rules left {int(preferred_split.isna().sum())} rows unassigned"
        )

    if _preferred_split_meets_gate(dataset, preferred_split, split_config):
        dataset["split"] = preferred_split
        strategy = "preferred_chronological"
        boundaries: Mapping[str, Any] = preferred
    else:
        dataset["split"] = _per_category_chronological_split(dataset)
        strategy = "per_category_chronological_80_10_10"
        boundaries = {"train": 0.8, "validation": 0.1, "test": 0.1}

    validation_mask = dataset["split"].eq("validation")
    validation_indices = list(dataset.loc[validation_mask].sort_values("timestamp").index)
    validation_midpoint = (len(validation_indices) + 1) // 2
    dataset.loc[
        validation_indices[:validation_midpoint], "split"
    ] = "validation_model_selection"
    dataset.loc[
        validation_indices[validation_midpoint:], "split"
    ] = "validation_policy_calibration"
    seed = int(settings.get("project", {}).get("seed", 42))
    caps = split_config["caps"]
    sampled_parts: list[pd.DataFrame] = []
    for split_name, cap_name in (
        ("train", "training_total"),
        ("validation_model_selection", "validation_total"),
        ("validation_policy_calibration", "validation_total"),
        ("test", "test_total"),
    ):
        part = dataset.loc[dataset["split"] == split_name]
        configured_cap = int(caps[cap_name])
        if split_name.startswith("validation_"):
            configured_cap = (configured_cap + 1) // 2
        if len(part) > configured_cap:
            part = part.sample(n=configured_cap, random_state=seed)
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, ignore_index=True).sort_values(
        ["timestamp", "_source_order"]
    )
    duplicate_splits = sampled.groupby("normalised_text_sha256")["split"].nunique()
    if (duplicate_splits > 1).any():
        raise SchemaError("A normalised review hash occurs in more than one split")

    counts = {str(key): int(value) for key, value in sampled["split"].value_counts().items()}
    return sampled, {
        "strategy": strategy,
        "boundaries": boundaries,
        "seed": seed,
        "counts": counts,
        "record_ids_by_split": {
            split: sorted(group["record_id"].astype(str).tolist())
            for split, group in sampled.groupby("split")
        },
    }


def _preferred_split_meets_gate(
    dataset: pd.DataFrame,
    preferred_split: pd.Series,
    split_config: Mapping[str, Any],
) -> bool:
    minimums = split_config.get("minimum_rows_per_category_class", {})
    categories = sorted(dataset["source_category"].dropna().unique())
    labels = sorted(dataset["sentiment_label"].dropna().unique())
    for split_name in ("validation", "test"):
        minimum = int(minimums.get(split_name, 0))
        if minimum == 0:
            continue
        counts = (
            dataset.loc[preferred_split.eq(split_name)]
            .groupby(["source_category", "sentiment_label"], observed=True)
            .size()
        )
        for category in categories:
            for label in labels:
                if int(counts.get((category, label), 0)) < minimum:
                    return False
    return True


def _per_category_chronological_split(dataset: pd.DataFrame) -> pd.Series:
    split = pd.Series(pd.NA, index=dataset.index, dtype="string")
    for category, category_rows in dataset.groupby("source_category", sort=True):
        ordered_indices = list(category_rows.sort_values(["timestamp", "_source_order"]).index)
        row_count = len(ordered_indices)
        if row_count < 3:
            raise SchemaError(
                "Fallback chronological split needs at least three curated reviews per "
                f"category; {category} has {row_count}"
            )
        validation_count = max(1, int(row_count * 0.1))
        test_count = max(1, int(row_count * 0.1))
        training_count = row_count - validation_count - test_count
        if training_count < 1:
            raise SchemaError(f"Fallback split leaves no training rows for {category}")
        split.loc[ordered_indices[:training_count]] = "train"
        split.loc[
            ordered_indices[training_count : training_count + validation_count]
        ] = "validation"
        split.loc[ordered_indices[training_count + validation_count :]] = "test"
    return split


def _normalise_text(value: str) -> str:
    unicode_normalised = unicodedata.normalize("NFKC", value)
    return " ".join(unicode_normalised.split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(category: str, kind: str, path: Path, rows: int) -> dict[str, Any]:
    return {
        "category": category,
        "kind": kind,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": rows,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
