"""Repeatable local storage benchmark for equivalent JSONL and Parquet data."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .data_pipeline import SchemaError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSource:
    name: str
    reviews_path: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark settings and filesystem locations."""

    config_path: Path
    raw_dir: Path
    work_dir: Path
    output_dir: Path
    settings: Mapping[str, Any]
    sources: tuple[BenchmarkSource, ...]

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        raw_dir: str | Path,
        work_dir: str | Path,
        output_dir: str | Path,
    ) -> "BenchmarkConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Benchmark configuration must be a YAML mapping")
        resolved_raw = Path(raw_dir)
        sources = tuple(
            BenchmarkSource(
                name=str(source["name"]),
                reviews_path=resolved_raw / str(source["reviews_file"]),
            )
            for source in settings.get("corpus", {}).get("categories", [])
        )
        if not sources:
            raise SchemaError("Benchmark configuration declares no source categories")
        return cls(
            config_path=resolved_config,
            raw_dir=resolved_raw,
            work_dir=Path(work_dir),
            output_dir=Path(output_dir),
            settings=settings,
            sources=sources,
        )


@dataclass(frozen=True)
class BenchmarkBundle:
    """Aggregate, publishable outputs from a storage benchmark run."""

    results_path: Path
    environment_path: Path
    decision_path: Path
    corpus_manifest_path: Path


def benchmark(config: BenchmarkConfig) -> BenchmarkBundle:
    """Materialise equivalent formats, then measure declared read scenarios."""

    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    corpus, extraction_seconds, source_records = _extract_corpus(config)
    variants, transformation_records = _write_variants(config, corpus)
    corpus_manifest_path = config.work_dir / "benchmark_corpus_manifest.json"
    _write_json(
        corpus_manifest_path,
        {
            "config_sha256": _sha256(config.config_path),
            "logical_rows": len(corpus),
            "rows_by_category": {
                str(category): int(rows)
                for category, rows in corpus["source_category"].value_counts().items()
            },
            "columns": list(corpus.columns),
            "source_files": source_records,
            "variants": [
                {
                    "format": variant["format"],
                    "codec": variant["codec"],
                    "row_group_size": variant["row_group_size"],
                    "filename": variant["path"].name,
                    "bytes": variant["path"].stat().st_size,
                    "sha256": _sha256(variant["path"]),
                }
                for variant in variants
            ],
        },
    )
    del corpus
    gc.collect()

    results = _measure_variants(config, variants)
    results_path = config.output_dir / "storage_benchmark.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    decision_path = config.output_dir / "storage_benchmark_decision.json"
    _write_json(decision_path, _select_production_format(results))

    benchmark_settings = config.settings["benchmark"]
    environment_path = config.output_dir / "storage_benchmark_environment.json"
    _write_json(
        environment_path,
        {
            "config_file": config.config_path.name,
            "config_sha256": _sha256(config.config_path),
            "corpus_manifest_sha256": _sha256(corpus_manifest_path),
            "logical_rows": int(results[0]["corpus_rows"]),
            "methodology": {
                "warmup_runs": int(benchmark_settings["warmup_runs"]),
                "measured_runs": int(benchmark_settings["measured_runs"]),
                "statistic": "median_with_min_max_and_range",
                "cache_policy": (
                    "One warm-up precedes measured repetitions; operating-system caches "
                    "are not manually cleared, so results describe repeated local reads"
                ),
                "transformation_time_excluded_from_read_time": True,
                "equivalent_logical_rows_across_formats": True,
                "corpus_ordering": (
                    "Category blocks follow declared source order; source row order is "
                    "preserved within each category"
                ),
                "predicate_caveat": (
                    "Category predicate results benefit from this clustered layout; a "
                    "differently ordered production file may achieve less row-group pruning"
                ),
            },
            "transformations": {
                "source_extraction_seconds": extraction_seconds,
                "variant_writes": transformation_records,
            },
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor() or "not_reported",
                "logical_cpu_count": os.cpu_count(),
                "memory_bytes": _system_memory_bytes(),
            },
            "results_file": results_path.name,
        },
    )
    return BenchmarkBundle(
        results_path=results_path,
        environment_path=environment_path,
        decision_path=decision_path,
        corpus_manifest_path=corpus_manifest_path,
    )


def _select_production_format(results: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(results)
    parquet = frame.loc[frame["format"].eq("parquet")]
    if parquet.empty:
        raise SchemaError("Storage decision requires at least one Parquet variant")
    ranked = (
        parquet.groupby(["codec", "row_group_size"], as_index=False)
        .agg(
            summed_scenario_medians_seconds=("read_median_seconds", "sum"),
            file_bytes=("file_bytes", "first"),
        )
        .sort_values(
            ["summed_scenario_medians_seconds", "file_bytes", "codec"],
            kind="stable",
        )
    )
    winner = ranked.iloc[0]
    selected = parquet.loc[
        parquet["codec"].eq(winner["codec"])
        & parquet["row_group_size"].eq(winner["row_group_size"])
    ]
    json_rows = frame.loc[frame["format"].eq("jsonl_gzip")]
    if json_rows.empty:
        raise SchemaError("Storage decision requires the JSONL.gz reference")
    json_size = int(json_rows["file_bytes"].iloc[0])
    selected_size = int(winner["file_bytes"])
    json_medians = json_rows.set_index("scenario")["read_median_seconds"]
    speedups = {
        str(row["scenario"]): float(
            json_medians.loc[row["scenario"]] / row["read_median_seconds"]
        )
        for _, row in selected.iterrows()
    }
    return {
        "selection_rule": (
            "Lowest sum of the three measured scenario medians; file bytes break ties"
        ),
        "selected": {
            "format": "parquet",
            "codec": str(winner["codec"]),
            "row_group_size": int(winner["row_group_size"]),
            "file_bytes": selected_size,
            "size_change_vs_jsonl_gzip_percent": (
                (selected_size - json_size) / json_size * 100.0
            ),
            "speedup_vs_jsonl_gzip": speedups,
        },
        "comparison": {
            "jsonl_gzip_file_bytes": json_size,
            "logical_rows": int(frame["corpus_rows"].iloc[0]),
            "candidate_ranking": [
                {
                    "codec": str(row["codec"]),
                    "row_group_size": int(row["row_group_size"]),
                    "summed_scenario_medians_seconds": float(
                        row["summed_scenario_medians_seconds"]
                    ),
                    "file_bytes": int(row["file_bytes"]),
                }
                for _, row in ranked.iterrows()
            ],
        },
        "caveat": (
            "This decision describes repeated local reads of the fixed benchmark corpus; "
            "it is not a universal codec ranking. The category predicate benefits from "
            "category-clustered rows and may show less pruning on another layout"
        ),
    }


def _extract_corpus(
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, float, list[dict[str, Any]]]:
    corpus_settings = config.settings["corpus"]
    rows_per_category = int(corpus_settings["rows_per_category"])
    chunk_rows = int(corpus_settings["chunk_rows"])
    if rows_per_category < 1 or chunk_rows < 1:
        raise SchemaError("Benchmark row and chunk counts must be positive")
    canonical_fields = [str(field) for field in corpus_settings["canonical_fields"]]
    aliases = corpus_settings.get("accepted_field_aliases", {})
    required = set(canonical_fields)
    parts: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for source in config.sources:
        if not source.reviews_path.is_file():
            raise SchemaError(f"Benchmark source does not exist: {source.reviews_path}")
        selected_rows = 0
        for frame in pd.read_json(
            source.reviews_path,
            lines=True,
            compression="infer",
            chunksize=chunk_rows,
        ):
            for alias, canonical in aliases.items():
                if canonical not in frame.columns and alias in frame.columns:
                    frame = frame.rename(columns={alias: canonical})
            missing = required.difference(frame.columns)
            if missing:
                raise SchemaError(
                    f"Benchmark source {source.reviews_path.name} is missing fields: "
                    f"{sorted(missing)}"
                )
            remaining = rows_per_category - selected_rows
            if remaining <= 0:
                break
            selected = frame.loc[:, canonical_fields].head(remaining).copy()
            selected["source_category"] = source.name
            parts.append(selected)
            selected_rows += len(selected)
            if selected_rows >= rows_per_category:
                break
        if selected_rows != rows_per_category:
            raise SchemaError(
                f"Benchmark requested {rows_per_category} rows from {source.name} "
                f"but found {selected_rows}"
            )
        source_records.append(
            {
                "category": source.name,
                "filename": source.reviews_path.name,
                "selected_rows": selected_rows,
                "source_bytes": source.reviews_path.stat().st_size,
                "source_sha256": _sha256(source.reviews_path),
            }
        )
        LOGGER.info("Selected %s benchmark rows from %s", selected_rows, source.name)

    corpus = pd.concat(parts, ignore_index=True)
    corpus = _normalise_corpus_types(corpus)
    return corpus, time.perf_counter() - started, source_records


def _normalise_corpus_types(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus["rating"] = pd.to_numeric(corpus["rating"], errors="raise").astype("float32")
    corpus["timestamp"] = pd.to_numeric(corpus["timestamp"], errors="raise").astype("int64")
    corpus["helpful_vote"] = pd.to_numeric(
        corpus["helpful_vote"], errors="coerce"
    ).astype("Int64")
    corpus["verified_purchase"] = corpus["verified_purchase"].astype("boolean")
    for column in ("text", "title", "asin", "parent_asin", "source_category"):
        corpus[column] = corpus[column].astype("string")
    return corpus


def _write_variants(
    config: BenchmarkConfig,
    corpus: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config.settings["benchmark"]
    variants: list[dict[str, Any]] = []
    transformations: list[dict[str, Any]] = []
    json_path = config.work_dir / "benchmark_corpus.jsonl.gz"
    started = time.perf_counter()
    corpus.to_json(
        json_path,
        orient="records",
        lines=True,
        compression={
            "method": "gzip",
            "compresslevel": int(settings.get("json_gzip_compresslevel", 6)),
            "mtime": 0,
        },
    )
    json_seconds = time.perf_counter() - started
    variants.append(
        {
            "format": "jsonl_gzip",
            "codec": "gzip",
            "row_group_size": 0,
            "path": json_path,
        }
    )
    transformations.append(
        {
            "filename": json_path.name,
            "format": "jsonl_gzip",
            "seconds": json_seconds,
        }
    )

    arrow_table = pa.Table.from_pandas(corpus, preserve_index=False)
    for codec in settings["parquet_codecs"]:
        codec = str(codec).lower()
        if codec not in {"snappy", "zstd"}:
            raise SchemaError(f"Unsupported benchmark Parquet codec: {codec}")
        for row_group_size in settings["parquet_row_group_sizes"]:
            row_group_size = int(row_group_size)
            if row_group_size < 1:
                raise SchemaError("Parquet row-group sizes must be positive")
            parquet_path = (
                config.work_dir
                / f"benchmark_{codec}_rg{row_group_size}.parquet"
            )
            started = time.perf_counter()
            pq.write_table(
                arrow_table,
                parquet_path,
                compression=codec,
                row_group_size=row_group_size,
            )
            seconds = time.perf_counter() - started
            variants.append(
                {
                    "format": "parquet",
                    "codec": codec,
                    "row_group_size": row_group_size,
                    "path": parquet_path,
                }
            )
            transformations.append(
                {
                    "filename": parquet_path.name,
                    "format": "parquet",
                    "codec": codec,
                    "row_group_size": row_group_size,
                    "seconds": seconds,
                }
            )
    return variants, transformations


def _measure_variants(
    config: BenchmarkConfig,
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = config.settings["benchmark"]
    warmups = int(settings["warmup_runs"])
    repetitions = int(settings["measured_runs"])
    if warmups < 0 or repetitions < 1:
        raise SchemaError("Benchmark requires non-negative warmups and measured runs")
    projection = [str(column) for column in settings["projected_columns"]]
    predicate = settings["predicate"]
    if str(predicate["operator"]) != "==":
        raise SchemaError("The benchmark currently supports only equality predicates")
    scenarios = (
        "full_schema",
        "four_column_projection",
        "category_predicate_projection",
    )
    results: list[dict[str, Any]] = []
    expected_rows: dict[str, int] = {}

    for variant in variants:
        for scenario in scenarios:
            for _ in range(warmups):
                _read_scenario(variant, scenario, projection, predicate)
                gc.collect()
            durations: list[float] = []
            rows = 0
            materialised_bytes = 0
            for _ in range(repetitions):
                started = time.perf_counter()
                rows, materialised_bytes = _read_scenario(
                    variant, scenario, projection, predicate
                )
                durations.append(time.perf_counter() - started)
                gc.collect()
            if scenario in expected_rows and expected_rows[scenario] != rows:
                raise SchemaError(
                    f"Benchmark variants returned different rows for {scenario}: "
                    f"{expected_rows[scenario]} and {rows}"
                )
            expected_rows[scenario] = rows
            results.append(
                {
                    "format": variant["format"],
                    "codec": variant["codec"],
                    "row_group_size": variant["row_group_size"],
                    "scenario": scenario,
                    "corpus_rows": expected_rows["full_schema"],
                    "rows": rows,
                    "columns_read": (
                        len(config.settings["corpus"]["canonical_fields"]) + 1
                        if scenario == "full_schema"
                        else len(projection)
                    ),
                    "file_bytes": variant["path"].stat().st_size,
                    "materialised_bytes": materialised_bytes,
                    "warmup_runs": warmups,
                    "measured_runs": repetitions,
                    "read_median_seconds": statistics.median(durations),
                    "read_min_seconds": min(durations),
                    "read_max_seconds": max(durations),
                    "read_range_seconds": max(durations) - min(durations),
                }
            )
            LOGGER.info(
                "%s %s row_group=%s %s median=%.4fs",
                variant["format"],
                variant["codec"],
                variant["row_group_size"],
                scenario,
                statistics.median(durations),
            )
    return results


def _read_scenario(
    variant: Mapping[str, Any],
    scenario: str,
    projection: list[str],
    predicate: Mapping[str, Any],
) -> tuple[int, int]:
    path = Path(variant["path"])
    if variant["format"] == "jsonl_gzip":
        frame = pd.read_json(path, lines=True, compression="gzip")
        if scenario == "four_column_projection":
            frame = frame.loc[:, projection]
        elif scenario == "category_predicate_projection":
            frame = frame.loc[
                frame[str(predicate["column"])].eq(predicate["value"]),
                projection,
            ]
    elif variant["format"] == "parquet":
        if scenario == "full_schema":
            frame = pd.read_parquet(path)
        elif scenario == "four_column_projection":
            frame = pd.read_parquet(path, columns=projection)
        else:
            frame = pd.read_parquet(
                path,
                columns=projection,
                filters=[
                    (
                        str(predicate["column"]),
                        str(predicate["operator"]),
                        predicate["value"],
                    )
                ],
            )
    else:  # defensive invariant
        raise SchemaError(f"Unknown benchmark format: {variant['format']}")
    return len(frame), int(frame.memory_usage(index=True, deep=True).sum())


def _system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
