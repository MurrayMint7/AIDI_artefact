from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import yaml

from amazon_sentiment.benchmark import BenchmarkConfig, benchmark


FIXTURES = Path(__file__).parent / "fixtures"


def _raw_fixture_directory(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for category in ("All_Beauty", "Video_Games"):
        with (FIXTURES / f"{category}.reviews.jsonl").open("rb") as source:
            with gzip.open(raw_dir / f"{category}.jsonl.gz", "wb") as target:
                shutil.copyfileobj(source, target)
    return raw_dir


def test_benchmark_compares_equivalent_json_and_parquet_corpora(
    tmp_path: Path,
) -> None:
    settings = {
        "project": {"seed": 42},
        "corpus": {
            "rows_per_category": 3,
            "chunk_rows": 2,
            "categories": [
                {"name": "All_Beauty", "reviews_file": "All_Beauty.jsonl.gz"},
                {"name": "Video_Games", "reviews_file": "Video_Games.jsonl.gz"},
            ],
            "canonical_fields": [
                "rating",
                "text",
                "title",
                "timestamp",
                "verified_purchase",
                "helpful_vote",
                "asin",
                "parent_asin",
            ],
            "accepted_field_aliases": {},
        },
        "benchmark": {
            "projected_columns": [
                "rating",
                "text",
                "parent_asin",
                "verified_purchase",
            ],
            "predicate": {
                "column": "source_category",
                "operator": "==",
                "value": "All_Beauty",
            },
            "parquet_codecs": ["snappy"],
            "parquet_row_group_sizes": [2],
            "json_gzip_compresslevel": 6,
            "warmup_runs": 0,
            "measured_runs": 2,
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    bundle = benchmark(
        BenchmarkConfig.from_yaml(
            config_path,
            raw_dir=_raw_fixture_directory(tmp_path),
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "evidence",
        )
    )

    results = pd.read_csv(bundle.results_path)
    assert len(results) == 6
    assert set(results["format"]) == {"jsonl_gzip", "parquet"}
    assert set(results["scenario"]) == {
        "full_schema",
        "four_column_projection",
        "category_predicate_projection",
    }
    assert set(results["measured_runs"]) == {2}
    assert set(results.loc[results["scenario"] != "category_predicate_projection", "rows"]) == {6}
    assert set(results.loc[results["scenario"] == "category_predicate_projection", "rows"]) == {3}
    assert (results["read_median_seconds"] >= 0).all()
    assert bundle.environment_path.is_file()
    assert bundle.decision_path.is_file()
    assert bundle.corpus_manifest_path.is_file()
    decision = yaml.safe_load(bundle.decision_path.read_text())
    assert decision["selected"]["format"] == "parquet"
    assert decision["selected"]["codec"] == "snappy"
    assert decision["selected"]["row_group_size"] == 2
    assert set(decision["selected"]["speedup_vs_jsonl_gzip"]) == {
        "full_schema",
        "four_column_projection",
        "category_predicate_projection",
    }
