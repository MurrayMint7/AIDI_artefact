from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from amazon_sentiment.data_pipeline import PipelineConfig, SchemaError, prepare


FIXTURES = Path(__file__).parent / "fixtures"


def _gzip_fixture(source: Path, destination: Path) -> None:
    with source.open("rb") as source_file, gzip.open(destination, "wb") as target_file:
        shutil.copyfileobj(source_file, target_file)


def _raw_fixture_directory(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for category in ("All_Beauty", "Video_Games"):
        _gzip_fixture(
            FIXTURES / f"{category}.reviews.jsonl",
            raw_dir / f"{category}.jsonl.gz",
        )
        _gzip_fixture(
            FIXTURES / f"{category}.metadata.jsonl",
            raw_dir / f"meta_{category}.jsonl.gz",
        )
    return raw_dir


def test_prepare_builds_a_minimised_dataset_and_evidence_bundle(tmp_path: Path) -> None:
    config = PipelineConfig.from_yaml(
        FIXTURES / "experiment.yaml",
        raw_dir=_raw_fixture_directory(tmp_path),
        output_dir=tmp_path / "output",
    )

    bundle = prepare(config)

    dataset = pd.read_parquet(bundle.dataset_path)
    assert len(dataset) == 6
    assert set(dataset["sentiment_label"]) == {"negative", "neutral", "positive"}
    assert dataset["split"].value_counts().to_dict() == {
        "train": 2,
        "validation_model_selection": 1,
        "validation_policy_calibration": 1,
        "test": 2,
    }
    assert dataset["metadata_matched"].value_counts().to_dict() == {True: 5, False: 1}

    prohibited = {"user_id", "images", "videos", "store", "description", "features"}
    assert prohibited.isdisjoint(dataset.columns)
    assert "asin" not in dataset.columns

    assert bundle.row_counts == {
        "source_reviews": 10,
        "empty_text_removed": 1,
        "exact_duplicates_removed": 1,
        "conflicting_label_rows_removed": 2,
        "curated_reviews": 6,
    }
    assert bundle.data_manifest_path.is_file()
    assert bundle.data_quality_path.is_file()
    assert bundle.join_report_path.is_file()
    assert bundle.split_manifest_path.is_file()


def test_prepare_uses_chronological_fallback_when_preferred_cells_are_too_small(
    tmp_path: Path,
) -> None:
    settings = yaml.safe_load((FIXTURES / "experiment.yaml").read_text())
    settings["split"]["minimum_rows_per_category_class"] = {
        "validation": 1,
        "test": 1,
    }
    config_path = tmp_path / "fallback-experiment.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    config = PipelineConfig.from_yaml(
        config_path,
        raw_dir=_raw_fixture_directory(tmp_path),
        output_dir=tmp_path / "output",
    )

    bundle = prepare(config)

    split_manifest = yaml.safe_load(bundle.split_manifest_path.read_text())
    dataset = pd.read_parquet(bundle.dataset_path)
    assert split_manifest["strategy"] == "per_category_chronological_80_10_10"
    assert set(dataset["split"]) == {
        "train",
        "validation_model_selection",
        "validation_policy_calibration",
        "test",
    }
    assert dataset.groupby(["source_category", "split"]).size().to_dict() == {
        ("All_Beauty", "test"): 1,
        ("All_Beauty", "train"): 1,
        ("All_Beauty", "validation_model_selection"): 1,
        ("Video_Games", "test"): 1,
        ("Video_Games", "train"): 1,
        ("Video_Games", "validation_policy_calibration"): 1,
    }


def test_prepare_rejects_a_source_with_an_unexpected_review_count(tmp_path: Path) -> None:
    settings = yaml.safe_load((FIXTURES / "experiment.yaml").read_text())
    settings["data"]["categories"][0]["expected_review_rows"] = 999
    config_path = tmp_path / "wrong-count-experiment.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    config = PipelineConfig.from_yaml(
        config_path,
        raw_dir=_raw_fixture_directory(tmp_path),
        output_dir=tmp_path / "output",
    )

    with pytest.raises(SchemaError, match="expected 999 review rows but found 7"):
        prepare(config)
