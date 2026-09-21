from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from amazon_sentiment.transformer import (
    ThroughputConfig,
    TokenLengthConfig,
    analyse_token_lengths,
)


class WordCountingTokenizer:
    model_max_length = 512
    name_or_path = "fixture-tokenizer"

    def __call__(self, texts: list[str], **_: object) -> dict[str, list[int]]:
        return {"length": [len(text.split()) + 2 for text in texts]}


def test_token_length_pilot_excludes_policy_and_test_rows(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        [
            ("one two", "negative", "All_Beauty", "train"),
            ("one two three four", "neutral", "All_Beauty", "train"),
            ("one", "positive", "Video_Games", "validation_model_selection"),
            ("one two three four five", "positive", "Video_Games", "validation_model_selection"),
            ("policy only token sequence is deliberately very long", "negative", "All_Beauty", "validation_policy_calibration"),
            ("test only token sequence is deliberately very long", "neutral", "Video_Games", "test"),
        ],
        columns=["text", "sentiment_label", "source_category", "split"],
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    config_path = tmp_path / "distilbert.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_id": "fixture/model",
                    "revision": "a" * 40,
                    "num_labels": 3,
                },
                "token_length_pilot": {
                    "included_splits": ["train", "validation_model_selection"],
                    "max_length_candidates": [4, 8],
                    "minimum_token_coverage": 0.75,
                    "batch_rows": 2,
                },
                "throughput_pilot": {
                    "sample_rows": 4,
                    "per_device_batch_size_candidates": [2, 4],
                    "effective_batch_size": 4,
                    "warmup_steps": 1,
                    "measured_steps": 2,
                    "maximum_projected_training_hours": 8.0,
                },
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.00002,
                    "weight_decay": 0.01,
                    "seed": 42,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = analyse_token_lengths(
        TokenLengthConfig.from_yaml(
            config_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "metrics",
        ),
        tokenizer=WordCountingTokenizer(),
    )

    summary = json.loads(bundle.summary_path.read_text())
    slices = pd.read_csv(bundle.slices_path)
    assert summary["rows"] == 4
    assert summary["included_splits"] == ["train", "validation_model_selection"]
    assert summary["test_evaluated"] is False
    assert summary["recommended_max_length"] == 8
    assert summary["overall"]["coverage"]["4"] == 0.5
    assert summary["overall"]["coverage"]["8"] == 1.0
    assert set(slices["slice_type"]) == {
        "overall",
        "sentiment_label",
        "source_category",
        "category_and_label",
    }
    assert "text" not in slices.columns

    throughput = ThroughputConfig.from_yaml(
        config_path,
        dataset_path=dataset_path,
        token_summary_path=bundle.summary_path,
        output_dir=tmp_path / "throughput",
    )
    assert throughput.max_length_candidates == (4, 8)
    assert throughput.per_device_batch_sizes == (2, 4)
    assert throughput.effective_batch_size == 4
