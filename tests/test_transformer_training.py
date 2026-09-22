from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from amazon_sentiment.data_pipeline import SchemaError
from amazon_sentiment.transformer import (
    TransformerBackendResult,
    TransformerTrainingConfig,
    train_transformer,
)


def _training_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "distilbert.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_id": "fixture/model",
                    "revision": "a" * 40,
                    "num_labels": 3,
                },
                "training": {
                    "max_length": 256,
                    "per_device_batch_size": 8,
                    "effective_batch_size": 32,
                    "epochs": 2,
                    "learning_rate": 0.00002,
                    "weight_decay": 0.01,
                    "dynamic_padding": True,
                    "seed": 42,
                    "selection_metric": "macro_f1",
                    "checkpoint_frequency": "each_epoch",
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_training_config_freezes_coverage_choice_and_conservative_batch(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "training_dataset.parquet"
    pd.DataFrame(
        [("train text", 0, "negative", "train")],
        columns=["text", "sentiment_id", "sentiment_label", "split"],
    ).to_parquet(dataset_path, index=False)
    decision_path = tmp_path / "training_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "frozen_without_throughput_pilot",
                "max_length": 256,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
            }
        ),
        encoding="utf-8",
    )

    config = TransformerTrainingConfig.from_yaml(
        _training_config(tmp_path),
        dataset_path=dataset_path,
        decision_path=decision_path,
        output_dir=tmp_path / "model-run",
        cache_dir=tmp_path / "cache",
    )

    assert config.max_length == 256
    assert config.per_device_batch_size == 8
    assert config.gradient_accumulation_steps == 4
    assert config.effective_batch_size == 32
    assert config.included_splits == ("train", "validation_model_selection")


def test_training_config_rejects_a_decision_that_disagrees_with_yaml(
    tmp_path: Path,
) -> None:
    decision_path = tmp_path / "training_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "frozen_without_throughput_pilot",
                "max_length": 128,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SchemaError, match="max_length"):
        TransformerTrainingConfig.from_yaml(
            _training_config(tmp_path),
            dataset_path=tmp_path / "dataset.parquet",
            decision_path=decision_path,
            output_dir=tmp_path / "model-run",
        )


class RecordingBackend:
    def __init__(self) -> None:
        self.seen_text: list[str] = []

    def train(
        self,
        config: TransformerTrainingConfig,
        training: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> TransformerBackendResult:
        self.seen_text = [*training["text"], *validation["text"]]
        model_dir = config.output_dir / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return TransformerBackendResult(
            model_dir=model_dir,
            validation_logits=np.array(
                [[4.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
                dtype=np.float32,
            ),
            validation_label_ids=np.array([0, 1, 2], dtype=np.int64),
            metrics={"eval_macro_f1": 1.0},
            training_seconds=1.25,
            resumed_from_checkpoint=None,
            environment={"backend": "fixture"},
        )


def test_train_transformer_excludes_protected_splits_and_writes_bundle(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "training_dataset.parquet"
    rows = [
        ("negative train", 0, "negative", "All_Beauty", "train"),
        ("neutral train", 1, "neutral", "Video_Games", "train"),
        ("positive train", 2, "positive", "All_Beauty", "train"),
        ("negative validation", 0, "negative", "All_Beauty", "validation_model_selection"),
        ("neutral validation", 1, "neutral", "Video_Games", "validation_model_selection"),
        ("positive validation", 2, "positive", "All_Beauty", "validation_model_selection"),
        ("policyonlytoken", 2, "positive", "All_Beauty", "validation_policy_calibration"),
        ("testonlytoken", 0, "negative", "Video_Games", "test"),
    ]
    pd.DataFrame(
        rows,
        columns=[
            "text",
            "sentiment_id",
            "sentiment_label",
            "source_category",
            "split",
        ],
    ).to_parquet(dataset_path, index=False)
    decision_path = tmp_path / "training_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "frozen_without_throughput_pilot",
                "max_length": 256,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
            }
        ),
        encoding="utf-8",
    )
    config = TransformerTrainingConfig.from_yaml(
        _training_config(tmp_path),
        dataset_path=dataset_path,
        decision_path=decision_path,
        output_dir=tmp_path / "model-run",
    )
    backend = RecordingBackend()

    bundle = train_transformer(config, backend=backend)

    assert "policyonlytoken" not in backend.seen_text
    assert "testonlytoken" not in backend.seen_text
    assert bundle.model_dir.is_dir()
    outputs = np.load(bundle.validation_outputs_path)
    assert outputs["logits"].shape == (3, 3)
    assert outputs["label_ids"].tolist() == [0, 1, 2]
    summary: dict[str, Any] = json.loads(bundle.training_summary_path.read_text())
    assert summary["training_rows"] == 3
    assert summary["model_selection_rows"] == 3
    assert summary["policy_calibration_evaluated"] is False
    assert summary["test_evaluated"] is False
    assert summary["max_length"] == 256
    assert summary["gradient_accumulation_steps"] == 4


def test_train_transformer_refuses_to_resume_with_changed_dataset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "training_dataset.parquet"
    frame = pd.DataFrame(
        [
            ("negative train", 0, "negative", "All_Beauty", "train"),
            ("neutral train", 1, "neutral", "Video_Games", "train"),
            ("positive train", 2, "positive", "All_Beauty", "train"),
            ("negative validation", 0, "negative", "All_Beauty", "validation_model_selection"),
            ("neutral validation", 1, "neutral", "Video_Games", "validation_model_selection"),
            ("positive validation", 2, "positive", "All_Beauty", "validation_model_selection"),
        ],
        columns=[
            "text",
            "sentiment_id",
            "sentiment_label",
            "source_category",
            "split",
        ],
    )
    frame.to_parquet(dataset_path, index=False)
    decision_path = tmp_path / "training_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "frozen_without_throughput_pilot",
                "max_length": 256,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
            }
        ),
        encoding="utf-8",
    )
    config = TransformerTrainingConfig.from_yaml(
        _training_config(tmp_path),
        dataset_path=dataset_path,
        decision_path=decision_path,
        output_dir=tmp_path / "model-run",
    )
    train_transformer(config, backend=RecordingBackend())
    frame.loc[0, "text"] = "changed negative train"
    frame.to_parquet(dataset_path, index=False)

    with pytest.raises(SchemaError, match="existing training run"):
        train_transformer(config, backend=RecordingBackend())
