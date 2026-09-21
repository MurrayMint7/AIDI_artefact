from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from amazon_sentiment.baselines import BaselineConfig, train_baselines


def _baseline_fixture(tmp_path: Path) -> BaselineConfig:
    rows = [
        ("awful broken product", 0, "negative", "train"),
        ("bad and disappointing", 0, "negative", "train"),
        ("average ordinary item", 1, "neutral", "train"),
        ("fine but unremarkable", 1, "neutral", "train"),
        ("excellent wonderful product", 2, "positive", "train"),
        ("great and delightful", 2, "positive", "train"),
        ("broken and bad", 0, "negative", "validation_model_selection"),
        ("ordinary item", 1, "neutral", "validation_model_selection"),
        ("wonderful and great", 2, "positive", "validation_model_selection"),
        ("policyonlytoken", 2, "positive", "validation_policy_calibration"),
        ("testonlytoken", 0, "negative", "test"),
    ]
    dataset = pd.DataFrame(
        rows,
        columns=["text", "sentiment_id", "sentiment_label", "split"],
    )
    dataset_path = tmp_path / "training_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    split_manifest_path = tmp_path / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "pre_sampling_training_class_counts": [
                    {
                        "source_category": "All_Beauty",
                        "sentiment_label": "negative",
                        "rows": 10,
                    },
                    {
                        "source_category": "All_Beauty",
                        "sentiment_label": "neutral",
                        "rows": 5,
                    },
                    {
                        "source_category": "All_Beauty",
                        "sentiment_label": "positive",
                        "rows": 100,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = {
        "project": {"seed": 42},
        "data": {
            "labels": {
                "negative": {"id": 0, "ratings": [1, 2]},
                "neutral": {"id": 1, "ratings": [3]},
                "positive": {"id": 2, "ratings": [4, 5]},
            }
        },
        "models": {
            "majority": {
                "prevalence_source": "full_pre_balance_training_population"
            },
            "tfidf_logistic_regression": {
                "word_ngram_range": [1, 2],
                "min_df_candidates": [1],
                "max_features_candidates": [100],
                "c_candidates": [1.0],
                "max_iter": 100,
                "tune_on": "model_selection",
            },
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return BaselineConfig.from_yaml(
        config_path,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
    )


def test_train_baselines_uses_only_training_and_model_selection_rows(
    tmp_path: Path,
) -> None:
    bundle = train_baselines(_baseline_fixture(tmp_path))

    assert bundle.majority_model_path.is_file()
    assert bundle.tfidf_model_path.is_file()
    assert bundle.metrics_path.is_file()

    majority = json.loads(bundle.majority_model_path.read_text())
    assert majority["predicted_label"] == "positive"
    assert majority["training_class_counts"] == {
        "negative": 10,
        "neutral": 5,
        "positive": 100,
    }

    metrics = json.loads(bundle.metrics_path.read_text())
    assert metrics["schema_version"] == 1
    assert metrics["evaluation_split"] == "validation_model_selection"
    assert metrics["test_evaluated"] is False
    assert metrics["runs"]["majority"]["metrics"]["rows"] == 3
    assert metrics["runs"]["tfidf_logistic_regression"]["metrics"]["rows"] == 3
    assert metrics["runs"]["tfidf_logistic_regression"]["selected_parameters"] == {
        "C": 1.0,
        "max_features": 100,
        "min_df": 1,
    }

    model = joblib.load(bundle.tfidf_model_path)
    vocabulary = model.named_steps["tfidf"].vocabulary_
    assert "testonlytoken" not in vocabulary
    assert "policyonlytoken" not in vocabulary
    probabilities = model.predict_proba(["great product", "bad product"])
    assert probabilities.shape == (2, 3)
    assert abs(probabilities.sum(axis=1) - 1.0).max() < 1e-12
