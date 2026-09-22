from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from amazon_sentiment.__main__ import main
from amazon_sentiment.data_pipeline import SchemaError
from amazon_sentiment.post_evaluation import EvidenceConfig, build_evaluation_evidence


class FixedBenchmarkBackend:
    def benchmark(self, config: EvidenceConfig, texts: list[str]) -> dict[str, Any]:
        assert len(texts) == 3
        return {
            "models": {
                "tfidf_logistic_regression": {
                    "serving_artifact_bytes": 10,
                    "serving_artifact_mebibytes": 10 / (1024**2),
                    "load_seconds": 0.01,
                    "latency_ms": {
                        "median": 1.0,
                        "p95": 1.5,
                        "minimum": 0.8,
                        "maximum": 1.6,
                    },
                },
                "distilbert": {
                    "serving_artifact_bytes": 100,
                    "serving_artifact_mebibytes": 100 / (1024**2),
                    "load_seconds": 0.1,
                    "latency_ms": {
                        "median": 5.0,
                        "p95": 6.0,
                        "minimum": 4.0,
                        "maximum": 6.5,
                    },
                },
            },
            "runtime": {"processor": "fixture CPU"},
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics() -> dict[str, Any]:
    per_class_tfidf = [
        {"label": "negative", "f1": 0.7},
        {"label": "neutral", "f1": 0.2},
        {"label": "positive", "f1": 0.9},
    ]
    per_class_distilbert = [
        {"label": "negative", "f1": 0.8},
        {"label": "neutral", "f1": 0.4},
        {"label": "positive", "f1": 0.9},
    ]

    def model_record(
        score: float,
        per_class: list[dict[str, Any]],
        matrix: list[list[int]],
        *,
        calibration: bool,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "classification": {
                "macro_f1": score,
                "per_class": per_class,
                "confusion_matrix": {"values": matrix},
            }
        }
        if calibration:
            record.update(
                {
                    "uncalibrated": {
                        "negative_log_likelihood": 0.7,
                        "multiclass_brier_score": 0.4,
                        "expected_calibration_error": 0.1,
                    },
                    "calibrated": {
                        "negative_log_likelihood": 0.5,
                        "multiclass_brier_score": 0.3,
                        "expected_calibration_error": 0.05,
                    },
                    "review_policy": {
                        "coverage": 0.75,
                        "selective_accuracy": 0.91,
                    },
                }
            )
        return record

    return {
        "schema_version": 1,
        "test_evaluated": True,
        "test_rows": 6,
        "models": {
            "majority": model_record(
                0.25,
                [
                    {"label": label, "f1": 0.0}
                    for label in ("negative", "neutral", "positive")
                ],
                [[0, 0, 2], [0, 0, 2], [0, 0, 2]],
                calibration=False,
            ),
            "tfidf_logistic_regression": model_record(
                0.6,
                per_class_tfidf,
                [[2, 0, 0], [1, 1, 0], [0, 0, 2]],
                calibration=True,
            ),
            "distilbert": model_record(
                0.7,
                per_class_distilbert,
                [[2, 0, 0], [0, 1, 1], [0, 0, 2]],
                calibration=True,
            ),
        },
        "bootstrap": {
            "macro_f1": {
                "majority": {"lower": 0.2, "upper": 0.3},
                "tfidf_logistic_regression": {"lower": 0.55, "upper": 0.65},
                "distilbert": {"lower": 0.65, "upper": 0.75},
            },
            "tfidf_minus_distilbert_macro_f1": {
                "lower": -0.15,
                "point_estimate": -0.1,
                "upper": -0.05,
            },
        },
    }


def _config(tmp_path: Path) -> EvidenceConfig:
    protocol_path = tmp_path / "evaluation_evidence.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "latency": {
                    "device": "cpu",
                    "batch_size": 1,
                    "warmup_runs": 1,
                    "measured_runs": 3,
                    "deterministic_sample_rows": 3,
                },
                "error_analysis": {
                    "model": "distilbert",
                    "highest_confidence_errors": 2,
                    "lowest_confidence_cases": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump({"models": {"distilbert": {"max_length": 256}}}),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        [
            (
                f"r{index}",
                f"private review text {index}",
                rating,
                label,
                "test",
                "category",
            )
            for index, (rating, label) in enumerate(
                [
                    (1, "negative"),
                    (3, "neutral"),
                    (5, "positive"),
                    (2, "negative"),
                    (3, "neutral"),
                    (4, "positive"),
                ]
            )
        ],
        columns=[
            "record_id",
            "text",
            "rating",
            "sentiment_label",
            "split",
            "source_category",
        ],
    ).assign(timestamp=pd.Timestamp("2026-01-01", tz="UTC")).to_parquet(
        dataset_path, index=False
    )
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "record_id": [f"r{index}" for index in range(6)],
            "sentiment_id": [0, 1, 2, 0, 1, 2],
            "sentiment_label": [
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
            ],
            "distilbert_predicted_id": [0, 2, 2, 1, 1, 2],
            "distilbert_confidence": [0.9, 0.95, 0.8, 0.7, 0.4, 0.6],
            "distilbert_automatic_route": [True, True, True, True, False, False],
            "tfidf_logistic_regression_predicted_id": [0, 0, 2, 0, 2, 2],
            "tfidf_logistic_regression_confidence": [0.8, 0.7, 0.9, 0.6, 0.5, 0.8],
            "tfidf_logistic_regression_automatic_route": [
                True,
                True,
                True,
                False,
                False,
                True,
            ],
        }
    ).to_parquet(predictions_path, index=False)
    slices_path = tmp_path / "slices.csv"
    pd.DataFrame(
        [
            (model, "category", value, 3, score)
            for model, score in [
                ("majority", 0.25),
                ("tfidf_logistic_regression", 0.6),
                ("distilbert", 0.7),
            ]
            for value in ["a", "b"]
        ],
        columns=["model", "slice_type", "slice_value", "rows", "macro_f1"],
    ).to_csv(slices_path, index=False)
    metrics = _metrics()
    metrics.update(
        {
            "dataset_sha256": _sha256(dataset_path),
            "predictions_sha256": _sha256(predictions_path),
            "slice_metrics_sha256": _sha256(slices_path),
        }
    )
    metrics_path = tmp_path / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    run_dir = tmp_path / "distilbert-run"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "calibration.json").write_text(
        json.dumps({"temperature": 1.0}), encoding="utf-8"
    )
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    (baseline_dir / "tfidf_sigmoid_calibrated.joblib").write_bytes(b"fixture")
    return EvidenceConfig.from_yaml(
        protocol_path,
        experiment_config_path=experiment_path,
        dataset_path=dataset_path,
        final_metrics_path=metrics_path,
        slice_metrics_path=slices_path,
        predictions_path=predictions_path,
        distilbert_run_dir=run_dir,
        baseline_model_dir=baseline_dir,
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        private_dir=tmp_path / "private",
    )


def test_post_evaluation_builds_public_evidence_and_private_error_sample(
    tmp_path: Path,
) -> None:
    bundle = build_evaluation_evidence(
        _config(tmp_path), benchmark_backend=FixedBenchmarkBackend()
    )

    benchmark = json.loads(bundle.benchmark_path.read_text(encoding="utf-8"))
    assert benchmark["protocol"]["batch_size"] == 1
    assert benchmark["models"]["distilbert"]["latency_ms"]["p95"] == 6.0
    recommendation = json.loads(bundle.recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["recommended_default_model"] == "distilbert"
    assert recommendation["proposed_ui_policy"]["status"] == (
        "post-test mitigation, not independently validated"
    )
    summary_text = bundle.error_summary_path.read_text(encoding="utf-8")
    assert "private review text" not in summary_text
    private = pd.read_csv(bundle.error_sample_path)
    assert len(private) == 4
    assert private["text"].str.contains("private review text").all()
    assert set(private["coding_status"]) == {"pending_author_review"}
    assert len(bundle.figure_paths) == 5
    assert all(
        path.read_text(encoding="utf-8").startswith("<svg")
        for path in bundle.figure_paths
    )


def test_post_evaluation_rejects_predictions_changed_after_final_metrics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    changed = pd.read_parquet(config.predictions_path)
    changed.loc[0, "distilbert_confidence"] = 0.1
    changed.to_parquet(config.predictions_path, index=False)

    with pytest.raises(SchemaError, match="predictions_sha256"):
        build_evaluation_evidence(config, benchmark_backend=FixedBenchmarkBackend())


def test_evaluation_evidence_command_exposes_public_and_private_outputs(
    capsys: Any,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluation-evidence", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--protocol" in help_text
    assert "--figures-dir" in help_text
    assert "--private-dir" in help_text

