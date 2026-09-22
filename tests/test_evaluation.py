from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from amazon_sentiment.__main__ import main
from amazon_sentiment.data_pipeline import SchemaError
from amazon_sentiment.evaluation import (
    CalibratedModelOutput,
    CalibrationConfig,
    FinalEvaluationConfig,
    FinalModelOutput,
    calibrate_models,
    evaluate_test,
    fit_sigmoid_calibration,
    fit_temperature_scaling,
)


def _config(tmp_path: Path) -> CalibrationConfig:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "labels": {
                        "negative": {"id": 0},
                        "neutral": {"id": 1},
                        "positive": {"id": 2},
                    }
                },
                "split": {"caps": {"validation_total": 6}},
                "evaluation": {
                    "expected_calibration_error_bins": 5,
                    "selective_accuracy_target": 0.90,
                    "threshold_selection_subset": "policy_calibration",
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_hash = hashlib.sha256(
        (tmp_path / "training_dataset.parquet").read_bytes()
    ).hexdigest()
    run_dir = tmp_path / "distilbert-run"
    (run_dir / "model").mkdir(parents=True)
    identity = {"schema_version": 1, "dataset_sha256": dataset_hash}
    (run_dir / "run_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    (run_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    checksum_lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            checksum_lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(run_dir).as_posix()}"
            )
    (run_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (tmp_path / "tfidf.joblib").write_bytes(b"fixture model")
    return CalibrationConfig.from_yaml(
        config_path,
        dataset_path=tmp_path / "training_dataset.parquet",
        distilbert_run_dir=tmp_path / "distilbert-run",
        tfidf_model_path=tmp_path / "tfidf.joblib",
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "predictions",
    )


class RecordingCalibrationBackend:
    def __init__(self) -> None:
        self.seen_text: list[str] = []

    def calibrate(
        self,
        config: CalibrationConfig,
        policy: pd.DataFrame,
    ) -> dict[str, CalibratedModelOutput]:
        self.seen_text = policy["text"].astype(str).tolist()
        labels = policy["sentiment_id"].to_numpy(dtype=np.int64)
        probabilities = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ],
            dtype=np.float64,
        )
        return {
            name: CalibratedModelOutput(
                label_ids=labels,
                uncalibrated_probabilities=probabilities,
                calibrated_probabilities=probabilities,
                calibration={"method": method},
            )
            for name, method in (
                ("tfidf_logistic_regression", "sigmoid"),
                ("distilbert", "temperature_scaling"),
            )
        }


def test_calibration_reads_only_policy_validation_and_records_test_as_untouched(
    tmp_path: Path,
) -> None:
    rows: list[tuple[Any, ...]] = [
        ("train secret", 0, "negative", "train", "r0"),
        ("selection secret", 1, "neutral", "validation_model_selection", "r1"),
        ("negative policy", 0, "negative", "validation_policy_calibration", "r2"),
        ("neutral policy", 1, "neutral", "validation_policy_calibration", "r3"),
        ("positive policy", 2, "positive", "validation_policy_calibration", "r4"),
        ("test secret", 2, "positive", "test", "r5"),
    ]
    pd.DataFrame(
        rows,
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    backend = RecordingCalibrationBackend()

    bundle = calibrate_models(_config(tmp_path), backend=backend)

    assert backend.seen_text == ["negative policy", "neutral policy", "positive policy"]
    summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))
    assert summary["expected_calibration_error_bins"] == 5
    assert summary["evaluation_split"] == "validation_policy_calibration"
    assert summary["calibration_rows"] == 3
    assert summary["test_evaluated"] is False
    assert "test secret" not in backend.seen_text


def test_calibration_rejects_a_dataset_changed_after_transformer_training(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "training_dataset.parquet"
    frame = pd.DataFrame(
        [
            ("negative policy", 0, "negative", "validation_policy_calibration", "r0"),
            ("neutral policy", 1, "neutral", "validation_policy_calibration", "r1"),
            ("positive policy", 2, "positive", "validation_policy_calibration", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    )
    frame.to_parquet(dataset_path, index=False)
    config = _config(tmp_path)
    frame.loc[0, "text"] = "changed after training"
    frame.to_parquet(dataset_path, index=False)

    with pytest.raises(SchemaError, match="dataset hash"):
        calibrate_models(config, backend=RecordingCalibrationBackend())


def test_calibration_rejects_a_tampered_training_handoff(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ("negative policy", 0, "negative", "validation_policy_calibration", "r0"),
            ("neutral policy", 1, "neutral", "validation_policy_calibration", "r1"),
            ("positive policy", 2, "positive", "validation_policy_calibration", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    config = _config(tmp_path)
    (config.distilbert_run_dir / "model" / "config.json").write_text(
        '{"tampered": true}', encoding="utf-8"
    )

    with pytest.raises(SchemaError, match="checksum"):
        calibrate_models(config, backend=RecordingCalibrationBackend())


def test_calibration_writes_metrics_thresholds_and_aligned_predictions(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [
            ("negative policy", 0, "negative", "validation_policy_calibration", "r0"),
            ("neutral policy", 1, "neutral", "validation_policy_calibration", "r1"),
            ("positive policy", 2, "positive", "validation_policy_calibration", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)

    bundle = calibrate_models(
        _config(tmp_path), backend=RecordingCalibrationBackend()
    )

    summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))
    distilbert = summary["models"]["distilbert"]
    assert distilbert["calibrated"]["negative_log_likelihood"] == pytest.approx(
        0.2231435513
    )
    assert distilbert["calibrated"]["multiclass_brier_score"] == pytest.approx(0.06)
    assert distilbert["calibrated"]["expected_calibration_error"] == pytest.approx(
        0.20
    )
    assert distilbert["review_policy"] == {
        "automatic_rows": 3,
        "coverage": 1.0,
        "human_review_rows": 0,
        "requirement_met": True,
        "selective_accuracy": 1.0,
        "selective_accuracy_target": 0.9,
        "threshold": pytest.approx(0.8),
    }
    thresholds = json.loads(bundle.thresholds_path.read_text(encoding="utf-8"))
    assert thresholds["test_evaluated"] is False
    assert thresholds["models"]["distilbert"]["threshold"] == pytest.approx(0.8)
    predictions = pd.read_parquet(bundle.predictions_path)
    assert predictions["record_id"].tolist() == ["r0", "r1", "r2"]
    assert predictions["distilbert_calibrated_predicted_id"].tolist() == [0, 1, 2]
    assert "text" not in predictions.columns


def test_temperature_scaling_softens_overconfident_logits_and_improves_nll() -> None:
    logits = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 2], dtype=np.int64)

    temperature, calibrated = fit_temperature_scaling(logits, labels)

    uncalibrated = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    row_indices = np.arange(len(labels))
    before_nll = -np.log(uncalibrated[row_indices, labels]).mean()
    after_nll = -np.log(calibrated[row_indices, labels]).mean()
    assert temperature > 1.0
    assert after_nll < before_nll
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(3))


def test_sigmoid_calibration_keeps_the_frozen_tfidf_estimator_unchanged() -> None:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(
        [
            "awful broken",
            "bad damaged",
            "poor useless",
            "ordinary average",
            "fine adequate",
            "okay item",
            "excellent wonderful",
            "great product",
            "perfect purchase",
        ],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    coefficients_before = model.named_steps["classifier"].coef_.copy()
    calibration_text = [
        "broken item",
        "bad purchase",
        "average product",
        "ordinary item",
        "great item",
        "excellent product",
    ]
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    calibrated_model, probabilities = fit_sigmoid_calibration(
        model, calibration_text, labels
    )

    assert probabilities.shape == (6, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(6))
    assert calibrated_model.classes_.tolist() == [0, 1, 2]
    assert model.named_steps["classifier"].coef_ == pytest.approx(coefficients_before)


def test_calibration_command_exposes_the_frozen_model_inputs(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["calibrate-models", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--distilbert-run" in help_text
    assert "--tfidf-model" in help_text
    assert "--predictions-dir" in help_text


class RecordingTestBackend:
    def __init__(self) -> None:
        self.seen_text: list[str] = []

    def predict(
        self,
        config: FinalEvaluationConfig,
        test: pd.DataFrame,
    ) -> dict[str, FinalModelOutput]:
        self.seen_text = test["text"].astype(str).tolist()
        labels = test["sentiment_id"].to_numpy(dtype=np.int64)
        perfect = np.eye(3, dtype=np.float64)[labels] * 0.7 + 0.1
        majority = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(test), 1))
        return {
            "majority": FinalModelOutput(labels, majority, majority),
            "tfidf_logistic_regression": FinalModelOutput(labels, perfect, perfect),
            "distilbert": FinalModelOutput(labels, perfect, perfect),
        }


def _test_config(
    tmp_path: Path,
    *,
    required_slices: list[str] | None = None,
) -> FinalEvaluationConfig:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 42},
                "data": {
                    "labels": {
                        "negative": {"id": 0},
                        "neutral": {"id": 1},
                        "positive": {"id": 2},
                    }
                },
                "split": {"caps": {"test_total": 3}},
                "evaluation": {
                    "bootstrap_resamples": 20,
                    "confidence_interval": 0.95,
                    "minimum_slice_rows": 1,
                    "required_slices": required_slices or [],
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "training_dataset.parquet"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    run_dir = tmp_path / "distilbert-run"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "model" / "model.safetensors").write_bytes(b"fixture weights")
    identity = {"schema_version": 1, "dataset_sha256": dataset_hash}
    (run_dir / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (run_dir / "training_summary.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            checksum_lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(run_dir).as_posix()}"
            )
    (run_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    (baseline_dir / "majority_model.json").write_text(
        json.dumps({"predicted_id": 2}), encoding="utf-8"
    )
    tfidf_path = baseline_dir / "tfidf_logistic_regression.joblib"
    tfidf_path.write_bytes(b"fixture tfidf")
    calibrated_tfidf_path = baseline_dir / "tfidf_sigmoid_calibrated.joblib"
    calibrated_tfidf_path.write_bytes(b"fixture calibrated tfidf")
    (baseline_dir / "tfidf_sigmoid_calibration.json").write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_hash,
                "base_model_sha256": hashlib.sha256(tfidf_path.read_bytes()).hexdigest(),
                "calibrated_model_sha256": hashlib.sha256(
                    calibrated_tfidf_path.read_bytes()
                ).hexdigest(),
                "test_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": 1.2,
                "dataset_sha256": dataset_hash,
                "model_sha256": hashlib.sha256(
                    (run_dir / "model" / "model.safetensors").read_bytes()
                ).hexdigest(),
                "test_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    thresholds_path = tmp_path / "review_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "test_evaluated": False,
                "models": {
                    "distilbert": {"threshold": 0.7, "requirement_met": True},
                    "tfidf_logistic_regression": {
                        "threshold": 0.65,
                        "requirement_met": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    calibration_summary_path = tmp_path / "policy_calibration_summary.json"
    calibration_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "test_evaluated": False,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "dataset_sha256": dataset_hash,
                "tfidf_model_sha256": hashlib.sha256(tfidf_path.read_bytes()).hexdigest(),
                "thresholds_sha256": hashlib.sha256(
                    thresholds_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return FinalEvaluationConfig.from_yaml(
        config_path,
        dataset_path=tmp_path / "training_dataset.parquet",
        distilbert_run_dir=tmp_path / "distilbert-run",
        baseline_model_dir=tmp_path / "baselines",
        calibration_summary_path=calibration_summary_path,
        thresholds_path=thresholds_path,
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "predictions",
    )


def test_final_evaluation_reads_only_the_untouched_test_split(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ("train secret", 0, "negative", "train", "r0"),
            ("policy secret", 1, "neutral", "validation_policy_calibration", "r1"),
            ("negative test", 0, "negative", "test", "r2"),
            ("neutral test", 1, "neutral", "test", "r3"),
            ("positive test", 2, "positive", "test", "r4"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    backend = RecordingTestBackend()

    bundle = evaluate_test(_test_config(tmp_path), backend=backend)

    assert backend.seen_text == ["negative test", "neutral test", "positive test"]
    metrics = json.loads(bundle.metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluation_split"] == "test"
    assert metrics["test_rows"] == 3
    assert metrics["test_evaluated"] is True


def test_final_evaluation_rejects_changed_frozen_thresholds(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ("negative test", 0, "negative", "test", "r0"),
            ("neutral test", 1, "neutral", "test", "r1"),
            ("positive test", 2, "positive", "test", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    config = _test_config(tmp_path)
    config.thresholds_path.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(SchemaError, match="threshold"):
        evaluate_test(config, backend=RecordingTestBackend())


def test_final_evaluation_rejects_a_tampered_calibrated_model(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ("negative test", 0, "negative", "test", "r0"),
            ("neutral test", 1, "neutral", "test", "r1"),
            ("positive test", 2, "positive", "test", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    config = _test_config(tmp_path)
    (config.baseline_model_dir / "tfidf_sigmoid_calibrated.joblib").write_bytes(
        b"tampered"
    )

    with pytest.raises(SchemaError, match="calibrated TF-IDF"):
        evaluate_test(config, backend=RecordingTestBackend())


def test_final_evaluation_refuses_to_overwrite_an_existing_test_result(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [
            ("negative test", 0, "negative", "test", "r0"),
            ("neutral test", 1, "neutral", "test", "r1"),
            ("positive test", 2, "positive", "test", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)
    config = _test_config(tmp_path)
    evaluate_test(config, backend=RecordingTestBackend())

    with pytest.raises(SchemaError, match="already exists"):
        evaluate_test(config, backend=RecordingTestBackend())


def test_final_evaluation_writes_metrics_uncertainty_and_private_predictions(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [
            ("negative test", 0, "negative", "test", "r0"),
            ("neutral test", 1, "neutral", "test", "r1"),
            ("positive test", 2, "positive", "test", "r2"),
        ],
        columns=["text", "sentiment_id", "sentiment_label", "split", "record_id"],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)

    bundle = evaluate_test(_test_config(tmp_path), backend=RecordingTestBackend())

    metrics = json.loads(bundle.metrics_path.read_text(encoding="utf-8"))
    assert metrics["models"]["distilbert"]["classification"]["macro_f1"] == 1.0
    assert metrics["models"]["majority"]["classification"]["macro_f1"] == pytest.approx(
        1 / 6
    )
    assert metrics["models"]["distilbert"]["review_policy"]["coverage"] == 1.0
    assert metrics["bootstrap"]["macro_f1"]["distilbert"] == {
        "confidence_level": 0.95,
        "lower": 1.0,
        "point_estimate": 1.0,
        "upper": 1.0,
    }
    assert metrics["bootstrap"]["tfidf_minus_distilbert_macro_f1"][
        "point_estimate"
    ] == 0.0
    predictions = pd.read_parquet(bundle.predictions_path)
    assert len(predictions) == 3
    assert "text" not in predictions.columns
    assert predictions["distilbert_predicted_id"].tolist() == [0, 1, 2]


def test_final_evaluation_command_exposes_the_frozen_inputs(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-test", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--distilbert-run" in help_text
    assert "--baseline-model-dir" in help_text
    assert "--calibration-summary" in help_text
    assert "--thresholds" in help_text


def test_final_evaluation_writes_declared_slice_metrics(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ("negative test", 0, "negative", "test", "r0", "All_Beauty"),
            ("neutral test", 1, "neutral", "test", "r1", "Video_Games"),
            ("positive test", 2, "positive", "test", "r2", "All_Beauty"),
        ],
        columns=[
            "text",
            "sentiment_id",
            "sentiment_label",
            "split",
            "record_id",
            "source_category",
        ],
    ).to_parquet(tmp_path / "training_dataset.parquet", index=False)

    bundle = evaluate_test(
        _test_config(tmp_path, required_slices=["category"]),
        backend=RecordingTestBackend(),
    )

    slices = pd.read_csv(bundle.slice_metrics_path)
    assert set(slices["model"]) == {
        "majority",
        "tfidf_logistic_regression",
        "distilbert",
    }
    assert set(slices["slice_type"]) == {"category"}
    assert set(slices["slice_value"]) == {"All_Beauty", "Video_Games"}
