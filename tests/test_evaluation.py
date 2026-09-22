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
    calibrate_models,
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
