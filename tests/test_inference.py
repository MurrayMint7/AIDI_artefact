from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from amazon_sentiment.data_pipeline import SchemaError
from amazon_sentiment.inference import (
    InferenceConfig,
    InferenceInputError,
    ModelOutput,
    Prediction,
    SentimentPredictor,
)
from amazon_sentiment.interface import build_app, present_prediction


class FixedBackend:
    def __init__(self, logits: list[float], *, token_length: int = 12):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.token_length = token_length
        self.calls: list[tuple[str, int]] = []

    def predict(self, text: str, *, max_length: int) -> ModelOutput:
        self.calls.append((text, max_length))
        return ModelOutput(logits=self.logits, token_length=self.token_length)


def _inference_config(tmp_path: Path, *, log: bool = True) -> InferenceConfig:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "labels": {
                        "negative": {"id": 0},
                        "neutral": {"id": 1},
                        "positive": {"id": 2},
                    }
                },
                "models": {"distilbert": {"max_length": 256}},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "distilbert-run"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "model.safetensors"
    weights_path.write_bytes(b"fixture weights")
    model_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "id2label": {
                    "0": "negative",
                    "1": "neutral",
                    "2": "positive",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": 2.0,
                "fit_split": "validation_policy_calibration",
                "test_evaluated": False,
                "model_sha256": model_hash,
            }
        ),
        encoding="utf-8",
    )
    thresholds_path = tmp_path / "review_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "stage": "policy_calibration",
                "evaluation_split": "validation_policy_calibration",
                "test_evaluated": False,
                "models": {
                    "distilbert": {
                        "threshold": 0.65,
                        "requirement_met": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    deployment_path = tmp_path / "deployment_recommendation.json"
    deployment_path.write_text(
        json.dumps(
            {
                "recommended_default_model": "distilbert",
                "proposed_ui_policy": {
                    "mandatory_human_review_for_predicted_labels": ["neutral"]
                },
            }
        ),
        encoding="utf-8",
    )
    return InferenceConfig.from_paths(
        experiment_path,
        distilbert_run_dir=run_dir,
        thresholds_path=thresholds_path,
        deployment_policy_path=deployment_path,
        log_path=tmp_path / "logs" / "inference.jsonl" if log else None,
    )


def _predictor(
    config: InferenceConfig,
    backend: FixedBackend,
) -> SentimentPredictor:
    clock_values = iter([10.0, 10.025])
    return SentimentPredictor(
        config,
        backend=backend,
        clock=lambda: next(clock_values),
        request_id_factory=lambda: "request-123",
        now_factory=lambda: datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc),
    )


def test_predict_applies_temperature_threshold_and_redacted_logging(
    tmp_path: Path,
) -> None:
    config = _inference_config(tmp_path)
    backend = FixedBackend([4.0, 0.0, 0.0], token_length=9)

    prediction = _predictor(config, backend).predict("Private review text")

    expected = np.exp(np.array([2.0, 0.0, 0.0]))
    expected /= expected.sum()
    assert backend.calls == [("Private review text", 256)]
    assert prediction.label == "negative"
    assert prediction.confidence == pytest.approx(expected[0])
    assert prediction.probabilities == pytest.approx(
        {"negative": expected[0], "neutral": expected[1], "positive": expected[2]}
    )
    assert prediction.automatic_route is True
    assert prediction.review_reason == "frozen_confidence_threshold_met"
    assert prediction.input_character_length == len("Private review text")
    assert prediction.input_token_length == 9
    assert prediction.latency_ms == pytest.approx(25.0)

    assert config.log_path is not None
    log_text = config.log_path.read_text(encoding="utf-8")
    record = json.loads(log_text)
    assert record["request_id"] == "request-123"
    assert record["automatic_route"] is True
    assert record["model_version"].startswith("distilbert:")
    assert "Private review text" not in log_text
    assert "text" not in record


def test_neutral_prediction_always_requires_human_review(tmp_path: Path) -> None:
    config = _inference_config(tmp_path, log=False)
    prediction = _predictor(config, FixedBackend([0.0, 20.0, 0.0])).predict(
        "It was acceptable."
    )

    assert prediction.label == "neutral"
    assert prediction.confidence > 0.99
    assert prediction.review_required is True
    assert prediction.review_reason == "mandatory_review_for_predicted_label"


def test_low_confidence_non_neutral_prediction_requires_review(tmp_path: Path) -> None:
    config = _inference_config(tmp_path, log=False)
    prediction = _predictor(config, FixedBackend([0.4, 0.0, 0.0])).predict(
        "Hard to tell"
    )

    assert prediction.label == "negative"
    assert prediction.confidence < config.threshold
    assert prediction.review_required is True
    assert prediction.review_reason == "below_frozen_confidence_threshold"


def test_empty_input_is_rejected_before_model_or_log_access(tmp_path: Path) -> None:
    config = _inference_config(tmp_path)
    backend = FixedBackend([1.0, 0.0, 0.0])

    with pytest.raises(InferenceInputError, match="Enter review text"):
        _predictor(config, backend).predict("  \n ")

    assert backend.calls == []
    assert config.log_path is not None
    assert not config.log_path.exists()


def test_inference_rejects_weights_that_do_not_match_calibration(tmp_path: Path) -> None:
    config = _inference_config(tmp_path, log=False)
    weights_path = config.distilbert_run_dir / "model" / "model.safetensors"
    weights_path.write_bytes(b"changed")

    with pytest.raises(SchemaError, match="calibrated model hash"):
        InferenceConfig.from_paths(
            config.experiment_config_path,
            distilbert_run_dir=config.distilbert_run_dir,
            thresholds_path=config.thresholds_path,
            deployment_policy_path=config.deployment_policy_path,
            log_path=None,
        )


class StubPredictor:
    def __init__(self, prediction: Prediction):
        self.prediction = prediction

    def predict(self, text: str) -> Prediction:
        if not text.strip():
            raise InferenceInputError("Enter review text before submitting.")
        return self.prediction


def _neutral_prediction() -> Prediction:
    return Prediction(
        request_id="request-123",
        timestamp_utc="2026-09-23T12:00:00+00:00",
        label="neutral",
        probabilities={"negative": 0.02, "neutral": 0.96, "positive": 0.02},
        confidence=0.96,
        review_required=True,
        review_reason="mandatory_review_for_predicted_label",
        model_version="distilbert:abc123",
        config_version="config123",
        policy_version="policy123",
        threshold=0.69,
        input_character_length=10,
        input_token_length=4,
        latency_ms=20.0,
    )


def test_interface_presents_label_probabilities_version_and_review_state() -> None:
    rendered = present_prediction("It was fine", StubPredictor(_neutral_prediction()))

    assert rendered.validation_html == ""
    assert "Human review required" in rendered.result_html
    assert "Neutral" in rendered.result_html
    assert "96.0%" in rendered.result_html
    assert "distilbert:abc123" in rendered.result_html
    assert "<table>" in rendered.result_html
    assert 'aria-live="polite"' in rendered.result_html
    assert "always require review" in rendered.result_html


def test_interface_returns_a_nearby_accessible_empty_input_error() -> None:
    rendered = present_prediction("", StubPredictor(_neutral_prediction()))

    assert 'role="alert"' in rendered.validation_html
    assert "Enter review text" in rendered.validation_html
    assert "Submit a review" in rendered.result_html


def test_gradio_app_builds_offline_with_required_controls() -> None:
    gradio = pytest.importorskip("gradio")

    demo = build_app(StubPredictor(_neutral_prediction()))
    config = demo.get_config_file()
    component_values = {
        component.get("props", {}).get("value")
        for component in config.get("components", [])
    }
    component_labels = {
        component.get("props", {}).get("label")
        for component in config.get("components", [])
    }

    assert isinstance(demo, gradio.Blocks)
    assert "Review text" in component_labels
    assert "Predict sentiment" in component_values
    assert "Clear" in component_values
    assert config.get("analytics_enabled") is False
