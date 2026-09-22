"""Calibrate frozen sentiment models without reading the protected test split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .data_pipeline import SchemaError


@dataclass(frozen=True)
class CalibrationConfig:
    """Resolved inputs for policy calibration."""

    config_path: Path
    dataset_path: Path
    distilbert_run_dir: Path
    tfidf_model_path: Path
    metrics_dir: Path
    predictions_dir: Path
    settings: Mapping[str, Any]

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        dataset_path: str | Path,
        distilbert_run_dir: str | Path,
        tfidf_model_path: str | Path,
        metrics_dir: str | Path,
        predictions_dir: str | Path,
    ) -> "CalibrationConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Experiment configuration must be a YAML mapping")
        evaluation = settings.get("evaluation", {})
        if evaluation.get("threshold_selection_subset") != "policy_calibration":
            raise SchemaError(
                "Calibration threshold selection must use policy_calibration"
            )
        return cls(
            config_path=resolved_config,
            dataset_path=Path(dataset_path),
            distilbert_run_dir=Path(distilbert_run_dir),
            tfidf_model_path=Path(tfidf_model_path),
            metrics_dir=Path(metrics_dir),
            predictions_dir=Path(predictions_dir),
            settings=settings,
        )


@dataclass(frozen=True)
class CalibratedModelOutput:
    """Aligned labels and probabilities before and after calibration."""

    label_ids: np.ndarray
    uncalibrated_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    calibration: Mapping[str, Any]
    raw_scores: np.ndarray | None = None


class CalibrationBackend(Protocol):
    """Runtime boundary for model loading, prediction and calibrator fitting."""

    def calibrate(
        self,
        config: CalibrationConfig,
        policy: pd.DataFrame,
    ) -> Mapping[str, CalibratedModelOutput]: ...


@dataclass(frozen=True)
class CalibrationBundle:
    """Aggregate policy-calibration evidence."""

    summary_path: Path
    thresholds_path: Path
    predictions_path: Path


class LocalCalibrationBackend:
    """Load the frozen local models and fit their declared calibrators."""

    def calibrate(
        self,
        config: CalibrationConfig,
        policy: pd.DataFrame,
    ) -> Mapping[str, CalibratedModelOutput]:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - depends on local runtime
            raise RuntimeError(
                "DistilBERT calibration requires a local PyTorch installation"
            ) from error

        labels = _ordered_labels(config.settings)
        expected_classes = np.array([label_id for label_id, _ in labels])
        label_ids = policy["sentiment_id"].to_numpy(dtype=np.int64)
        texts = policy["text"].astype(str).tolist()

        tfidf_model = joblib.load(config.tfidf_model_path)
        tfidf_classes = np.asarray(getattr(tfidf_model, "classes_", []))
        if not np.array_equal(tfidf_classes, expected_classes):
            raise RuntimeError(
                f"Frozen TF-IDF classes {tfidf_classes.tolist()} do not match "
                f"configured classes {expected_classes.tolist()}"
            )
        tfidf_uncalibrated = np.asarray(
            tfidf_model.predict_proba(texts), dtype=np.float64
        )
        tfidf_scores = np.asarray(tfidf_model.decision_function(texts), dtype=np.float64)
        tfidf_calibrated_model, tfidf_calibrated = fit_sigmoid_calibration(
            tfidf_model, texts, label_ids
        )
        tfidf_calibrator_path = (
            config.tfidf_model_path.parent / "tfidf_sigmoid_calibrated.joblib"
        )
        joblib.dump(tfidf_calibrated_model, tfidf_calibrator_path)
        tfidf_calibration_path = (
            config.tfidf_model_path.parent / "tfidf_sigmoid_calibration.json"
        )
        _write_json(
            tfidf_calibration_path,
            {
                "schema_version": 1,
                "method": "sigmoid",
                "fit_split": "validation_policy_calibration",
                "rows": int(len(policy)),
                "dataset_sha256": _sha256(config.dataset_path),
                "base_model_sha256": _sha256(config.tfidf_model_path),
                "calibrated_model_sha256": _sha256(tfidf_calibrator_path),
                "test_evaluated": False,
            },
        )

        model_dir = config.distilbert_run_dir / "model"
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), use_fast=True, local_files_only=True
        )
        transformer = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer.to(device)
        transformer.eval()
        evaluation_settings = config.settings.get("evaluation", {})
        batch_size = int(evaluation_settings.get("distilbert_inference_batch_size", 16))
        max_length = int(
            config.settings.get("models", {})
            .get("distilbert", {})
            .get("max_length", 256)
        )
        if batch_size < 1 or max_length < 1:
            raise SchemaError("DistilBERT inference batch size and max length must be positive")
        logit_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[offset : offset + batch_size],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = transformer(**encoded).logits
                logit_batches.append(logits.detach().cpu().numpy())
        distilbert_logits = np.concatenate(logit_batches, axis=0).astype(np.float64)
        distilbert_uncalibrated = _softmax(distilbert_logits)
        temperature, distilbert_calibrated = fit_temperature_scaling(
            distilbert_logits, label_ids
        )
        distilbert_calibration_path = config.distilbert_run_dir / "calibration.json"
        _write_json(
            distilbert_calibration_path,
            {
                "schema_version": 1,
                "method": "temperature_scaling",
                "temperature": temperature,
                "fit_split": "validation_policy_calibration",
                "rows": int(len(policy)),
                "dataset_sha256": _sha256(config.dataset_path),
                "model_sha256": _model_weights_sha256(model_dir),
                "test_evaluated": False,
            },
        )
        return {
            "tfidf_logistic_regression": CalibratedModelOutput(
                label_ids=label_ids,
                raw_scores=tfidf_scores,
                uncalibrated_probabilities=tfidf_uncalibrated,
                calibrated_probabilities=tfidf_calibrated,
                calibration={
                    "method": "sigmoid",
                    "artifact": str(tfidf_calibrator_path),
                    "metadata": str(tfidf_calibration_path),
                },
            ),
            "distilbert": CalibratedModelOutput(
                label_ids=label_ids,
                raw_scores=distilbert_logits,
                uncalibrated_probabilities=distilbert_uncalibrated,
                calibrated_probabilities=distilbert_calibrated,
                calibration={
                    "method": "temperature_scaling",
                    "temperature": temperature,
                    "artifact": str(distilbert_calibration_path),
                },
            ),
        }


def fit_temperature_scaling(
    logits: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit one positive temperature by minimising multiclass log loss."""

    resolved_logits = np.asarray(logits, dtype=np.float64)
    resolved_labels = np.asarray(label_ids, dtype=np.int64)
    if resolved_logits.ndim != 2 or resolved_logits.shape[0] != len(resolved_labels):
        raise ValueError("Temperature scaling needs aligned two-dimensional logits")
    if not np.isfinite(resolved_logits).all():
        raise ValueError("Temperature scaling logits must be finite")
    if (
        (resolved_labels < 0).any()
        or (resolved_labels >= resolved_logits.shape[1]).any()
    ):
        raise ValueError("Temperature scaling labels are outside the logit columns")

    row_indices = np.arange(len(resolved_labels))

    def objective(log_temperature: float) -> float:
        probabilities = _softmax(resolved_logits / np.exp(log_temperature))
        likelihoods = np.clip(
            probabilities[row_indices, resolved_labels], 1e-12, 1.0
        )
        return float(-np.log(likelihoods).mean())

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(100.0)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    temperature = float(np.exp(result.x))
    return temperature, _softmax(resolved_logits / temperature)


def fit_sigmoid_calibration(
    frozen_model: Any,
    texts: list[str],
    label_ids: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """Fit sigmoid calibration without refitting the supplied classifier."""

    resolved_labels = np.asarray(label_ids, dtype=np.int64)
    if len(texts) != len(resolved_labels) or not texts:
        raise ValueError("Sigmoid calibration needs aligned non-empty texts and labels")
    _, class_counts = np.unique(resolved_labels, return_counts=True)
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        raise ValueError("Sigmoid calibration needs at least two rows for every class")
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(frozen_model),
        method="sigmoid",
        cv=min(5, int(class_counts.min())),
    )
    calibrated_model.fit(texts, resolved_labels)
    probabilities = np.asarray(calibrated_model.predict_proba(texts), dtype=np.float64)
    return calibrated_model, probabilities


def calibrate_models(
    config: CalibrationConfig,
    *,
    backend: CalibrationBackend | None = None,
) -> CalibrationBundle:
    """Calibrate frozen models using only policy-calibration validation rows."""

    _validate_frozen_inputs(config)
    required = {"text", "sentiment_id", "sentiment_label", "split", "record_id"}
    policy = pd.read_parquet(
        config.dataset_path,
        columns=sorted(required),
        filters=[("split", "==", "validation_policy_calibration")],
    )
    missing = required.difference(policy.columns)
    if missing:
        raise SchemaError(f"Calibration dataset is missing fields: {sorted(missing)}")
    if policy.empty:
        raise SchemaError("Calibration selected zero policy-validation rows")
    unexpected = set(policy["split"].astype(str)).difference(
        {"validation_policy_calibration"}
    )
    if unexpected:
        raise SchemaError(f"Calibration read protected splits: {sorted(unexpected)}")

    expected_rows = int(
        config.settings.get("split", {}).get("caps", {}).get("validation_total", 0)
    ) // 2
    if expected_rows and len(policy) != expected_rows:
        raise SchemaError(
            f"Calibration expected {expected_rows} policy rows, found {len(policy)}"
        )

    policy = policy.reset_index(drop=True)
    outputs = (backend or LocalCalibrationBackend()).calibrate(config, policy)
    expected_models = {"tfidf_logistic_regression", "distilbert"}
    if set(outputs) != expected_models:
        raise RuntimeError(
            f"Calibration backend returned models {sorted(outputs)}, "
            f"expected {sorted(expected_models)}"
        )

    labels = _ordered_labels(config.settings)
    label_ids = [label_id for label_id, _ in labels]
    if label_ids != list(range(len(label_ids))):
        raise SchemaError("Calibration label IDs must be contiguous and start at zero")
    expected_label_ids = policy["sentiment_id"].to_numpy(dtype=np.int64)
    target = float(
        config.settings.get("evaluation", {}).get("selective_accuracy_target", 0.90)
    )
    calibration_bins = int(
        config.settings.get("evaluation", {}).get(
            "expected_calibration_error_bins", 10
        )
    )
    if not 0 < target <= 1:
        raise SchemaError("selective_accuracy_target must be in (0, 1]")
    if calibration_bins < 2:
        raise SchemaError("expected_calibration_error_bins must be at least two")

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.predictions_dir / "policy_calibration_predictions.parquet"
    prediction_frame = policy[["record_id", "sentiment_id", "sentiment_label"]].copy()
    model_summaries: dict[str, Any] = {}
    threshold_records: dict[str, Any] = {}
    for model_name in sorted(outputs):
        output = outputs[model_name]
        _validate_model_output(
            model_name,
            output,
            expected_label_ids=expected_label_ids,
            num_labels=len(labels),
        )
        uncalibrated = np.asarray(output.uncalibrated_probabilities, dtype=np.float64)
        calibrated = np.asarray(output.calibrated_probabilities, dtype=np.float64)
        policy_record = _select_review_policy(calibrated, expected_label_ids, target)
        model_summaries[model_name] = {
            "calibration": dict(output.calibration),
            "uncalibrated": _calibration_metrics(
                uncalibrated, expected_label_ids, bins=calibration_bins
            ),
            "calibrated": _calibration_metrics(
                calibrated, expected_label_ids, bins=calibration_bins
            ),
            "review_policy": policy_record,
        }
        threshold_records[model_name] = policy_record
        prefix = f"{model_name}_calibrated"
        prediction_frame[f"{prefix}_predicted_id"] = calibrated.argmax(axis=1)
        prediction_frame[f"{prefix}_confidence"] = calibrated.max(axis=1)
        for label_id, label_name in labels:
            prediction_frame[
                f"{model_name}_uncalibrated_probability_{label_name}"
            ] = uncalibrated[:, label_id]
            prediction_frame[f"{prefix}_probability_{label_name}"] = calibrated[:, label_id]
            if output.raw_scores is not None:
                prediction_frame[f"{model_name}_raw_score_{label_name}"] = np.asarray(
                    output.raw_scores
                )[:, label_id]

    prediction_frame.to_parquet(predictions_path, index=False)
    thresholds_path = config.metrics_dir / "review_thresholds.json"
    _write_json(
        thresholds_path,
        {
            "schema_version": 1,
            "stage": "policy_calibration",
            "evaluation_split": "validation_policy_calibration",
            "test_evaluated": False,
            "models": threshold_records,
        },
    )
    summary_path = config.metrics_dir / "policy_calibration_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "stage": "policy_calibration",
            "evaluation_split": "validation_policy_calibration",
            "calibration_rows": int(len(policy)),
            "expected_calibration_error_bins": calibration_bins,
            "test_evaluated": False,
            "config_sha256": _sha256(config.config_path),
            "dataset_sha256": _sha256(config.dataset_path),
            "tfidf_model_sha256": _sha256(config.tfidf_model_path),
            "predictions_sha256": _sha256(predictions_path),
            "thresholds_sha256": _sha256(thresholds_path),
            "label_order": [
                {"id": label_id, "label": label_name}
                for label_id, label_name in labels
            ],
            "models": model_summaries,
        },
    )
    return CalibrationBundle(
        summary_path=summary_path,
        thresholds_path=thresholds_path,
        predictions_path=predictions_path,
    )


def _ordered_labels(settings: Mapping[str, Any]) -> list[tuple[int, str]]:
    configured = settings.get("data", {}).get("labels", {})
    labels = sorted(
        (int(label_config["id"]), str(label_name))
        for label_name, label_config in configured.items()
    )
    if not labels or len({label_id for label_id, _ in labels}) != len(labels):
        raise SchemaError("Calibration configuration needs unique integer label IDs")
    return labels


def _validate_model_output(
    model_name: str,
    output: CalibratedModelOutput,
    *,
    expected_label_ids: np.ndarray,
    num_labels: int,
) -> None:
    observed_label_ids = np.asarray(output.label_ids, dtype=np.int64)
    if not np.array_equal(observed_label_ids, expected_label_ids):
        raise RuntimeError(f"{model_name} returned misaligned calibration labels")
    expected_shape = (len(expected_label_ids), num_labels)
    for description, values in (
        ("uncalibrated", output.uncalibrated_probabilities),
        ("calibrated", output.calibrated_probabilities),
    ):
        probabilities = np.asarray(values, dtype=np.float64)
        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"{model_name} returned {description} probabilities with shape "
                f"{probabilities.shape}, expected {expected_shape}"
            )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError(f"{model_name} returned invalid {description} probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(
                f"{model_name} returned {description} rows that do not sum to one"
            )
    if output.raw_scores is not None:
        raw_scores = np.asarray(output.raw_scores, dtype=np.float64)
        if raw_scores.shape != expected_shape or not np.isfinite(raw_scores).all():
            raise RuntimeError(f"{model_name} returned invalid raw calibration scores")


def _calibration_metrics(
    probabilities: np.ndarray,
    label_ids: np.ndarray,
    *,
    bins: int,
) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    row_indices = np.arange(len(label_ids))
    negative_log_likelihood = float(-np.log(clipped[row_indices, label_ids]).mean())
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[label_ids]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == label_ids
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return {
        "negative_log_likelihood": negative_log_likelihood,
        "multiclass_brier_score": brier,
        "expected_calibration_error": float(ece),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _select_review_policy(
    probabilities: np.ndarray,
    label_ids: np.ndarray,
    target: float,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == label_ids
    selected_threshold: float | None = None
    selected_mask = np.zeros(len(label_ids), dtype=bool)
    selected_accuracy: float | None = None
    for threshold in np.unique(confidence):
        mask = confidence >= threshold
        accuracy = float(correct[mask].mean())
        if accuracy >= target and int(mask.sum()) > int(selected_mask.sum()):
            selected_threshold = float(threshold)
            selected_mask = mask
            selected_accuracy = accuracy
    automatic_rows = int(selected_mask.sum())
    return {
        "selective_accuracy_target": target,
        "requirement_met": selected_threshold is not None,
        "threshold": selected_threshold,
        "coverage": float(automatic_rows / len(label_ids)),
        "selective_accuracy": selected_accuracy,
        "automatic_rows": automatic_rows,
        "human_review_rows": int(len(label_ids) - automatic_rows),
    }


def _validate_frozen_inputs(config: CalibrationConfig) -> None:
    if not config.dataset_path.is_file():
        raise SchemaError(f"Calibration dataset does not exist: {config.dataset_path}")
    if not config.tfidf_model_path.is_file():
        raise SchemaError(f"Frozen TF-IDF model does not exist: {config.tfidf_model_path}")
    if not (config.distilbert_run_dir / "model").is_dir():
        raise SchemaError("Transferred DistilBERT model directory does not exist")
    _verify_handoff_checksums(config.distilbert_run_dir)

    identity = _read_json(config.distilbert_run_dir / "run_identity.json")
    summary = _read_json(config.distilbert_run_dir / "training_summary.json")
    dataset_hash = _sha256(config.dataset_path)
    if identity.get("dataset_sha256") != dataset_hash:
        raise SchemaError("DistilBERT run identity and calibration dataset hash differ")
    if summary.get("dataset_sha256") != dataset_hash:
        raise SchemaError("DistilBERT training summary and calibration dataset hash differ")


def _verify_handoff_checksums(run_dir: Path) -> None:
    manifest_path = run_dir / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        raise SchemaError("Transferred DistilBERT handoff lacks SHA256SUMS.txt")
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SchemaError(f"Invalid handoff checksum line {line_number}")
        expected_hash, relative_name = parts
        relative_path = Path(relative_name.strip())
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SchemaError(f"Unsafe handoff checksum path: {relative_name}")
        target = run_dir / relative_path
        if not target.is_file():
            raise SchemaError(f"Handoff checksum target is missing: {relative_name}")
        if _sha256(target) != expected_hash.lower():
            raise SchemaError(f"Handoff checksum failed: {relative_name}")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SchemaError(f"Required calibration input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"Calibration input must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_weights_sha256(model_dir: Path) -> str:
    candidates = [
        path
        for path in (
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise SchemaError(
            f"Expected one DistilBERT weight file in {model_dir}, found {len(candidates)}"
        )
    return _sha256(candidates[0])


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
