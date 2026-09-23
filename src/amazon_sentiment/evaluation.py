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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .data_pipeline import SchemaError
from .inference import LocalDistilBertBackend


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


@dataclass(frozen=True)
class FinalEvaluationConfig:
    """Frozen inputs for the one-time protected test evaluation."""

    config_path: Path
    dataset_path: Path
    distilbert_run_dir: Path
    baseline_model_dir: Path
    calibration_summary_path: Path
    thresholds_path: Path
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
        baseline_model_dir: str | Path,
        calibration_summary_path: str | Path,
        thresholds_path: str | Path,
        metrics_dir: str | Path,
        predictions_dir: str | Path,
    ) -> "FinalEvaluationConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Experiment configuration must be a YAML mapping")
        return cls(
            config_path=resolved_config,
            dataset_path=Path(dataset_path),
            distilbert_run_dir=Path(distilbert_run_dir),
            baseline_model_dir=Path(baseline_model_dir),
            calibration_summary_path=Path(calibration_summary_path),
            thresholds_path=Path(thresholds_path),
            metrics_dir=Path(metrics_dir),
            predictions_dir=Path(predictions_dir),
            settings=settings,
        )


@dataclass(frozen=True)
class FinalModelOutput:
    """Aligned test labels and model probabilities."""

    label_ids: np.ndarray
    uncalibrated_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    raw_scores: np.ndarray | None = None
    token_lengths: np.ndarray | None = None


class FinalEvaluationBackend(Protocol):
    """Runtime boundary for frozen-model test prediction."""

    def predict(
        self,
        config: FinalEvaluationConfig,
        test: pd.DataFrame,
    ) -> Mapping[str, FinalModelOutput]: ...


@dataclass(frozen=True)
class FinalEvaluationBundle:
    """Outputs from the one-time protected test evaluation."""

    metrics_path: Path
    predictions_path: Path
    slice_metrics_path: Path


class LocalTestEvaluationBackend:
    """Generate test predictions from the frozen local model bundles."""

    def predict(
        self,
        config: FinalEvaluationConfig,
        test: pd.DataFrame,
    ) -> Mapping[str, FinalModelOutput]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on local runtime
            raise RuntimeError(
                "DistilBERT test evaluation requires a local PyTorch installation"
            ) from error

        labels = _ordered_labels(config.settings)
        expected_classes = np.array([label_id for label_id, _ in labels])
        label_ids = test["sentiment_id"].to_numpy(dtype=np.int64)
        texts = test["text"].astype(str).tolist()

        majority = _read_json(config.baseline_model_dir / "majority_model.json")
        majority_id = int(majority["predicted_id"])
        majority_probabilities = np.zeros((len(test), len(labels)), dtype=np.float64)
        majority_probabilities[:, majority_id] = 1.0

        tfidf_model = joblib.load(
            config.baseline_model_dir / "tfidf_logistic_regression.joblib"
        )
        tfidf_calibrated_model = joblib.load(
            config.baseline_model_dir / "tfidf_sigmoid_calibrated.joblib"
        )
        for name, model in (
            ("TF-IDF", tfidf_model),
            ("calibrated TF-IDF", tfidf_calibrated_model),
        ):
            observed = np.asarray(getattr(model, "classes_", []))
            if not np.array_equal(observed, expected_classes):
                raise RuntimeError(
                    f"{name} classes {observed.tolist()} do not match "
                    f"{expected_classes.tolist()}"
                )
        tfidf_uncalibrated = np.asarray(
            tfidf_model.predict_proba(texts), dtype=np.float64
        )
        tfidf_calibrated = np.asarray(
            tfidf_calibrated_model.predict_proba(texts), dtype=np.float64
        )
        tfidf_scores = np.asarray(tfidf_model.decision_function(texts), dtype=np.float64)

        model_dir = config.distilbert_run_dir / "model"
        calibration = _read_json(config.distilbert_run_dir / "calibration.json")
        temperature = float(calibration["temperature"])
        if temperature <= 0:
            raise SchemaError("Frozen DistilBERT temperature must be positive")
        runtime = LocalDistilBertBackend(
            model_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        evaluation_settings = config.settings.get("evaluation", {})
        batch_size = int(evaluation_settings.get("distilbert_inference_batch_size", 16))
        max_length = int(
            config.settings.get("models", {})
            .get("distilbert", {})
            .get("max_length", 256)
        )
        distilbert_logits, token_lengths = runtime.predict_many(
            texts,
            max_length=max_length,
            batch_size=batch_size,
        )
        return {
            "majority": FinalModelOutput(
                label_ids,
                majority_probabilities,
                majority_probabilities,
            ),
            "tfidf_logistic_regression": FinalModelOutput(
                label_ids,
                tfidf_uncalibrated,
                tfidf_calibrated,
                raw_scores=tfidf_scores,
            ),
            "distilbert": FinalModelOutput(
                label_ids,
                _softmax(distilbert_logits),
                _softmax(distilbert_logits / temperature),
                raw_scores=distilbert_logits,
                token_lengths=token_lengths,
            ),
        }


def evaluate_test(
    config: FinalEvaluationConfig,
    *,
    backend: FinalEvaluationBackend | None = None,
) -> FinalEvaluationBundle:
    """Evaluate frozen models using only the protected test split."""

    metrics_path = config.metrics_dir / "final_test_metrics.json"
    if metrics_path.exists():
        raise SchemaError(
            f"Final test result already exists and will not be overwritten: {metrics_path}"
        )
    _validate_final_evaluation_inputs(config)
    required_slices = list(
        config.settings.get("evaluation", {}).get("required_slices", [])
    )
    required = {
        "text",
        "sentiment_id",
        "sentiment_label",
        "split",
        "record_id",
        *_slice_source_columns(required_slices),
    }
    test = pd.read_parquet(
        config.dataset_path,
        columns=sorted(required),
        filters=[("split", "==", "test")],
    )
    missing = required.difference(test.columns)
    if missing:
        raise SchemaError(f"Test dataset is missing fields: {sorted(missing)}")
    if test.empty:
        raise SchemaError("Final evaluation selected zero test rows")
    unexpected = set(test["split"].astype(str)).difference({"test"})
    if unexpected:
        raise SchemaError(f"Final evaluation read non-test splits: {sorted(unexpected)}")
    expected_rows = int(
        config.settings.get("split", {}).get("caps", {}).get("test_total", 0)
    )
    if expected_rows and len(test) != expected_rows:
        raise SchemaError(f"Final evaluation expected {expected_rows} rows, found {len(test)}")

    test = test.reset_index(drop=True)
    outputs = (backend or LocalTestEvaluationBackend()).predict(config, test)
    expected_models = {"majority", "tfidf_logistic_regression", "distilbert"}
    if set(outputs) != expected_models:
        raise RuntimeError(
            f"Test backend returned models {sorted(outputs)}, "
            f"expected {sorted(expected_models)}"
        )
    labels = _ordered_labels(config.settings)
    expected_label_ids = test["sentiment_id"].to_numpy(dtype=np.int64)
    for model_name, output in outputs.items():
        _validate_model_output(
            model_name,
            output,
            expected_label_ids=expected_label_ids,
            num_labels=len(labels),
        )

    thresholds = _read_json(config.thresholds_path)["models"]
    calibration_bins = int(
        config.settings.get("evaluation", {}).get(
            "expected_calibration_error_bins", 10
        )
    )
    model_metrics: dict[str, Any] = {}
    predicted_ids: dict[str, np.ndarray] = {}
    prediction_frame = test[["record_id", "sentiment_id", "sentiment_label"]].copy()
    for model_name in sorted(outputs):
        output = outputs[model_name]
        uncalibrated = np.asarray(output.uncalibrated_probabilities, dtype=np.float64)
        calibrated = np.asarray(output.calibrated_probabilities, dtype=np.float64)
        predictions = calibrated.argmax(axis=1)
        predicted_ids[model_name] = predictions
        model_record: dict[str, Any] = {
            "classification": _classification_metrics(
                expected_label_ids, predictions, labels
            )
        }
        prediction_frame[f"{model_name}_predicted_id"] = predictions
        prediction_frame[f"{model_name}_confidence"] = calibrated.max(axis=1)
        for label_id, label_name in labels:
            prediction_frame[f"{model_name}_probability_{label_name}"] = calibrated[
                :, label_id
            ]
        if model_name != "majority":
            model_record["uncalibrated"] = _calibration_metrics(
                uncalibrated, expected_label_ids, bins=calibration_bins
            )
            model_record["calibrated"] = _calibration_metrics(
                calibrated, expected_label_ids, bins=calibration_bins
            )
            threshold = float(thresholds[model_name]["threshold"])
            model_record["review_policy"] = _evaluate_frozen_threshold(
                calibrated, expected_label_ids, threshold, labels
            )
            prediction_frame[f"{model_name}_automatic_route"] = (
                calibrated.max(axis=1) >= threshold
            )
        model_metrics[model_name] = model_record

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.predictions_dir / "final_test_predictions.parquet"
    prediction_frame.to_parquet(predictions_path, index=False)
    slice_metrics_path = config.metrics_dir / "final_test_slice_metrics.csv"
    slice_metrics = _build_slice_metrics(
        test,
        outputs,
        predicted_ids,
        labels,
        thresholds,
        required_slices=required_slices,
        minimum_rows=int(
            config.settings.get("evaluation", {}).get("minimum_slice_rows", 200)
        ),
    )
    slice_metrics.to_csv(slice_metrics_path, index=False)
    bootstrap = _bootstrap_macro_f1(
        expected_label_ids,
        predicted_ids,
        label_ids=[label_id for label_id, _ in labels],
        resamples=int(
            config.settings.get("evaluation", {}).get("bootstrap_resamples", 1000)
        ),
        confidence_level=float(
            config.settings.get("evaluation", {}).get("confidence_interval", 0.95)
        ),
        seed=int(config.settings.get("project", {}).get("seed", 42)),
    )
    _write_json(
        metrics_path,
        {
            "schema_version": 1,
            "stage": "final_test_evaluation",
            "evaluation_split": "test",
            "test_rows": int(len(test)),
            "test_evaluated": True,
            "config_sha256": _sha256(config.config_path),
            "dataset_sha256": _sha256(config.dataset_path),
            "calibration_summary_sha256": _sha256(config.calibration_summary_path),
            "thresholds_sha256": _sha256(config.thresholds_path),
            "predictions_sha256": _sha256(predictions_path),
            "slice_metrics_sha256": _sha256(slice_metrics_path),
            "label_order": [
                {"id": label_id, "label": label_name}
                for label_id, label_name in labels
            ],
            "models": model_metrics,
            "bootstrap": bootstrap,
        },
    )
    return FinalEvaluationBundle(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        slice_metrics_path=slice_metrics_path,
    )


def _slice_source_columns(required_slices: list[str]) -> set[str]:
    columns = {
        "category": "source_category",
        "test_period_half": "timestamp",
        "verified_purchase": "verified_purchase",
        "helpful_vote_zero_vs_positive": "helpful_vote",
        "price_band_with_missing": "price",
        "rating_number_band": "rating_number",
    }
    supported = {*columns, "token_length_quartile"}
    unexpected = set(required_slices).difference(supported)
    if unexpected:
        raise SchemaError(f"Unsupported required slices: {sorted(unexpected)}")
    return {columns[name] for name in required_slices if name in columns}


def _build_slice_metrics(
    test: pd.DataFrame,
    outputs: Mapping[str, FinalModelOutput],
    predictions: Mapping[str, np.ndarray],
    labels: list[tuple[int, str]],
    thresholds: Mapping[str, Any],
    *,
    required_slices: list[str],
    minimum_rows: int,
) -> pd.DataFrame:
    if minimum_rows < 1:
        raise SchemaError("minimum_slice_rows must be positive")
    assignments: dict[str, pd.Series] = {}
    if "category" in required_slices:
        assignments["category"] = test["source_category"].astype(str)
    if "test_period_half" in required_slices:
        timestamps = pd.to_datetime(test["timestamp"], utc=True)
        order = timestamps.rank(method="first")
        assignments["test_period_half"] = pd.Series(
            np.where(order <= len(test) / 2, "earlier_half", "later_half")
        )
    if "verified_purchase" in required_slices:
        assignments["verified_purchase"] = test["verified_purchase"].map(
            {True: "verified", False: "not_verified"}
        )
    if "helpful_vote_zero_vs_positive" in required_slices:
        assignments["helpful_vote_zero_vs_positive"] = pd.Series(
            np.where(test["helpful_vote"].fillna(0).astype(float) > 0, "positive", "zero")
        )
    if "token_length_quartile" in required_slices:
        token_lengths = outputs["distilbert"].token_lengths
        if token_lengths is None or len(token_lengths) != len(test):
            raise RuntimeError("DistilBERT did not return aligned token lengths")
        assignments["token_length_quartile"] = _quartile_bands(
            pd.Series(token_lengths), include_missing=False
        )
    if "price_band_with_missing" in required_slices:
        assignments["price_band_with_missing"] = _quartile_bands(
            pd.to_numeric(test["price"], errors="coerce"), include_missing=True
        )
    if "rating_number_band" in required_slices:
        assignments["rating_number_band"] = _quartile_bands(
            pd.to_numeric(test["rating_number"], errors="coerce"),
            include_missing=True,
        )

    expected = test["sentiment_id"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    for slice_type, values in assignments.items():
        for slice_value in sorted(values.dropna().astype(str).unique()):
            selected = values.astype(str).eq(slice_value).to_numpy()
            rows = int(selected.sum())
            if rows < minimum_rows:
                continue
            for model_name in sorted(predictions):
                metrics = _classification_metrics(
                    expected[selected], predictions[model_name][selected], labels
                )
                record: dict[str, Any] = {
                    "model": model_name,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "rows": rows,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "coverage": None,
                    "selective_accuracy": None,
                }
                if model_name != "majority":
                    probabilities = np.asarray(
                        outputs[model_name].calibrated_probabilities
                    )[selected]
                    policy = _evaluate_frozen_threshold(
                        probabilities,
                        expected[selected],
                        float(thresholds[model_name]["threshold"]),
                        labels,
                    )
                    record["coverage"] = policy["coverage"]
                    record["selective_accuracy"] = policy["selective_accuracy"]
                records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "model",
            "slice_type",
            "slice_value",
            "rows",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "coverage",
            "selective_accuracy",
        ],
    )


def _quartile_bands(values: pd.Series, *, include_missing: bool) -> pd.Series:
    result = pd.Series(index=values.index, dtype="object")
    present = values.notna()
    if present.any():
        ranked = values[present].rank(method="first")
        result.loc[present] = pd.qcut(
            ranked,
            q=4,
            labels=["q1_low", "q2", "q3", "q4_high"],
        ).astype(str)
    if include_missing:
        result.loc[~present] = "missing"
    return result


def _classification_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    labels: list[tuple[int, str]],
) -> dict[str, Any]:
    label_ids = [label_id for label_id, _ in labels]
    precision, recall, class_f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=label_ids,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                expected,
                predicted,
                labels=label_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": [
            {
                "id": label_id,
                "label": label_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, (label_id, label_name) in enumerate(labels)
        ],
        "confusion_matrix": {
            "label_order": [label_name for _, label_name in labels],
            "values": confusion_matrix(
                expected, predicted, labels=label_ids
            ).tolist(),
        },
    }


def _evaluate_frozen_threshold(
    probabilities: np.ndarray,
    label_ids: np.ndarray,
    threshold: float,
    labels: list[tuple[int, str]],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    automatic = confidence >= threshold
    automatic_rows = int(automatic.sum())
    selective_accuracy = (
        float((predictions[automatic] == label_ids[automatic]).mean())
        if automatic_rows
        else None
    )
    per_class = []
    for label_id, label_name in labels:
        class_rows = label_ids == label_id
        class_row_count = int(class_rows.sum())
        class_automatic = class_rows & automatic
        routed = int(class_automatic.sum())
        per_class.append(
            {
                "id": label_id,
                "label": label_name,
                "rows": class_row_count,
                "automatic_rows": routed,
                "coverage": (
                    float(routed / class_row_count) if class_row_count else None
                ),
                "selective_accuracy": (
                    float(
                        (
                            predictions[class_automatic]
                            == label_ids[class_automatic]
                        ).mean()
                    )
                    if routed
                    else None
                ),
            }
        )
    return {
        "threshold": threshold,
        "automatic_rows": automatic_rows,
        "human_review_rows": int(len(label_ids) - automatic_rows),
        "coverage": float(automatic_rows / len(label_ids)),
        "selective_accuracy": selective_accuracy,
        "per_true_class": per_class,
    }


def _bootstrap_macro_f1(
    expected: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    label_ids: list[int],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if resamples < 1 or not 0 < confidence_level < 1:
        raise SchemaError("Bootstrap settings must be positive with confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(expected == label) for label in label_ids]
    samples: dict[str, list[float]] = {name: [] for name in predictions}
    differences: list[float] = []
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        scores = {
            name: float(
                f1_score(
                    expected[indices],
                    values[indices],
                    labels=label_ids,
                    average="macro",
                    zero_division=0,
                )
            )
            for name, values in predictions.items()
        }
        for name, score in scores.items():
            samples[name].append(score)
        differences.append(scores["tfidf_logistic_regression"] - scores["distilbert"])
    alpha = (1.0 - confidence_level) / 2.0

    def interval(point: float, values: list[float]) -> dict[str, float]:
        return {
            "point_estimate": point,
            "confidence_level": confidence_level,
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }

    points = {
        name: float(
            f1_score(
                expected,
                values,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        )
        for name, values in predictions.items()
    }
    return {
        "resamples": resamples,
        "method": "paired_stratified_percentile",
        "macro_f1": {
            name: interval(points[name], samples[name]) for name in sorted(predictions)
        },
        "tfidf_minus_distilbert_macro_f1": interval(
            points["tfidf_logistic_regression"] - points["distilbert"],
            differences,
        ),
    }


def _validate_final_evaluation_inputs(config: FinalEvaluationConfig) -> None:
    if not config.dataset_path.is_file():
        raise SchemaError(f"Final evaluation dataset does not exist: {config.dataset_path}")
    summary = _read_json(config.calibration_summary_path)
    thresholds = _read_json(config.thresholds_path)
    if summary.get("test_evaluated") is not False:
        raise SchemaError("Calibration summary does not preserve the untouched test gate")
    if thresholds.get("test_evaluated") is not False:
        raise SchemaError("Frozen thresholds do not preserve the untouched test gate")
    if summary.get("config_sha256") != _sha256(config.config_path):
        raise SchemaError("Calibration and final evaluation configuration hashes differ")
    if summary.get("dataset_sha256") != _sha256(config.dataset_path):
        raise SchemaError("Calibration and final evaluation dataset hashes differ")
    if summary.get("thresholds_sha256") != _sha256(config.thresholds_path):
        raise SchemaError("Frozen threshold checksum does not match calibration summary")
    tfidf_model_path = config.baseline_model_dir / "tfidf_logistic_regression.joblib"
    if summary.get("tfidf_model_sha256") != _sha256(tfidf_model_path):
        raise SchemaError("Frozen TF-IDF model checksum does not match calibration summary")
    _verify_handoff_checksums(config.distilbert_run_dir)

    threshold_models = thresholds.get("models", {})
    expected_threshold_models = {"distilbert", "tfidf_logistic_regression"}
    if set(threshold_models) != expected_threshold_models:
        raise SchemaError("Frozen thresholds do not contain both probabilistic models")
    for model_name in sorted(expected_threshold_models):
        record = threshold_models[model_name]
        threshold = record.get("threshold")
        if record.get("requirement_met") is not True or not isinstance(
            threshold, (int, float)
        ):
            raise SchemaError(f"Frozen threshold is not approved for {model_name}")
        if not 0 <= float(threshold) <= 1:
            raise SchemaError(f"Frozen threshold is outside [0, 1] for {model_name}")

    dataset_hash = _sha256(config.dataset_path)
    distilbert_calibration = _read_json(
        config.distilbert_run_dir / "calibration.json"
    )
    if distilbert_calibration.get("test_evaluated") is not False:
        raise SchemaError("DistilBERT calibration has an invalid test gate")
    if distilbert_calibration.get("dataset_sha256") != dataset_hash:
        raise SchemaError("DistilBERT calibration dataset checksum differs")
    if distilbert_calibration.get("model_sha256") != _model_weights_sha256(
        config.distilbert_run_dir / "model"
    ):
        raise SchemaError("DistilBERT calibration model checksum differs")
    temperature = distilbert_calibration.get("temperature")
    if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
        raise SchemaError("DistilBERT calibration temperature is invalid")

    tfidf_calibration = _read_json(
        config.baseline_model_dir / "tfidf_sigmoid_calibration.json"
    )
    calibrated_tfidf_path = (
        config.baseline_model_dir / "tfidf_sigmoid_calibrated.joblib"
    )
    if tfidf_calibration.get("test_evaluated") is not False:
        raise SchemaError("TF-IDF calibration has an invalid test gate")
    if tfidf_calibration.get("dataset_sha256") != dataset_hash:
        raise SchemaError("TF-IDF calibration dataset checksum differs")
    if tfidf_calibration.get("base_model_sha256") != _sha256(tfidf_model_path):
        raise SchemaError("TF-IDF calibration base model checksum differs")
    if tfidf_calibration.get("calibrated_model_sha256") != _sha256(
        calibrated_tfidf_path
    ):
        raise SchemaError("Frozen calibrated TF-IDF model checksum differs")
    if not (config.baseline_model_dir / "majority_model.json").is_file():
        raise SchemaError("Frozen majority model does not exist")


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
