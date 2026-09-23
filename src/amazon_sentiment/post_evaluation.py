"""Build post-test efficiency, visual and recommendation evidence.

This module consumes the frozen final-test outputs. It never rewrites the
one-time test metrics. Public outputs contain aggregate values only.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import joblib
import numpy as np
import pandas as pd
import yaml

from .data_pipeline import SchemaError
from .inference import LocalDistilBertBackend


MODEL_LABELS = {
    "majority": "Majority",
    "tfidf_logistic_regression": "TF-IDF",
    "distilbert": "DistilBERT",
}
MODEL_COLOURS = {
    "majority": "#7a7a7a",
    "tfidf_logistic_regression": "#2b6cb0",
    "distilbert": "#c05621",
}
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(frozen=True)
class EvidenceConfig:
    """Resolved inputs and protocol for post-test evidence generation."""

    protocol_path: Path
    experiment_config_path: Path
    dataset_path: Path
    final_metrics_path: Path
    slice_metrics_path: Path
    predictions_path: Path
    distilbert_run_dir: Path
    baseline_model_dir: Path
    metrics_dir: Path
    figures_dir: Path
    settings: Mapping[str, Any]

    @classmethod
    def from_yaml(
        cls,
        protocol_path: str | Path,
        *,
        experiment_config_path: str | Path,
        dataset_path: str | Path,
        final_metrics_path: str | Path,
        slice_metrics_path: str | Path,
        predictions_path: str | Path,
        distilbert_run_dir: str | Path,
        baseline_model_dir: str | Path,
        metrics_dir: str | Path,
        figures_dir: str | Path,
    ) -> "EvidenceConfig":
        resolved = Path(protocol_path)
        settings = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Evaluation evidence protocol must be a YAML mapping")
        latency = settings.get("latency", {})
        positive_fields = {
            "latency.warmup_runs": latency.get("warmup_runs"),
            "latency.measured_runs": latency.get("measured_runs"),
            "latency.deterministic_sample_rows": latency.get(
                "deterministic_sample_rows"
            ),
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or value < 1:
                raise SchemaError(f"{name} must be a positive integer")
        if latency.get("device") != "cpu" or latency.get("batch_size") != 1:
            raise SchemaError("Evidence latency protocol must use CPU batch size one")
        return cls(
            protocol_path=resolved,
            experiment_config_path=Path(experiment_config_path),
            dataset_path=Path(dataset_path),
            final_metrics_path=Path(final_metrics_path),
            slice_metrics_path=Path(slice_metrics_path),
            predictions_path=Path(predictions_path),
            distilbert_run_dir=Path(distilbert_run_dir),
            baseline_model_dir=Path(baseline_model_dir),
            metrics_dir=Path(metrics_dir),
            figures_dir=Path(figures_dir),
            settings=settings,
        )


@dataclass(frozen=True)
class EvaluationEvidenceBundle:
    """Paths produced by the post-test evidence stage."""

    benchmark_path: Path
    recommendation_path: Path
    figure_paths: tuple[Path, ...]


class InferenceBenchmarkBackend(Protocol):
    """Boundary used to test evidence orchestration without loading models."""

    def benchmark(
        self,
        config: EvidenceConfig,
        texts: list[str],
    ) -> Mapping[str, Any]: ...


class LocalInferenceBenchmarkBackend:
    """Measure warmed batch-one prediction on the local CPU."""

    def benchmark(
        self,
        config: EvidenceConfig,
        texts: list[str],
    ) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Local latency benchmarking requires PyTorch and Transformers"
            ) from error

        protocol = config.settings["latency"]
        warmups = int(protocol["warmup_runs"])
        repetitions = int(protocol["measured_runs"])
        experiment = yaml.safe_load(
            config.experiment_config_path.read_text(encoding="utf-8")
        )
        max_length = int(experiment["models"]["distilbert"]["max_length"])

        tfidf_path = config.baseline_model_dir / "tfidf_sigmoid_calibrated.joblib"
        started = time.perf_counter()
        tfidf = joblib.load(tfidf_path)
        tfidf_load_seconds = time.perf_counter() - started

        def predict_tfidf(text: str) -> np.ndarray:
            return np.asarray(tfidf.predict_proba([text]), dtype=np.float64)[0]

        tfidf_timings = _time_batch_one_predictions(
            predict_tfidf,
            texts,
            warmups=warmups,
            repetitions=repetitions,
        )

        model_dir = config.distilbert_run_dir / "model"
        started = time.perf_counter()
        runtime = LocalDistilBertBackend(model_dir, device="cpu")
        distilbert_load_seconds = time.perf_counter() - started
        calibration = _read_json(config.distilbert_run_dir / "calibration.json")
        temperature = float(calibration["temperature"])
        if temperature <= 0:
            raise SchemaError("DistilBERT temperature must be positive")

        def predict_distilbert(text: str) -> np.ndarray:
            output = runtime.predict(text, max_length=max_length)
            logits = np.asarray(output.logits, dtype=np.float64)[np.newaxis, :]
            return _softmax(logits / temperature)[0]

        distilbert_timings = _time_batch_one_predictions(
            predict_distilbert,
            texts,
            warmups=warmups,
            repetitions=repetitions,
        )
        tfidf_size = _path_size(tfidf_path)
        distilbert_paths = [model_dir, config.distilbert_run_dir / "calibration.json"]
        distilbert_size = sum(_path_size(path) for path in distilbert_paths)
        return {
            "models": {
                "tfidf_logistic_regression": {
                    "serving_artifact_bytes": tfidf_size,
                    "serving_artifact_mebibytes": tfidf_size / (1024**2),
                    "load_seconds": tfidf_load_seconds,
                    "latency_ms": _summarise_timings(tfidf_timings),
                },
                "distilbert": {
                    "serving_artifact_bytes": distilbert_size,
                    "serving_artifact_mebibytes": distilbert_size / (1024**2),
                    "load_seconds": distilbert_load_seconds,
                    "latency_ms": _summarise_timings(distilbert_timings),
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": _cpu_model(),
                "logical_cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "transformers": _package_version("transformers"),
                "scikit_learn": _package_version("scikit-learn"),
            },
        }


def build_evaluation_evidence(
    config: EvidenceConfig,
    *,
    benchmark_backend: InferenceBenchmarkBackend | None = None,
) -> EvaluationEvidenceBundle:
    """Build efficiency, plot and recommendation evidence."""

    final_metrics, predictions, slices = _validate_inputs(config)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = int(config.settings["latency"]["deterministic_sample_rows"])
    texts = _deterministic_latency_texts(
        config.dataset_path,
        predictions["record_id"].astype(str),
        rows=sample_rows,
    )
    benchmark_values = (benchmark_backend or LocalInferenceBenchmarkBackend()).benchmark(
        config, texts
    )
    benchmark = {
        "schema_version": 1,
        "stage": "post_test_inference_benchmark",
        "test_metrics_sha256": _sha256(config.final_metrics_path),
        "predictions_sha256": _sha256(config.predictions_path),
        "dataset_sha256": _sha256(config.dataset_path),
        "protocol_sha256": _sha256(config.protocol_path),
        "protocol": config.settings["latency"],
        "sample_selection": "sha256(record_id), ascending",
        "sample_rows": len(texts),
        **benchmark_values,
    }
    benchmark_path = config.metrics_dir / "inference_benchmark.json"
    _write_json(benchmark_path, benchmark)

    figure_paths = _generate_figures(
        final_metrics,
        predictions,
        slices,
        config.figures_dir,
    )
    recommendation = _deployment_recommendation(final_metrics, benchmark)
    recommendation["figure_sha256"] = {
        path.name: _sha256(path) for path in figure_paths
    }
    recommendation_path = config.metrics_dir / "deployment_recommendation.json"
    _write_json(recommendation_path, recommendation)
    return EvaluationEvidenceBundle(
        benchmark_path=benchmark_path,
        recommendation_path=recommendation_path,
        figure_paths=tuple(figure_paths),
    )


def _validate_inputs(
    config: EvidenceConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required_files = [
        config.protocol_path,
        config.experiment_config_path,
        config.dataset_path,
        config.final_metrics_path,
        config.slice_metrics_path,
        config.predictions_path,
        config.distilbert_run_dir / "model" / "config.json",
        config.distilbert_run_dir / "calibration.json",
        config.baseline_model_dir / "tfidf_sigmoid_calibrated.joblib",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SchemaError(f"Post-test evidence inputs do not exist: {missing}")
    final_metrics = _read_json(config.final_metrics_path)
    if final_metrics.get("test_evaluated") is not True:
        raise SchemaError("Final metrics do not record a completed test evaluation")
    expected_hashes = {
        "dataset_sha256": config.dataset_path,
        "predictions_sha256": config.predictions_path,
        "slice_metrics_sha256": config.slice_metrics_path,
    }
    for field, path in expected_hashes.items():
        if final_metrics.get(field) != _sha256(path):
            raise SchemaError(f"Frozen {field} does not match {path}")
    predictions = pd.read_parquet(config.predictions_path)
    required_prediction_columns = {
        "record_id",
        "sentiment_id",
        "sentiment_label",
        "distilbert_predicted_id",
        "distilbert_confidence",
        "distilbert_automatic_route",
        "tfidf_logistic_regression_predicted_id",
        "tfidf_logistic_regression_confidence",
        "tfidf_logistic_regression_automatic_route",
    }
    missing_columns = required_prediction_columns.difference(predictions.columns)
    if missing_columns:
        raise SchemaError(
            f"Final predictions are missing fields: {sorted(missing_columns)}"
        )
    if predictions["record_id"].duplicated().any():
        raise SchemaError("Final predictions contain duplicate record IDs")
    if len(predictions) != int(final_metrics.get("test_rows", -1)):
        raise SchemaError("Final prediction row count differs from final metrics")
    slices = pd.read_csv(config.slice_metrics_path)
    required_slice_columns = {
        "model",
        "slice_type",
        "slice_value",
        "rows",
        "macro_f1",
    }
    if required_slice_columns.difference(slices.columns):
        raise SchemaError("Slice metrics do not contain the reporting schema")
    return final_metrics, predictions, slices


def _deterministic_latency_texts(
    dataset_path: Path,
    allowed_record_ids: pd.Series,
    *,
    rows: int,
) -> list[str]:
    test = pd.read_parquet(
        dataset_path,
        columns=["record_id", "text", "split"],
        filters=[("split", "==", "test")],
    )
    allowed = set(allowed_record_ids)
    test = test[test["record_id"].astype(str).isin(allowed)].copy()
    if len(test) != len(allowed):
        raise SchemaError("Latency inputs do not align with frozen predictions")
    test["selection_key"] = test["record_id"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    selected = test.sort_values(["selection_key", "record_id"]).head(rows)
    if len(selected) < rows:
        raise SchemaError(f"Latency protocol requests {rows} rows, found {len(selected)}")
    return selected["text"].astype(str).tolist()


def _time_batch_one_predictions(
    predict: Any,
    texts: list[str],
    *,
    warmups: int,
    repetitions: int,
) -> list[float]:
    if not texts:
        raise ValueError("Latency benchmark requires text inputs")
    for index in range(warmups):
        probabilities = np.asarray(predict(texts[index % len(texts)]))
        _validate_probabilities(probabilities)
    timings = []
    for index in range(repetitions):
        started = time.perf_counter_ns()
        probabilities = np.asarray(predict(texts[index % len(texts)]))
        elapsed = time.perf_counter_ns() - started
        _validate_probabilities(probabilities)
        timings.append(elapsed / 1_000_000)
    return timings


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise RuntimeError("Benchmark predictor returned invalid probabilities")
    if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("Benchmark probabilities do not sum to one")


def _summarise_timings(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _generate_figures(
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    slices: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = [
        output_dir / "model_comparison.svg",
        output_dir / "confusion_matrices.svg",
        output_dir / "calibration_comparison.svg",
        output_dir / "routing_policy.svg",
        output_dir / "slice_macro_f1_ranges.svg",
    ]
    _model_comparison_figure(metrics, paths[0])
    _confusion_matrix_figure(metrics, paths[1])
    _calibration_figure(metrics, paths[2])
    _routing_figure(metrics, predictions, paths[3])
    _slice_figure(slices, paths[4])
    return paths


def _model_comparison_figure(metrics: Mapping[str, Any], path: Path) -> None:
    width, height = 900, 520
    left, top, chart_width, chart_height = 100, 90, 740, 330
    models = ["majority", "tfidf_logistic_regression", "distilbert"]
    scores = [metrics["models"][name]["classification"]["macro_f1"] for name in models]
    intervals = [metrics["bootstrap"]["macro_f1"][name] for name in models]
    body = [_svg_title("Final-test macro-F1 with 95% bootstrap intervals", width)]
    body.extend(_axes(left, top, chart_width, chart_height, y_max=0.8, y_label="Macro-F1"))
    bar_width = 130
    gap = 100
    for index, (name, score, interval) in enumerate(zip(models, scores, intervals)):
        x = left + 70 + index * (bar_width + gap)
        y = top + chart_height * (1 - score / 0.8)
        body.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{top + chart_height - y:.1f}" '
            f'fill="{MODEL_COLOURS[name]}" rx="4"/>'
        )
        lower_y = top + chart_height * (1 - float(interval["lower"]) / 0.8)
        upper_y = top + chart_height * (1 - float(interval["upper"]) / 0.8)
        centre = x + bar_width / 2
        body.append(
            f'<line x1="{centre}" y1="{upper_y:.1f}" x2="{centre}" y2="{lower_y:.1f}" stroke="#111" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{centre - 9}" y1="{upper_y:.1f}" x2="{centre + 9}" y2="{upper_y:.1f}" stroke="#111" stroke-width="2"/>'
        )
        body.append(_text(centre, y - 12, f"{score:.3f}", anchor="middle", weight="700"))
        body.append(_text(centre, top + chart_height + 35, MODEL_LABELS[name], anchor="middle"))
    difference = -float(metrics["bootstrap"]["tfidf_minus_distilbert_macro_f1"]["point_estimate"])
    body.append(_text(width / 2, 485, f"DistilBERT advantage over TF-IDF: {difference:.3f}", anchor="middle", size=17))
    _write_svg(path, width, height, body)


def _confusion_matrix_figure(metrics: Mapping[str, Any], path: Path) -> None:
    width, height = 980, 500
    body = [_svg_title("Final-test confusion matrices, row-normalised by true class", width)]
    for panel, name in enumerate(["tfidf_logistic_regression", "distilbert"]):
        matrix = np.asarray(
            metrics["models"][name]["classification"]["confusion_matrix"]["values"],
            dtype=float,
        )
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
        origin_x = 145 + panel * 480
        origin_y = 145
        cell = 90
        body.append(_text(origin_x + 135, 92, MODEL_LABELS[name], anchor="middle", size=20, weight="700"))
        for index, label in SENTIMENT_LABELS.items():
            body.append(_text(origin_x + index * cell + 45, 125, label.title(), anchor="middle", size=14))
            body.append(_text(origin_x - 15, origin_y + index * cell + 50, label.title(), anchor="end", size=14))
        for row in range(3):
            for column in range(3):
                value = float(normalised[row, column])
                shade = int(245 - 155 * value)
                colour = f"rgb({shade},{shade + 10},{255})"
                x = origin_x + column * cell
                y = origin_y + row * cell
                body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{colour}" stroke="#fff"/>')
                body.append(_text(x + 45, y + 43, f"{value:.1%}", anchor="middle", size=16, weight="700"))
                body.append(_text(x + 45, y + 65, f"n={int(matrix[row, column])}", anchor="middle", size=12))
    body.append(_text(25, 290, "True label", size=15, weight="700", rotate=-90))
    body.append(_text(width / 2, 475, "Predicted label", anchor="middle", size=15, weight="700"))
    _write_svg(path, width, height, body)


def _calibration_figure(metrics: Mapping[str, Any], path: Path) -> None:
    width, height = 1000, 560
    body = [_svg_title("Calibration metrics before and after calibration", width)]
    metric_fields = [
        ("negative_log_likelihood", "Negative log-likelihood", 0.8),
        ("multiclass_brier_score", "Multiclass Brier score", 0.5),
        ("expected_calibration_error", "Expected calibration error", 0.12),
    ]
    for panel, (field, label, maximum) in enumerate(metric_fields):
        origin_x = 75 + panel * 315
        top = 130
        chart_height = 300
        body.append(_text(origin_x + 120, 102, label, anchor="middle", size=16, weight="700"))
        body.extend(_small_axes(origin_x, top, 240, chart_height, maximum))
        bars = []
        for model in ["tfidf_logistic_regression", "distilbert"]:
            bars.extend(
                [
                    (model, "Before", float(metrics["models"][model]["uncalibrated"][field])),
                    (model, "After", float(metrics["models"][model]["calibrated"][field])),
                ]
            )
        for index, (model, state, value) in enumerate(bars):
            x = origin_x + 18 + index * 55
            y = top + chart_height * (1 - value / maximum)
            opacity = "0.45" if state == "Before" else "1.0"
            body.append(
                f'<rect x="{x}" y="{y:.1f}" width="38" height="{top + chart_height - y:.1f}" '
                f'fill="{MODEL_COLOURS[model]}" fill-opacity="{opacity}"/>'
            )
            body.append(_text(x + 19, y - 7, f"{value:.3f}", anchor="middle", size=11))
            body.append(_text(x + 19, top + chart_height + 22, state[0], anchor="middle", size=12))
        body.append(_text(origin_x + 50, 485, "TF-IDF", anchor="middle", size=13))
        body.append(_text(origin_x + 160, 485, "DistilBERT", anchor="middle", size=13))
    body.append(_text(width / 2, 535, "B = before; A = after. Lower is better for all three metrics.", anchor="middle", size=14))
    _write_svg(path, width, height, body)


def _routing_figure(
    metrics: Mapping[str, Any], predictions: pd.DataFrame, path: Path
) -> None:
    width, height = 900, 560
    left, top, chart_width, chart_height = 105, 90, 700, 380
    body = [_svg_title("Test risk-coverage curves and frozen operating points", width)]
    body.extend(_xy_axes(left, top, chart_width, chart_height, "Coverage", "Selective accuracy"))
    target_y = top + chart_height * (1 - (0.90 - 0.5) / 0.5)
    body.append(f'<line x1="{left}" y1="{target_y:.1f}" x2="{left + chart_width}" y2="{target_y:.1f}" stroke="#777" stroke-dasharray="7 5"/>')
    body.append(_text(left + chart_width - 5, target_y - 8, "90% target", anchor="end", size=13))
    true_ids = predictions["sentiment_id"].to_numpy()
    for model in ["tfidf_logistic_regression", "distilbert"]:
        confidence = predictions[f"{model}_confidence"].to_numpy(dtype=float)
        predicted = predictions[f"{model}_predicted_id"].to_numpy(dtype=int)
        points = []
        for threshold in np.quantile(confidence, np.linspace(0, 0.99, 80)):
            selected = confidence >= threshold
            coverage = float(selected.mean())
            accuracy = float((predicted[selected] == true_ids[selected]).mean())
            points.append((coverage, accuracy))
        coordinates = " ".join(
            f"{left + coverage * chart_width:.1f},{top + chart_height * (1 - max(0.0, accuracy - 0.5) / 0.5):.1f}"
            for coverage, accuracy in points
        )
        body.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{MODEL_COLOURS[model]}" stroke-width="3"/>'
        )
        policy = metrics["models"][model]["review_policy"]
        px = left + float(policy["coverage"]) * chart_width
        py = top + chart_height * (1 - (float(policy["selective_accuracy"]) - 0.5) / 0.5)
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{MODEL_COLOURS[model]}" stroke="#fff" stroke-width="2"/>')
        body.append(_text(px + 10, py - 10, MODEL_LABELS[model], size=13, weight="700"))
    body.append(_text(width / 2, 535, "Curves use final-test predictions for diagnosis; thresholds were frozen on validation data.", anchor="middle", size=13))
    _write_svg(path, width, height, body)


def _slice_figure(slices: pd.DataFrame, path: Path) -> None:
    width, height = 1000, 720
    left, top, chart_width = 255, 80, 650
    body = [_svg_title("Within-slice macro-F1 ranges", width)]
    selected = slices[slices["model"].isin(["tfidf_logistic_regression", "distilbert"])]
    slice_types = sorted(selected["slice_type"].astype(str).unique())
    row_height = 72
    for row, slice_type in enumerate(slice_types):
        centre_y = top + row * row_height + 34
        body.append(_text(left - 18, centre_y + 4, slice_type.replace("_", " "), anchor="end", size=14))
        for offset, model in [(-10, "tfidf_logistic_regression"), (10, "distilbert")]:
            values = selected[
                (selected["slice_type"] == slice_type) & (selected["model"] == model)
            ]["macro_f1"].astype(float)
            if values.empty:
                continue
            minimum, maximum = float(values.min()), float(values.max())
            y = centre_y + offset
            x1, x2 = left + minimum * chart_width, left + maximum * chart_width
            body.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{MODEL_COLOURS[model]}" stroke-width="7" stroke-linecap="round"/>')
            body.append(_text(x2 + 8, y + 4, f"{minimum:.3f}-{maximum:.3f}", size=11))
    baseline_y = top + len(slice_types) * row_height + 15
    body.append(f'<line x1="{left}" y1="{baseline_y}" x2="{left + chart_width}" y2="{baseline_y}" stroke="#222"/>')
    for value in np.linspace(0, 1, 6):
        x = left + value * chart_width
        body.append(f'<line x1="{x}" y1="{baseline_y}" x2="{x}" y2="{baseline_y + 7}" stroke="#222"/>')
        body.append(_text(x, baseline_y + 27, f"{value:.1f}", anchor="middle", size=12))
    body.append(_text(left, height - 24, "TF-IDF", size=13, weight="700", fill=MODEL_COLOURS["tfidf_logistic_regression"]))
    body.append(_text(left + 100, height - 24, "DistilBERT", size=13, weight="700", fill=MODEL_COLOURS["distilbert"]))
    body.append(_text(left + chart_width / 2, height - 24, "Macro-F1", anchor="middle", size=14))
    _write_svg(path, width, height, body)


def _deployment_recommendation(
    metrics: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> dict[str, Any]:
    distilbert = metrics["models"]["distilbert"]
    tfidf = metrics["models"]["tfidf_logistic_regression"]
    difference = metrics["bootstrap"]["tfidf_minus_distilbert_macro_f1"]
    advantage_interval = {
        "lower": -float(difference["upper"]),
        "point_estimate": -float(difference["point_estimate"]),
        "upper": -float(difference["lower"]),
    }
    latency = benchmark["models"]
    size_ratio = (
        float(latency["distilbert"]["serving_artifact_bytes"])
        / float(latency["tfidf_logistic_regression"]["serving_artifact_bytes"])
    )
    latency_ratio = (
        float(latency["distilbert"]["latency_ms"]["median"])
        / float(latency["tfidf_logistic_regression"]["latency_ms"]["median"])
    )
    neutral_distilbert = next(
        row for row in distilbert["classification"]["per_class"] if row["label"] == "neutral"
    )
    neutral_tfidf = next(
        row for row in tfidf["classification"]["per_class"] if row["label"] == "neutral"
    )
    return {
        "schema_version": 1,
        "stage": "post_test_deployment_recommendation",
        "recommended_default_model": "distilbert",
        "fallback_model": "tfidf_logistic_regression",
        "scope": "local academic demonstration and analyst decision support",
        "decision": (
            "Use DistilBERT by default because its predeclared primary metric and "
            "neutral-class F1 are materially higher. Accept the measured latency, "
            "size and maintenance costs for the local demonstration; retain TF-IDF "
            "as a lightweight fallback."
        ),
        "predictive_evidence": {
            "distilbert_macro_f1": distilbert["classification"]["macro_f1"],
            "tfidf_macro_f1": tfidf["classification"]["macro_f1"],
            "distilbert_advantage_95pct_interval": advantage_interval,
            "distilbert_neutral_f1": neutral_distilbert["f1"],
            "tfidf_neutral_f1": neutral_tfidf["f1"],
        },
        "efficiency_tradeoff": {
            "distilbert_to_tfidf_median_latency_ratio": latency_ratio,
            "distilbert_to_tfidf_serving_size_ratio": size_ratio,
            "no_stakeholder_latency_sla_available": True,
        },
        "proposed_ui_policy": {
            "status": "post-test mitigation, not independently validated",
            "mandatory_human_review_for_predicted_labels": ["neutral"],
            "negative_positive_rule": "apply the frozen DistilBERT confidence threshold",
            "warning": (
                "The protected test diagnosed poor neutral selective accuracy. A future "
                "holdout or prospective evaluation is required before claiming that the "
                "revised policy meets a production accuracy target."
            ),
        },
        "limitations": [
            "Amazon star ratings are distant labels rather than verified sentiment annotations.",
            "The latency result describes one recorded local CPU and batch-one protocol.",
            "The 90% selective-accuracy requirement was illustrative, not stakeholder validated.",
            "Manual qualitative error coding was excluded from the final project scope; "
            "limitations are based on aggregate class, calibration, routing and slice evidence.",
        ],
    }


def _axes(
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    y_max: float,
    y_label: str,
) -> list[str]:
    body = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222"/>',
    ]
    for value in np.linspace(0, y_max, 5):
        y = top + height * (1 - value / y_max)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(_text(left - 12, y + 5, f"{value:.1f}", anchor="end", size=13))
    body.append(_text(25, top + height / 2, y_label, anchor="middle", size=15, weight="700", rotate=-90))
    return body


def _small_axes(left: float, top: float, width: float, height: float, maximum: float) -> list[str]:
    return [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222"/>',
        _text(left - 8, top + 5, f"{maximum:.2f}", anchor="end", size=11),
        _text(left - 8, top + height + 5, "0", anchor="end", size=11),
    ]


def _xy_axes(
    left: float, top: float, width: float, height: float, x_label: str, y_label: str
) -> list[str]:
    body = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#fff" stroke="#222"/>',
    ]
    for value in np.linspace(0, 1, 6):
        x = left + value * width
        body.append(_text(x, top + height + 24, f"{value:.1f}", anchor="middle", size=12))
    for value in np.linspace(0.5, 1.0, 6):
        y = top + height * (1 - (value - 0.5) / 0.5)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#eee"/>')
        body.append(_text(left - 12, y + 4, f"{value:.1f}", anchor="end", size=12))
    body.append(_text(left + width / 2, top + height + 50, x_label, anchor="middle", size=14, weight="700"))
    body.append(_text(28, top + height / 2, y_label, anchor="middle", size=14, weight="700", rotate=-90))
    return body


def _svg_title(value: str, width: int) -> str:
    return _text(width / 2, 42, value, anchor="middle", size=23, weight="700")


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    anchor: str = "start",
    size: int = 15,
    weight: str = "400",
    rotate: int | None = None,
    fill: str = "#1a202c",
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    escaped = html.escape(str(value))
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>{escaped}</text>'
    )


def _write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<g font-family="Arial, Helvetica, sans-serif">\n'
        + "\n".join(body)
        + "\n</g>\n</svg>\n"
    )
    path.write_text(content, encoding="utf-8")


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
