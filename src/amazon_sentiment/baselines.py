"""Train local majority and TF-IDF baselines through one testable interface."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from .data_pipeline import SchemaError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineConfig:
    """Resolved settings and filesystem inputs for local baseline training."""

    config_path: Path
    dataset_path: Path
    split_manifest_path: Path
    model_dir: Path
    metrics_dir: Path
    settings: Mapping[str, Any]

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        dataset_path: str | Path,
        split_manifest_path: str | Path,
        model_dir: str | Path,
        metrics_dir: str | Path,
    ) -> "BaselineConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("Experiment configuration must be a YAML mapping")
        return cls(
            config_path=resolved_config,
            dataset_path=Path(dataset_path),
            split_manifest_path=Path(split_manifest_path),
            model_dir=Path(model_dir),
            metrics_dir=Path(metrics_dir),
            settings=settings,
        )


@dataclass(frozen=True)
class BaselineBundle:
    """Saved models and aggregate model-selection evidence."""

    majority_model_path: Path
    tfidf_model_path: Path
    metrics_path: Path


def train_baselines(config: BaselineConfig) -> BaselineBundle:
    """Fit both baselines without reading policy-calibration or test examples."""

    dataset = _read_dataset(config.dataset_path)
    training = dataset.loc[dataset["split"].eq("train")].copy()
    validation = dataset.loc[
        dataset["split"].eq("validation_model_selection")
    ].copy()
    if training.empty:
        raise SchemaError("Baseline dataset has no training rows")
    if validation.empty:
        raise SchemaError("Baseline dataset has no model-selection validation rows")

    labels = _ordered_labels(config.settings)
    label_ids = [label_id for label_id, _ in labels]
    label_names = [label_name for _, label_name in labels]
    _validate_observed_labels(training, validation, set(label_ids))

    split_manifest = _read_json(config.split_manifest_path)
    majority_counts = _pre_balance_class_counts(split_manifest, label_names)
    majority_label = max(
        label_names,
        key=lambda name: (majority_counts[name], -label_names.index(name)),
    )
    label_name_to_id = {name: label_id for label_id, name in labels}
    majority_id = label_name_to_id[majority_label]

    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    majority_model_path = config.model_dir / "majority_model.json"
    tfidf_model_path = config.model_dir / "tfidf_logistic_regression.joblib"
    metrics_path = config.metrics_dir / "baseline_model_selection_metrics.json"

    majority_model = {
        "model_type": "majority_class",
        "predicted_id": majority_id,
        "predicted_label": majority_label,
        "training_class_counts": majority_counts,
        "prevalence_source": "full_pre_balance_training_population",
        "label_order": [
            {"id": label_id, "label": label_name}
            for label_id, label_name in labels
        ],
    }
    _write_json(majority_model_path, majority_model)
    majority_predictions = np.full(len(validation), majority_id, dtype=np.int8)
    majority_metrics = _classification_metrics(
        validation["sentiment_id"].to_numpy(),
        majority_predictions,
        labels,
    )

    selected_model, selected_parameters, candidates, training_seconds = _fit_tfidf_grid(
        training,
        validation,
        config.settings,
        labels,
    )
    joblib.dump(selected_model, tfidf_model_path)
    tfidf_predictions = selected_model.predict(validation["text"].astype(str))
    tfidf_metrics = _classification_metrics(
        validation["sentiment_id"].to_numpy(),
        tfidf_predictions,
        labels,
    )
    probabilities = selected_model.predict_proba(validation["text"].astype(str))
    probability_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))

    _write_json(
        metrics_path,
        {
            "schema_version": 1,
            "config_sha256": _sha256(config.config_path),
            "dataset_sha256": _sha256(config.dataset_path),
            "evaluation_split": "validation_model_selection",
            "test_evaluated": False,
            "selection_metric": "macro_f1",
            "label_order": [
                {"id": label_id, "label": label_name}
                for label_id, label_name in labels
            ],
            "software": {"scikit_learn": sklearn.__version__},
            "runs": {
                "majority": {
                    "model_type": "majority_class",
                    "selected_label": majority_label,
                    "training_population_rows": int(sum(majority_counts.values())),
                    "training_class_counts": majority_counts,
                    "model_bytes": majority_model_path.stat().st_size,
                    "metrics": majority_metrics,
                },
                "tfidf_logistic_regression": {
                    "model_type": "tfidf_logistic_regression",
                    "selected_parameters": selected_parameters,
                    "candidates": candidates,
                    "training_rows": len(training),
                    "model_selection_rows": len(validation),
                    "training_seconds": training_seconds,
                    "model_bytes": tfidf_model_path.stat().st_size,
                    "probability_row_sum_max_error": probability_error,
                    "metrics": tfidf_metrics,
                },
            },
        },
    )
    return BaselineBundle(
        majority_model_path=majority_model_path,
        tfidf_model_path=tfidf_model_path,
        metrics_path=metrics_path,
    )


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SchemaError(f"Prepared dataset does not exist: {path}")
    required = {"text", "sentiment_id", "sentiment_label", "split"}
    dataset = pd.read_parquet(
        path,
        columns=sorted(required),
        filters=[
            (
                "split",
                "in",
                ["train", "validation_model_selection"],
            )
        ],
    )
    missing = required.difference(dataset.columns)
    if missing:
        raise SchemaError(f"Prepared dataset is missing fields: {sorted(missing)}")
    non_empty = dataset["text"].notna() & dataset["text"].astype("string").str.strip().ne("")
    if not non_empty.all():
        raise SchemaError(f"Prepared dataset contains {int((~non_empty).sum())} empty texts")
    return dataset


def _ordered_labels(settings: Mapping[str, Any]) -> list[tuple[int, str]]:
    configured = settings.get("data", {}).get("labels", {})
    labels = sorted(
        (int(label_config["id"]), str(label_name))
        for label_name, label_config in configured.items()
    )
    if not labels or len({label_id for label_id, _ in labels}) != len(labels):
        raise SchemaError("Baseline configuration needs labels with unique integer IDs")
    return labels


def _validate_observed_labels(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    configured_ids: set[int],
) -> None:
    observed = set(
        pd.concat([training["sentiment_id"], validation["sentiment_id"]])
        .astype(int)
        .unique()
    )
    unexpected = observed.difference(configured_ids)
    if unexpected:
        raise SchemaError(f"Dataset contains undeclared sentiment IDs: {sorted(unexpected)}")
    missing_training = configured_ids.difference(set(training["sentiment_id"].astype(int)))
    if missing_training:
        raise SchemaError(
            "TF-IDF training requires every configured class; missing IDs: "
            f"{sorted(missing_training)}"
        )


def _pre_balance_class_counts(
    split_manifest: Mapping[str, Any],
    label_names: list[str],
) -> dict[str, int]:
    records = split_manifest.get("pre_sampling_training_class_counts")
    if not isinstance(records, list):
        raise SchemaError(
            "Split manifest lacks pre_sampling_training_class_counts required by "
            "the majority baseline"
        )
    counts = {label: 0 for label in label_names}
    for record in records:
        label = str(record.get("sentiment_label"))
        if label not in counts:
            raise SchemaError(f"Split manifest contains undeclared sentiment label: {label}")
        counts[label] += int(record["rows"])
    if sum(counts.values()) == 0:
        raise SchemaError("Pre-balance training population is empty")
    return counts


def _fit_tfidf_grid(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    settings: Mapping[str, Any],
    labels: list[tuple[int, str]],
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]], float]:
    model_settings = settings.get("models", {}).get("tfidf_logistic_regression", {})
    ngram_range = tuple(int(value) for value in model_settings.get("word_ngram_range", [1, 2]))
    if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise SchemaError("TF-IDF word_ngram_range must contain two ordered positive integers")
    min_df_candidates = [int(value) for value in model_settings.get("min_df_candidates", [])]
    max_feature_candidates = [
        int(value) for value in model_settings.get("max_features_candidates", [])
    ]
    c_candidates = [float(value) for value in model_settings.get("c_candidates", [])]
    if not min_df_candidates or not max_feature_candidates or not c_candidates:
        raise SchemaError("TF-IDF tuning candidates must be explicitly configured")
    if min(min_df_candidates + max_feature_candidates) < 1 or min(c_candidates) <= 0:
        raise SchemaError("TF-IDF tuning candidates must be positive")

    seed = int(settings.get("project", {}).get("seed", 42))
    max_iter = int(model_settings.get("max_iter", 1000))
    label_ids = [label_id for label_id, _ in labels]
    training_text = training["text"].astype(str)
    training_labels = training["sentiment_id"].astype(int)
    validation_text = validation["text"].astype(str)
    validation_labels = validation["sentiment_id"].astype(int)

    selected_model: Pipeline | None = None
    selected_parameters: dict[str, Any] | None = None
    selected_score = float("-inf")
    candidate_records: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for min_df, max_features, c_value in product(
        min_df_candidates,
        max_feature_candidates,
        c_candidates,
    ):
        parameters = {
            "C": c_value,
            "max_features": max_features,
            "min_df": min_df,
        }
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=max_iter,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        candidate_start = time.perf_counter()
        model.fit(training_text, training_labels)
        predictions = model.predict(validation_text)
        score = float(
            f1_score(
                validation_labels,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        )
        candidate_records.append(
            {
                "parameters": parameters,
                "macro_f1": score,
                "training_seconds": time.perf_counter() - candidate_start,
                "vocabulary_features": len(model.named_steps["tfidf"].vocabulary_),
            }
        )
        LOGGER.info(
            "TF-IDF candidate min_df=%s max_features=%s C=%s macro_f1=%.4f",
            min_df,
            max_features,
            c_value,
            score,
        )
        if score > selected_score:
            selected_score = score
            selected_model = model
            selected_parameters = parameters

    if selected_model is None or selected_parameters is None:  # defensive invariant
        raise RuntimeError("TF-IDF grid produced no model")
    LOGGER.info(
        "Selected TF-IDF parameters %s with validation macro_f1=%.4f",
        selected_parameters,
        selected_score,
    )
    return (
        selected_model,
        selected_parameters,
        candidate_records,
        time.perf_counter() - total_start,
    )


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
        "rows": len(expected),
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
            "values": confusion_matrix(expected, predicted, labels=label_ids).tolist(),
        },
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SchemaError(f"Required manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"Manifest must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
