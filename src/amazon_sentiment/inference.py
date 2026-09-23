"""Serve the frozen DistilBERT model through one governed inference interface."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import yaml

from .data_pipeline import SchemaError


@dataclass(frozen=True)
class InferenceConfig:
    """Resolved paths and policy for local inference."""

    experiment_config_path: Path
    distilbert_run_dir: Path
    thresholds_path: Path
    deployment_policy_path: Path
    log_path: Path | None
    labels: tuple[str, ...]
    max_length: int
    threshold: float
    mandatory_review_labels: frozenset[str]
    temperature: float
    model_version: str
    config_version: str
    policy_version: str

    @classmethod
    def from_paths(
        cls,
        experiment_config_path: str | Path,
        *,
        distilbert_run_dir: str | Path,
        thresholds_path: str | Path,
        deployment_policy_path: str | Path,
        log_path: str | Path | None = "artifacts/logs/inference.jsonl",
    ) -> "InferenceConfig":
        experiment_path = Path(experiment_config_path)
        run_dir = Path(distilbert_run_dir)
        resolved_thresholds = Path(thresholds_path)
        resolved_policy = Path(deployment_policy_path)

        settings = _read_yaml(experiment_path)
        labels = _ordered_labels(settings)
        max_length = int(
            settings.get("models", {}).get("distilbert", {}).get("max_length", 0)
        )
        if max_length < 1:
            raise SchemaError("DistilBERT max_length must be positive")

        thresholds = _read_json(resolved_thresholds)
        if thresholds.get("stage") != "policy_calibration":
            raise SchemaError("Threshold bundle must come from policy_calibration")
        if thresholds.get("evaluation_split") != "validation_policy_calibration":
            raise SchemaError(
                "Threshold bundle must be fitted on validation_policy_calibration"
            )
        if thresholds.get("test_evaluated") is not False:
            raise SchemaError("Threshold bundle must record test_evaluated as false")
        threshold_record = thresholds.get("models", {}).get("distilbert")
        if not isinstance(threshold_record, dict):
            raise SchemaError("Threshold bundle is missing the DistilBERT policy")
        threshold = float(threshold_record.get("threshold", math.nan))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise SchemaError("Frozen DistilBERT threshold must be in [0, 1]")
        if not bool(threshold_record.get("requirement_met", False)):
            raise SchemaError("Frozen DistilBERT routing requirement was not met")

        calibration = _read_json(run_dir / "calibration.json")
        if calibration.get("method") != "temperature_scaling":
            raise SchemaError("DistilBERT calibration must use temperature_scaling")
        if calibration.get("fit_split") != "validation_policy_calibration":
            raise SchemaError(
                "DistilBERT calibration must be fitted on validation_policy_calibration"
            )
        if calibration.get("test_evaluated") is not False:
            raise SchemaError("Calibration must record test_evaluated as false")
        temperature = float(calibration.get("temperature", math.nan))
        if not math.isfinite(temperature) or temperature <= 0:
            raise SchemaError("Frozen DistilBERT temperature must be positive")

        model_config = _read_json(run_dir / "model" / "config.json")
        model_labels = tuple(
            str(model_config.get("id2label", {}).get(str(index), ""))
            for index in range(len(labels))
        )
        if model_labels != labels:
            raise SchemaError(
                f"Model labels {model_labels!r} do not match configured labels {labels!r}"
            )

        weights_path = run_dir / "model" / "model.safetensors"
        expected_model_hash = str(calibration.get("model_sha256", ""))
        if not expected_model_hash:
            raise SchemaError("Calibration metadata is missing model_sha256")
        if _sha256(weights_path) != expected_model_hash:
            raise SchemaError("Model weights do not match the calibrated model hash")

        deployment = _read_json(resolved_policy)
        if deployment.get("recommended_default_model") != "distilbert":
            raise SchemaError("Deployment policy does not select DistilBERT by default")
        proposed_policy = deployment.get("proposed_ui_policy", {})
        mandatory = proposed_policy.get(
            "mandatory_human_review_for_predicted_labels", []
        )
        if not isinstance(mandatory, list) or not mandatory:
            raise SchemaError("Deployment policy must declare mandatory review labels")
        mandatory_labels = frozenset(str(label) for label in mandatory)
        unknown_labels = mandatory_labels.difference(labels)
        if unknown_labels:
            raise SchemaError(
                f"Deployment policy has unknown review labels: {sorted(unknown_labels)}"
            )

        model_version = f"distilbert:{expected_model_hash[:12]}"
        policy_hash = hashlib.sha256(
            resolved_thresholds.read_bytes() + resolved_policy.read_bytes()
        ).hexdigest()
        return cls(
            experiment_config_path=experiment_path,
            distilbert_run_dir=run_dir,
            thresholds_path=resolved_thresholds,
            deployment_policy_path=resolved_policy,
            log_path=Path(log_path) if log_path is not None else None,
            labels=labels,
            max_length=max_length,
            threshold=threshold,
            mandatory_review_labels=mandatory_labels,
            temperature=temperature,
            model_version=model_version,
            config_version=_sha256(experiment_path)[:12],
            policy_version=policy_hash[:12],
        )


@dataclass(frozen=True)
class ModelOutput:
    """Raw output returned by an inference backend."""

    logits: np.ndarray
    token_length: int


class InferenceBackend(Protocol):
    """Internal seam for the local model runtime and deterministic test adapter."""

    def predict(self, text: str, *, max_length: int) -> ModelOutput: ...


@dataclass(frozen=True)
class Prediction:
    """One calibrated sentiment and routing decision."""

    request_id: str
    timestamp_utc: str
    label: str
    probabilities: Mapping[str, float]
    confidence: float
    review_required: bool
    review_reason: str
    model_version: str
    config_version: str
    policy_version: str
    threshold: float
    input_character_length: int
    input_token_length: int
    latency_ms: float

    @property
    def automatic_route(self) -> bool:
        return not self.review_required

    def log_record(self) -> dict[str, Any]:
        """Return the privacy-safe operational record; review text is never present."""

        record = asdict(self)
        record["automatic_route"] = self.automatic_route
        return record


class InferenceInputError(ValueError):
    """Raised when review text cannot be classified."""


class JsonlPredictionLogger:
    """Append prediction metadata without retaining submitted review text."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, prediction: Prediction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            prediction.log_record(), sort_keys=True, ensure_ascii=True
        ) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class SentimentPredictor:
    """Load once, then expose the complete serving behavior as ``predict(text)``."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        backend: InferenceBackend | None = None,
        clock: Callable[[], float] = time.perf_counter,
        request_id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.config = config
        self._backend = backend or LocalDistilBertBackend(
            config.distilbert_run_dir / "model", device="cpu"
        )
        self._clock = clock
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))
        self._now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self._logger = (
            JsonlPredictionLogger(config.log_path)
            if config.log_path is not None
            else None
        )

    def predict(self, text: str) -> Prediction:
        """Classify one non-empty review and apply the frozen review policy."""

        if not isinstance(text, str) or not text.strip():
            raise InferenceInputError("Enter review text before submitting.")

        started = self._clock()
        output = self._backend.predict(text, max_length=self.config.max_length)
        logits = np.asarray(output.logits, dtype=np.float64)
        if logits.shape == (1, len(self.config.labels)):
            logits = logits[0]
        if logits.shape != (len(self.config.labels),) or not np.isfinite(logits).all():
            raise RuntimeError("Model returned invalid sentiment logits")
        probabilities_array = _softmax(logits / self.config.temperature)
        predicted_id = int(probabilities_array.argmax())
        label = self.config.labels[predicted_id]
        confidence = float(probabilities_array[predicted_id])
        if label in self.config.mandatory_review_labels:
            review_required = True
            review_reason = "mandatory_review_for_predicted_label"
        elif confidence < self.config.threshold:
            review_required = True
            review_reason = "below_frozen_confidence_threshold"
        else:
            review_required = False
            review_reason = "frozen_confidence_threshold_met"

        prediction = Prediction(
            request_id=self._request_id_factory(),
            timestamp_utc=self._now_factory().astimezone(timezone.utc).isoformat(),
            label=label,
            probabilities={
                name: float(probabilities_array[index])
                for index, name in enumerate(self.config.labels)
            },
            confidence=confidence,
            review_required=review_required,
            review_reason=review_reason,
            model_version=self.config.model_version,
            config_version=self.config.config_version,
            policy_version=self.config.policy_version,
            threshold=self.config.threshold,
            input_character_length=len(text),
            input_token_length=int(output.token_length),
            latency_ms=(self._clock() - started) * 1000,
        )
        if self._logger is not None:
            self._logger.write(prediction)
        return prediction


class LocalDistilBertBackend:
    """Local-only Hugging Face adapter used by serving."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu"):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Local DistilBERT inference requires requirements-evaluation.txt"
            ) from error

        self._torch = torch
        self._device = torch.device(device)
        resolved_model_dir = str(Path(model_dir))
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_dir, use_fast=True, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_dir, local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()

    def predict(self, text: str, *, max_length: int) -> ModelOutput:
        logits, token_lengths = self.predict_many(
            [text],
            max_length=max_length,
            batch_size=1,
        )
        return ModelOutput(logits=logits[0], token_length=int(token_lengths[0]))

    def predict_many(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned raw logits and processed token lengths."""

        if not texts:
            raise ValueError("DistilBERT inference needs at least one text")
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        logit_batches: list[np.ndarray] = []
        token_length_batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                encoded = self._tokenizer(
                    list(texts[offset : offset + batch_size]),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                token_length_batches.append(
                    encoded["attention_mask"].sum(dim=1).cpu().numpy()
                )
                encoded = {
                    key: value.to(self._device) for key, value in encoded.items()
                }
                logit_batches.append(
                    self._model(**encoded).logits.detach().cpu().numpy()
                )
        return (
            np.concatenate(logit_batches, axis=0).astype(np.float64),
            np.concatenate(token_length_batches).astype(np.int64),
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _ordered_labels(settings: Mapping[str, Any]) -> tuple[str, ...]:
    label_settings = settings.get("data", {}).get("labels", {})
    if not isinstance(label_settings, dict) or not label_settings:
        raise SchemaError("Experiment configuration is missing sentiment labels")
    ordered: list[tuple[int, str]] = []
    for label, values in label_settings.items():
        if not isinstance(values, dict) or "id" not in values:
            raise SchemaError(f"Label {label!r} is missing an integer id")
        ordered.append((int(values["id"]), str(label)))
    ordered.sort()
    if [label_id for label_id, _ in ordered] != list(range(len(ordered))):
        raise SchemaError("Sentiment label IDs must be contiguous and start at zero")
    return tuple(label for _, label in ordered)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SchemaError(f"Required inference file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"Expected a YAML mapping in {path}")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SchemaError(f"Required inference file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise SchemaError(f"Required inference file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
