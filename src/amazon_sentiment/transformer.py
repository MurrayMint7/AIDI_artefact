"""Governed pilot utilities for the DistilBERT training stage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .data_pipeline import SchemaError


REQUIRED_COLUMNS = ("text", "sentiment_label", "source_category", "split")


@dataclass(frozen=True)
class TokenLengthConfig:
    """Resolved inputs for an aggregate-only tokenizer length pilot."""

    config_path: Path
    dataset_path: Path
    output_dir: Path
    settings: Mapping[str, Any]
    model_id: str
    revision: str
    included_splits: tuple[str, ...]
    max_length_candidates: tuple[int, ...]
    minimum_token_coverage: float
    batch_rows: int
    cache_dir: Path | None = None

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        dataset_path: str | Path,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
    ) -> "TokenLengthConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("DistilBERT configuration must be a YAML mapping")

        model = settings.get("model", {})
        pilot = settings.get("token_length_pilot", {})
        model_id = str(model.get("model_id", "")).strip()
        revision = str(model.get("revision", "")).strip()
        included_splits = tuple(str(value) for value in pilot.get("included_splits", []))
        candidates = tuple(sorted({int(value) for value in pilot.get("max_length_candidates", [])}))
        minimum_coverage = float(pilot.get("minimum_token_coverage", 0.0))
        batch_rows = int(pilot.get("batch_rows", 0))

        if not model_id or not revision:
            raise SchemaError("A model_id and immutable revision are required")
        if not included_splits:
            raise SchemaError("Token-length pilot declares no included splits")
        forbidden = {"test", "validation_policy_calibration"}.intersection(included_splits)
        if forbidden:
            raise SchemaError(
                "Token-length pilot cannot inspect protected splits: "
                + ", ".join(sorted(forbidden))
            )
        if not candidates or candidates[0] <= 0:
            raise SchemaError("max_length_candidates must contain positive integers")
        if not 0.0 < minimum_coverage <= 1.0:
            raise SchemaError("minimum_token_coverage must be in (0, 1]")
        if batch_rows <= 0:
            raise SchemaError("batch_rows must be positive")

        return cls(
            config_path=resolved_config,
            dataset_path=Path(dataset_path),
            output_dir=Path(output_dir),
            settings=settings,
            model_id=model_id,
            revision=revision,
            included_splits=included_splits,
            max_length_candidates=candidates,
            minimum_token_coverage=minimum_coverage,
            batch_rows=batch_rows,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
        )


@dataclass(frozen=True)
class TokenLengthBundle:
    """Publishable aggregate evidence from the tokenizer pilot."""

    summary_path: Path
    slices_path: Path
    resolved_config_path: Path


@dataclass(frozen=True)
class ThroughputConfig:
    """Resolved inputs for a CUDA training-throughput pilot."""

    config_path: Path
    dataset_path: Path
    token_summary_path: Path
    output_dir: Path
    cache_dir: Path | None
    model_id: str
    revision: str
    max_length_candidates: tuple[int, ...]
    sample_rows: int
    per_device_batch_sizes: tuple[int, ...]
    effective_batch_size: int
    warmup_steps: int
    measured_steps: int
    maximum_projected_training_hours: float
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        dataset_path: str | Path,
        token_summary_path: str | Path,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
    ) -> "ThroughputConfig":
        resolved_config = Path(config_path)
        settings = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise SchemaError("DistilBERT configuration must be a YAML mapping")
        model = settings.get("model", {})
        lengths = settings.get("token_length_pilot", {})
        pilot = settings.get("throughput_pilot", {})
        training = settings.get("training", {})
        candidates = tuple(sorted({int(v) for v in lengths.get("max_length_candidates", [])}))
        batch_sizes = tuple(sorted({int(v) for v in pilot.get("per_device_batch_size_candidates", [])}))
        effective_batch_size = int(pilot.get("effective_batch_size", 0))
        if not candidates or not batch_sizes:
            raise SchemaError("Throughput pilot requires length and batch-size candidates")
        if any(value <= 0 for value in (*candidates, *batch_sizes, effective_batch_size)):
            raise SchemaError("Throughput lengths and batch sizes must be positive")
        if any(effective_batch_size % size for size in batch_sizes):
            raise SchemaError("Effective batch size must be divisible by every candidate")
        return cls(
            config_path=resolved_config,
            dataset_path=Path(dataset_path),
            token_summary_path=Path(token_summary_path),
            output_dir=Path(output_dir),
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_length_candidates=candidates,
            sample_rows=int(pilot["sample_rows"]),
            per_device_batch_sizes=batch_sizes,
            effective_batch_size=effective_batch_size,
            warmup_steps=int(pilot["warmup_steps"]),
            measured_steps=int(pilot["measured_steps"]),
            maximum_projected_training_hours=float(
                pilot["maximum_projected_training_hours"]
            ),
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(training["seed"]),
        )


@dataclass(frozen=True)
class ThroughputBundle:
    """Aggregate outputs from a Colab CUDA throughput pilot."""

    results_path: Path
    decision_path: Path
    environment_path: Path


def analyse_token_lengths(
    config: TokenLengthConfig,
    *,
    tokenizer: Any | None = None,
) -> TokenLengthBundle:
    """Measure untruncated token lengths without reading protected split rows."""

    started = time.perf_counter()
    _validate_dataset_schema(config.dataset_path)
    frame = pd.read_parquet(
        config.dataset_path,
        columns=list(REQUIRED_COLUMNS),
        filters=[("split", "in", list(config.included_splits))],
    )
    if frame.empty:
        raise SchemaError("Token-length pilot selected zero rows")
    unexpected = set(frame["split"].astype(str)).difference(config.included_splits)
    if unexpected:
        raise SchemaError(f"Parquet filter returned unexpected splits: {sorted(unexpected)}")
    if frame["text"].isna().any():
        raise SchemaError("Token-length pilot encountered null review text")

    resolved_tokenizer = tokenizer or _load_tokenizer(config)
    lengths: list[int] = []
    texts = frame["text"].astype(str)
    for offset in range(0, len(frame), config.batch_rows):
        encoded = resolved_tokenizer(
            texts.iloc[offset : offset + config.batch_rows].tolist(),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)
    if len(lengths) != len(frame):
        raise SchemaError("Tokenizer returned a different number of lengths than input rows")
    frame = frame.drop(columns=["text"]).assign(token_length=lengths)

    slice_records = _slice_records(frame, config.max_length_candidates)
    overall = next(record for record in slice_records if record["slice_type"] == "overall")
    recommended = _recommend_length(
        overall["coverage"],
        config.max_length_candidates,
        config.minimum_token_coverage,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "distilbert_token_length_summary.json"
    slices_path = config.output_dir / "distilbert_token_length_slices.csv"
    resolved_config_path = config.output_dir / "distilbert_resolved_config.json"

    slice_frame = pd.DataFrame(
        [_flatten_slice_record(record, config.max_length_candidates) for record in slice_records]
    )
    slice_frame.to_csv(slices_path, index=False)
    split_counts = {
        str(key): int(value)
        for key, value in frame["split"].value_counts(sort=False).sort_index().items()
    }
    summary = {
        "schema_version": 1,
        "stage": "token_length_pilot",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "tokenizer_name_or_path": str(
            getattr(resolved_tokenizer, "name_or_path", config.model_id)
        ),
        "tokenizer_class": type(resolved_tokenizer).__name__,
        "tokenizer_model_max_length": _safe_int(
            getattr(resolved_tokenizer, "model_max_length", None)
        ),
        "rows": int(len(frame)),
        "included_splits": list(config.included_splits),
        "rows_by_split": split_counts,
        "test_evaluated": False,
        "policy_calibration_evaluated": False,
        "max_length_candidates": list(config.max_length_candidates),
        "minimum_token_coverage": config.minimum_token_coverage,
        "recommended_max_length": recommended,
        "recommendation_status": "provisional_pending_colab_throughput_pilot",
        "overall": overall,
        "evidence": {
            "config_sha256": _sha256(config.config_path),
            "dataset_sha256": _sha256(config.dataset_path),
            "slices_sha256": _sha256(slices_path),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    _write_json(summary_path, summary)
    _write_json(
        resolved_config_path,
        {
            "schema_version": 1,
            "model": {"model_id": config.model_id, "revision": config.revision},
            "token_length_pilot": {
                "included_splits": list(config.included_splits),
                "max_length_candidates": list(config.max_length_candidates),
                "minimum_token_coverage": config.minimum_token_coverage,
                "batch_rows": config.batch_rows,
                "recommended_max_length": recommended,
                "status": "provisional_pending_colab_throughput_pilot",
            },
            "source_config_sha256": _sha256(config.config_path),
            "dataset_sha256": _sha256(config.dataset_path),
        },
    )
    return TokenLengthBundle(summary_path, slices_path, resolved_config_path)


def benchmark_training_throughput(config: ThroughputConfig) -> ThroughputBundle:
    """Benchmark short CUDA training loops without touching protected rows."""

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.data.data_collator import DataCollatorWithPadding
    except ImportError as error:  # pragma: no cover - exercised in Colab
        raise RuntimeError("The throughput pilot requires PyTorch and Transformers") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run the throughput pilot in a GPU Colab runtime")
    if not config.token_summary_path.is_file():
        raise FileNotFoundError(config.token_summary_path)
    token_summary = json.loads(config.token_summary_path.read_text(encoding="utf-8"))
    if token_summary.get("test_evaluated") is not False:
        raise SchemaError("Token summary does not prove test exclusion")
    if token_summary.get("evidence", {}).get("dataset_sha256") != _sha256(
        config.dataset_path
    ):
        raise SchemaError("Token summary and throughput dataset hashes do not match")
    if token_summary.get("model_revision") != config.revision:
        raise SchemaError("Token summary and throughput model revisions do not match")
    preferred_length = int(token_summary["recommended_max_length"])
    if preferred_length not in config.max_length_candidates:
        raise SchemaError("Token-summary recommendation is not a configured candidate")

    _validate_dataset_schema(config.dataset_path)
    frame = pd.read_parquet(
        config.dataset_path,
        columns=["text", "sentiment_label", "split"],
        filters=[("split", "==", "train")],
    )
    if frame.empty or not frame["split"].eq("train").all():
        raise SchemaError("Throughput pilot must contain training rows only")
    train_rows = int(len(frame))
    sample_size = min(config.sample_rows, train_rows)
    sample = frame.sample(n=sample_size, random_state=config.seed).reset_index(drop=True)
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    labels = sample["sentiment_label"].map(label_to_id)
    if labels.isna().any():
        raise SchemaError("Throughput sample contains an unknown sentiment label")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        use_fast=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    results: list[dict[str, Any]] = []
    for max_length in config.max_length_candidates:
        encoded = tokenizer(
            sample["text"].astype(str).tolist(),
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        features = [
            {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "labels": int(labels.iloc[index]),
            }
            for index in range(sample_size)
        ]
        for per_device_batch_size in config.per_device_batch_sizes:
            results.append(
                _run_throughput_candidate(
                    config,
                    features,
                    max_length,
                    per_device_batch_size,
                    collator,
                    torch,
                    AutoModelForSequenceClassification,
                    train_rows,
                )
            )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "distilbert_throughput_results.json"
    decision_path = config.output_dir / "distilbert_pilot_decision.json"
    environment_path = config.output_dir / "distilbert_colab_environment.json"
    _write_json(
        results_path,
        {
            "schema_version": 1,
            "test_evaluated": False,
            "policy_calibration_evaluated": False,
            "sample_source_split": "train",
            "sample_rows": sample_size,
            "results": results,
        },
    )
    decision = _throughput_decision(config, token_summary, results)
    _write_json(decision_path, decision)
    _write_json(
        environment_path,
        _cuda_environment(config, torch, tokenizer, results_path, decision_path),
    )
    return ThroughputBundle(results_path, decision_path, environment_path)


def _run_throughput_candidate(
    config: ThroughputConfig,
    features: list[dict[str, Any]],
    max_length: int,
    per_device_batch_size: int,
    collator: Any,
    torch: Any,
    model_factory: Any,
    train_rows: int,
) -> dict[str, Any]:
    accumulation_steps = config.effective_batch_size // per_device_batch_size
    candidate = {
        "max_length": max_length,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
    }
    model = None
    try:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        model = model_factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
            num_labels=3,
        ).to("cuda")
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        cursor = 0

        def one_optimizer_step() -> None:
            nonlocal cursor
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation_steps):
                indexes = [
                    (cursor + offset) % len(features)
                    for offset in range(per_device_batch_size)
                ]
                cursor = (cursor + per_device_batch_size) % len(features)
                batch = collator([features[index] for index in indexes])
                batch = {key: value.to("cuda") for key, value in batch.items()}
                loss = model(**batch).loss / accumulation_steps
                loss.backward()
            optimizer.step()

        for _ in range(config.warmup_steps):
            one_optimizer_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_seconds: list[float] = []
        for _ in range(config.measured_steps):
            started = time.perf_counter()
            one_optimizer_step()
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - started)
        median_seconds = statistics.median(step_seconds)
        examples_per_second = config.effective_batch_size / median_seconds
        projected_hours = train_rows * config.epochs / examples_per_second / 3600.0
        candidate.update(
            {
                "status": "success",
                "measured_optimizer_steps": config.measured_steps,
                "median_optimizer_step_seconds": round(median_seconds, 6),
                "min_optimizer_step_seconds": round(min(step_seconds), 6),
                "max_optimizer_step_seconds": round(max(step_seconds), 6),
                "examples_per_second": round(examples_per_second, 4),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "projected_two_epoch_training_hours": round(projected_hours, 4),
                "within_time_budget": projected_hours
                <= config.maximum_projected_training_hours,
            }
        )
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        candidate.update({"status": "cuda_out_of_memory", "within_time_budget": False})
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return candidate


def _throughput_decision(
    config: ThroughputConfig,
    token_summary: Mapping[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    preferred = int(token_summary["recommended_max_length"])
    viable = [
        row
        for row in results
        if row["max_length"] == preferred
        and row["status"] == "success"
        and row["within_time_budget"]
    ]
    if not viable:
        return {
            "status": "blocked",
            "reason": "No successful batch candidate met the time budget at the coverage-selected length",
            "coverage_selected_max_length": preferred,
            "test_evaluated": False,
        }
    winner = max(viable, key=lambda row: row["examples_per_second"])
    return {
        "status": "frozen",
        "selection_rule": (
            "Use the token-coverage-selected length, then choose the fastest successful "
            "per-device batch whose projected two-epoch run is within the declared budget"
        ),
        "max_length": preferred,
        "per_device_batch_size": winner["per_device_batch_size"],
        "gradient_accumulation_steps": winner["gradient_accumulation_steps"],
        "effective_batch_size": winner["effective_batch_size"],
        "token_coverage": token_summary["overall"]["coverage"][str(preferred)],
        "projected_training_hours": winner["projected_two_epoch_training_hours"],
        "maximum_projected_training_hours": config.maximum_projected_training_hours,
        "test_evaluated": False,
    }


def _cuda_environment(
    config: ThroughputConfig,
    torch: Any,
    tokenizer: Any,
    results_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "cuda": str(torch.version.cuda),
        "cudnn": str(cudnn_version) if cudnn_version is not None else "not_available",
        "torch": str(torch.__version__),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "pyarrow": _package_version("pyarrow"),
        "fsspec": _package_version("fsspec"),
        "rich": _package_version("rich"),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "accelerate": _package_version("accelerate"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "typer": _package_version("typer"),
        "tokenizer_class": type(tokenizer).__name__,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "git_commit": _git_commit(),
        "seed": config.seed,
        "config_sha256": _sha256(config.config_path),
        "dataset_sha256": _sha256(config.dataset_path),
        "token_summary_sha256": _sha256(config.token_summary_path),
        "results_sha256": _sha256(results_path),
        "decision_sha256": _sha256(decision_path),
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _load_tokenizer(config: TokenLengthConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Install requirements-transformer.txt before running the tokenizer pilot"
        ) from error
    return AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        use_fast=True,
    )


def _validate_dataset_schema(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(REQUIRED_COLUMNS).difference(names)
    if missing:
        raise SchemaError(f"Dataset is missing required columns: {sorted(missing)}")


def _slice_records(
    frame: pd.DataFrame,
    candidates: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = [_summarise_lengths(frame["token_length"], "overall", "all", candidates)]
    for column, slice_type in (
        ("sentiment_label", "sentiment_label"),
        ("source_category", "source_category"),
    ):
        for value, group in frame.groupby(column, sort=True, observed=True):
            records.append(
                _summarise_lengths(group["token_length"], slice_type, str(value), candidates)
            )
    for (category, label), group in frame.groupby(
        ["source_category", "sentiment_label"], sort=True, observed=True
    ):
        records.append(
            _summarise_lengths(
                group["token_length"],
                "category_and_label",
                f"{category}|{label}",
                candidates,
            )
        )
    return records


def _summarise_lengths(
    lengths: pd.Series,
    slice_type: str,
    slice_value: str,
    candidates: tuple[int, ...],
) -> dict[str, Any]:
    values = lengths.to_numpy(dtype=np.int64)
    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "rows": int(values.size),
        "mean": round(float(np.mean(values)), 4),
        "p50": float(np.quantile(values, 0.50, method="linear")),
        "p90": float(np.quantile(values, 0.90, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "max": int(np.max(values)),
        "coverage": {
            str(candidate): round(float(np.mean(values <= candidate)), 6)
            for candidate in candidates
        },
    }


def _flatten_slice_record(
    record: Mapping[str, Any], candidates: tuple[int, ...]
) -> dict[str, Any]:
    flat = {key: value for key, value in record.items() if key != "coverage"}
    coverage = record["coverage"]
    for candidate in candidates:
        flat[f"coverage_at_{candidate}"] = coverage[str(candidate)]
        flat[f"truncation_rate_at_{candidate}"] = round(
            1.0 - float(coverage[str(candidate)]), 6
        )
    return flat


def _recommend_length(
    coverage: Mapping[str, float],
    candidates: tuple[int, ...],
    threshold: float,
) -> int:
    return next(
        (candidate for candidate in candidates if coverage[str(candidate)] >= threshold),
        candidates[-1],
    )


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed < 10**12 else None


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
