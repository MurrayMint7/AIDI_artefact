"""Command-line adapters for the artefact modules."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .acquisition import AcquisitionConfig, acquire
from .baselines import BaselineConfig, train_baselines
from .benchmark import BenchmarkConfig, benchmark
from .data_pipeline import PipelineConfig, prepare
from .transformer import (
    ThroughputConfig,
    TokenLengthConfig,
    TransformerTrainingConfig,
    analyse_token_lengths,
    benchmark_training_throughput,
    train_transformer,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amazon_sentiment")
    commands = parser.add_subparsers(dest="command", required=True)

    download_parser = commands.add_parser("download", help="Acquire declared raw files")
    _add_common_config(download_parser)
    download_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    prepare_parser = commands.add_parser("prepare", help="Build the curated dataset")
    _add_common_config(prepare_parser)
    prepare_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
    )

    baseline_parser = commands.add_parser(
        "baselines", help="Train majority and TF-IDF baselines"
    )
    _add_common_config(baseline_parser)
    baseline_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/generated/processed/training_dataset.parquet"),
    )
    baseline_parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/generated/manifests/split_manifest.json"),
    )
    baseline_parser.add_argument(
        "--model-dir", type=Path, default=Path("models/baselines")
    )
    baseline_parser.add_argument(
        "--metrics-dir", type=Path, default=Path("artifacts/metrics")
    )

    benchmark_parser = commands.add_parser(
        "benchmark", help="Compare equivalent JSONL.gz and Parquet reads"
    )
    _add_common_config(
        benchmark_parser,
        default=Path("config/storage_benchmark.yaml"),
    )
    benchmark_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    benchmark_parser.add_argument(
        "--work-dir", type=Path, default=Path("data/benchmark")
    )
    benchmark_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/metrics")
    )

    token_parser = commands.add_parser(
        "token-length", help="Measure DistilBERT token coverage without test access"
    )
    _add_common_config(token_parser, default=Path("config/distilbert.yaml"))
    token_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/generated/processed/training_dataset.parquet"),
    )
    token_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/metrics")
    )
    token_parser.add_argument(
        "--cache-dir", type=Path, default=Path("models/huggingface")
    )

    throughput_parser = commands.add_parser(
        "throughput", help="Benchmark DistilBERT training configurations on CUDA"
    )
    _add_common_config(throughput_parser, default=Path("config/distilbert.yaml"))
    throughput_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/generated/processed/training_dataset.parquet"),
    )
    throughput_parser.add_argument(
        "--token-summary",
        type=Path,
        default=Path("artifacts/metrics/distilbert_token_length_summary.json"),
    )
    throughput_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/metrics")
    )
    throughput_parser.add_argument(
        "--cache-dir", type=Path, default=Path("models/huggingface")
    )

    train_parser = commands.add_parser(
        "train-transformer", help="Fine-tune and checkpoint DistilBERT on CUDA"
    )
    _add_common_config(train_parser, default=Path("config/distilbert.yaml"))
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/generated/processed/training_dataset.parquet"),
    )
    train_parser.add_argument(
        "--decision",
        type=Path,
        default=Path("artifacts/metrics/distilbert_training_decision.json"),
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path("models/distilbert-run")
    )
    train_parser.add_argument(
        "--cache-dir", type=Path, default=Path("models/huggingface")
    )

    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if arguments.command == "download":
        bundle = acquire(
            AcquisitionConfig.from_yaml(arguments.config, raw_dir=arguments.raw_dir)
        )
        print(
            json.dumps(
                {
                    "files": [str(path) for path in bundle.files],
                    "manifest": str(bundle.manifest_path),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "prepare":
        bundle = prepare(
            PipelineConfig.from_yaml(
                arguments.config,
                raw_dir=arguments.raw_dir,
                output_dir=arguments.output_dir,
            )
        )
        print(
            json.dumps(
                {
                    "dataset": str(bundle.dataset_path),
                    "data_manifest": str(bundle.data_manifest_path),
                    "data_quality": str(bundle.data_quality_path),
                    "join_report": str(bundle.join_report_path),
                    "split_manifest": str(bundle.split_manifest_path),
                    "row_counts": bundle.row_counts,
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "benchmark":
        benchmark_bundle = benchmark(
            BenchmarkConfig.from_yaml(
                arguments.config,
                raw_dir=arguments.raw_dir,
                work_dir=arguments.work_dir,
                output_dir=arguments.output_dir,
            )
        )
        print(
            json.dumps(
                {
                    "results": str(benchmark_bundle.results_path),
                    "environment": str(benchmark_bundle.environment_path),
                    "decision": str(benchmark_bundle.decision_path),
                    "corpus_manifest": str(benchmark_bundle.corpus_manifest_path),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "token-length":
        token_bundle = analyse_token_lengths(
            TokenLengthConfig.from_yaml(
                arguments.config,
                dataset_path=arguments.dataset,
                output_dir=arguments.output_dir,
                cache_dir=arguments.cache_dir,
            )
        )
        print(
            json.dumps(
                {
                    "summary": str(token_bundle.summary_path),
                    "slices": str(token_bundle.slices_path),
                    "resolved_config": str(token_bundle.resolved_config_path),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "throughput":
        throughput_bundle = benchmark_training_throughput(
            ThroughputConfig.from_yaml(
                arguments.config,
                dataset_path=arguments.dataset,
                token_summary_path=arguments.token_summary,
                output_dir=arguments.output_dir,
                cache_dir=arguments.cache_dir,
            )
        )
        print(
            json.dumps(
                {
                    "results": str(throughput_bundle.results_path),
                    "decision": str(throughput_bundle.decision_path),
                    "environment": str(throughput_bundle.environment_path),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "train-transformer":
        training_bundle = train_transformer(
            TransformerTrainingConfig.from_yaml(
                arguments.config,
                dataset_path=arguments.dataset,
                decision_path=arguments.decision,
                output_dir=arguments.output_dir,
                cache_dir=arguments.cache_dir,
            )
        )
        print(
            json.dumps(
                {
                    "model": str(training_bundle.model_dir),
                    "validation_outputs": str(training_bundle.validation_outputs_path),
                    "summary": str(training_bundle.training_summary_path),
                    "environment": str(training_bundle.environment_path),
                },
                indent=2,
            )
        )
        return 0

    baseline_bundle = train_baselines(
        BaselineConfig.from_yaml(
            arguments.config,
            dataset_path=arguments.dataset,
            split_manifest_path=arguments.split_manifest,
            model_dir=arguments.model_dir,
            metrics_dir=arguments.metrics_dir,
        )
    )
    print(
        json.dumps(
            {
                "majority_model": str(baseline_bundle.majority_model_path),
                "tfidf_model": str(baseline_bundle.tfidf_model_path),
                "metrics": str(baseline_bundle.metrics_path),
            },
            indent=2,
        )
    )
    return 0


def _add_common_config(
    parser: argparse.ArgumentParser,
    *,
    default: Path = Path("config/experiment.yaml"),
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
    )


if __name__ == "__main__":
    raise SystemExit(main())
