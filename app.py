"""Launch the local-only Gradio sentiment interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from amazon_sentiment.inference import InferenceConfig, SentimentPredictor
from amazon_sentiment.interface import build_app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SENTIMENT_CONFIG", "config/experiment.yaml")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get("SENTIMENT_MODEL_DIR", "models/distilbert-run")
        ),
        help="Local DistilBERT run bundle containing model/ and calibration.json",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(
            os.environ.get(
                "SENTIMENT_THRESHOLDS", "artifacts/metrics/review_thresholds.json"
            )
        ),
    )
    parser.add_argument(
        "--deployment-policy",
        type=Path,
        default=Path(
            os.environ.get(
                "SENTIMENT_DEPLOYMENT_POLICY",
                "artifacts/metrics/deployment_recommendation.json",
            )
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(
            os.environ.get("SENTIMENT_LOG_PATH", "artifacts/logs/inference.jsonl")
        ),
        help="Ignored privacy-safe JSONL operational log",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SENTIMENT_PORT", "7860")),
    )
    arguments = parser.parse_args(argv)

    config = InferenceConfig.from_paths(
        arguments.config,
        distilbert_run_dir=arguments.model_dir,
        thresholds_path=arguments.thresholds,
        deployment_policy_path=arguments.deployment_policy,
        log_path=arguments.log_path,
    )
    app = build_app(SentimentPredictor(config))
    app.launch(
        server_name="127.0.0.1",
        server_port=arguments.port,
        share=False,
        inbrowser=False,
        show_error=False,
        css=app.stage5_css,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
