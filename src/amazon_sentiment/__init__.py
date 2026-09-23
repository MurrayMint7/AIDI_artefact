"""Amazon review sentiment artefact."""

from .acquisition import AcquisitionBundle, AcquisitionConfig, acquire
from .baselines import BaselineBundle, BaselineConfig, train_baselines
from .benchmark import BenchmarkBundle, BenchmarkConfig, benchmark
from .data_pipeline import DatasetBundle, PipelineConfig, SchemaError, prepare
from .inference import InferenceConfig, Prediction, SentimentPredictor

__all__ = [
    "AcquisitionBundle",
    "AcquisitionConfig",
    "BaselineBundle",
    "BaselineConfig",
    "BenchmarkBundle",
    "BenchmarkConfig",
    "DatasetBundle",
    "InferenceConfig",
    "PipelineConfig",
    "Prediction",
    "SchemaError",
    "SentimentPredictor",
    "acquire",
    "benchmark",
    "prepare",
    "train_baselines",
]
