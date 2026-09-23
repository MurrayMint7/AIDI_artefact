# Amazon review sentiment artefact

This repository contains the implementation and public evidence for an academic
artefact that compares majority-class and TF-IDF baselines with a fine-tuned
DistilBERT model for three-class Amazon review sentiment classification.

The public repository intentionally excludes source reviews, processed datasets,
row-level predictions, operational logs, credentials, checkpoints and model
weights. See `.gitignore` before adding generated files.

## Current status

The governed dataset, local baselines, DistilBERT fine-tune, calibration and
one-time protected test evaluation are complete. The shared local inference
module and Gradio interface apply the frozen calibration and review threshold;
neutral predictions always require analyst review following the final-test risk
finding. Stage 5 is complete for the scoped local demonstration. Browser-based
accessibility and screenshot evidence was not produced and remains an explicit limitation. The full preparation run
reduced 5,326,143 source reviews to a frozen 48,000-row modelling dataset.
DistilBERT achieved test macro-F1 0.6844 compared with 0.6017 for TF-IDF, while
the local CPU evidence records the corresponding latency and size trade-off.
Aggregate metrics and report figures are public; source reviews, row-level
predictions and model weights remain excluded.

## Data-pipeline interface

Callers construct `PipelineConfig` from the frozen YAML configuration and invoke
one public operation:

```python
from amazon_sentiment import PipelineConfig, prepare

config = PipelineConfig.from_yaml(
    "config/experiment.yaml",
    raw_dir="data/raw",
    output_dir="data/generated",
)
bundle = prepare(config)
print(bundle.dataset_path)
```

`prepare(config)` validates and minimises the source fields, removes empty and
duplicate reviews, constructs rating-derived labels, performs a many-to-one left
join on `parent_asin`, creates chronological splits and writes Parquet plus JSON
evidence reports.

## Reproduce the local stages

```bash
.venv/bin/python -m amazon_sentiment download \
  --config config/experiment.yaml --raw-dir data/raw
.venv/bin/python -m amazon_sentiment prepare \
  --config config/experiment.yaml --raw-dir data/raw \
  --output-dir data/generated
.venv/bin/python -m amazon_sentiment benchmark \
  --config config/storage_benchmark.yaml --raw-dir data/raw \
  --work-dir data/benchmark --output-dir artifacts/metrics
.venv/bin/python -m amazon_sentiment baselines \
  --config config/experiment.yaml \
  --dataset data/generated/processed/training_dataset.parquet \
  --split-manifest data/generated/manifests/split_manifest.json \
  --model-dir models/baselines --metrics-dir artifacts/metrics
```

Generated source data, row-level manifests and model files stay outside Git.
Aggregate baseline metrics are written to `artifacts/metrics/` for report evidence.

## DistilBERT training

Install the optional tokenizer stack locally, then measure token coverage without
reading the policy-calibration or test subsets:

```bash
.venv/bin/python -m pip install -r requirements-transformer.txt
.venv/bin/python -m amazon_sentiment token-length \
  --config config/distilbert.yaml \
  --dataset data/generated/processed/training_dataset.parquet \
  --output-dir artifacts/metrics --cache-dir models/huggingface
```

The aggregate result selected 256 tokens. The optional throughput pilot was
skipped; `artifacts/metrics/distilbert_training_decision.json` records the
conservative physical batch and this unmeasured limitation. Open
`notebooks/train_distilbert_colab.ipynb` in a GPU Colab runtime to run the
required two-epoch fine-tune. The notebook expects the governed Parquet handoff
in private Google Drive, writes resumable checkpoints and the final private
bundle there, and never commits or pushes model weights. Reproducible Hugging
Face downloads use Colab's local filesystem rather than the mounted Drive;
this avoids persisting partial or corrupted cache blobs between runtimes.
It recreates a project-only dependency directory and installs every explicitly
pinned package there without resolving transitive dependencies. The directory
is exposed to training subprocesses through `PYTHONPATH`. This avoids Colab's
unavailable `venv`/`ensurepip` path and prevents pip from replacing one half of
Colab's matched Torch/torchvision/CUDA stack, so no runtime restart is required.

## Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pytest
```

For local DistilBERT calibration and evaluation, install the matching CPU-only
Torch build in the same environment:

```bash
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python -m amazon_sentiment calibrate-models
```

The calibration command reads only `validation_policy_calibration`, fits
temperature scaling for DistilBERT and sigmoid calibration for the frozen
TF-IDF model, and locks the 90% selective-accuracy review thresholds. It does
not read the protected test split.

After the calibration evidence is reviewed and committed, the protected test
evaluation is intentionally a one-shot command:

```bash
.venv/bin/python -m amazon_sentiment evaluate-test
```

It validates every frozen input hash, reads exactly the 12,000 `test` rows,
applies the frozen calibrators and thresholds, and writes aggregate metrics,
paired bootstrap intervals and declared slice metrics. If
`artifacts/metrics/final_test_metrics.json` already exists, the command refuses
to overwrite it. Row-level predictions remain under the ignored
`artifacts/predictions/` directory.

After that one-time result exists, build the local CPU efficiency benchmark,
deployment recommendation and report figures without reopening or overwriting
the test metric:

```bash
.venv/bin/python -m amazon_sentiment evaluation-evidence
```

The batch-one benchmark protocol is versioned separately in
`config/evaluation_evidence.yaml` so the frozen experiment configuration is not
changed after testing. Aggregate JSON and SVG files are public. Manual qualitative
error coding is outside the final project scope.

The workspace uses an isolated virtual environment with pinned runtime and
development requirements. `pip check` should report no broken dependencies.

## Local inference interface

Install the pinned local inference and Gradio environment, then launch the app:

```bash
.venv/bin/python -m pip install -r requirements-interface.txt
.venv/bin/python -m pip install -e .
.venv/bin/python app.py
```

The server binds to `127.0.0.1:7860` and never creates a public Gradio share
link. The app checks the local model hash against its calibration metadata at
startup, shows all calibrated class probabilities, and displays either
`Automatic route` or `Human review required` in text. Review text is not stored
in the operational log. The ignored JSONL log contains a generated request ID,
timestamp, model/config/policy versions, input lengths, prediction, confidence,
review decision and latency.

Paths can be changed with `--model-dir`, `--thresholds`,
`--deployment-policy`, `--log-path` and `--port`. The corresponding environment
variables are `SENTIMENT_CONFIG`, `SENTIMENT_MODEL_DIR`, `SENTIMENT_THRESHOLDS`,
`SENTIMENT_DEPLOYMENT_POLICY`, `SENTIMENT_LOG_PATH` and `SENTIMENT_PORT`.

Run the offline inference and UI construction checks with:

```bash
.venv/bin/python -m pytest -q tests/test_inference.py
```

## Repository layout

```text
config/experiment.yaml          Frozen experiment decisions
src/amazon_sentiment/           Python package
tests/fixtures/                 Synthetic, non-Amazon test records
tests/                          Behavioural tests for acquisition, preparation,
                                benchmarking and baseline training
artifacts/                      Aggregate public evidence generated later
data/                           Ignored source and processed records
models/                         Ignored local model bundles
docs/                           Ignored local working documentation
```

No Git commits are created automatically during implementation.
