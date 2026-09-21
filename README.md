# Amazon review sentiment artefact

This repository contains the implementation and public evidence for an academic
artefact that compares majority-class and TF-IDF baselines with a fine-tuned
DistilBERT model for three-class Amazon review sentiment classification.

The public repository intentionally excludes source reviews, processed datasets,
row-level predictions, operational logs, credentials, checkpoints and model
weights. See `.gitignore` before adding generated files.

## Current status

The governed dataset and both local baselines have been produced. The full
preparation run reduced 5,326,143 source reviews to a frozen 48,000-row modelling
dataset. The selected TF-IDF model was chosen only on the model-selection
validation subset; the test subset remains untouched. The storage benchmark
selected Zstandard Parquet with 65,536-row groups for the production handoff,
so the data-foundation exit gate is complete.

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

## DistilBERT pilot

Install the optional tokenizer stack locally, then measure token coverage without
reading the policy-calibration or test subsets:

```bash
.venv/bin/python -m pip install -r requirements-transformer.txt
.venv/bin/python -m amazon_sentiment token-length \
  --config config/distilbert.yaml \
  --dataset data/generated/processed/training_dataset.parquet \
  --output-dir artifacts/metrics --cache-dir models/huggingface
```

The aggregate result selects a provisional sequence length. Open
`notebooks/train_distilbert_colab.ipynb` in a GPU Colab runtime to benchmark
actual training steps and freeze the sequence length and physical batch size.
The notebook expects the governed Parquet handoff in private Google Drive; it
does not contain preparation logic and never commits or pushes changes.

## Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pytest
```

The workspace uses an isolated virtual environment with pinned runtime and
development requirements. `pip check` should report no broken dependencies.

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
