# Model card: Amazon review sentiment triage

Version: 1.0
Date: 23 September 2026
Status: frozen academic demonstration
Default model: DistilBERT
Fallback model: TF-IDF with logistic regression

## Summary

This model classifies one Amazon review body as negative, neutral or positive for an e-commerce insights analyst. It supports a review-triage decision: eligible high-confidence negative and positive predictions may be routed automatically, while low-confidence predictions and every neutral prediction require human review.

The artefact is a local academic demonstration, not a production moderation, customer-satisfaction or product-quality system. DistilBERT is the default because it achieved the strongest final-test macro-F1 and materially better neutral-class F1. TF-IDF remains the lower-cost comparison and conceptual fallback.

## Model details

| Field | Value |
|---|---|
| Architecture | `distilbert/distilbert-base-uncased` with a three-class sequence-classification head |
| Immutable base revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Input | Review body text only |
| Output labels | `negative`, `neutral`, `positive` |
| Label rule | Ratings 1–2 negative; rating 3 neutral; ratings 4–5 positive |
| Maximum sequence length | 256 tokens |
| Calibration | Scalar temperature scaling, temperature 1.2755715225830961 |
| Routing threshold | 0.694306130317525 |
| Additional UI rule | Every predicted neutral review requires human review |
| Training seed | 42 |
| Training | Two epochs; batch 8; gradient accumulation 4; effective batch 32 |
| Training hardware | One Tesla T4 in Google Colab |
| Training time | 329.5 seconds |
| Serving environment measured | Local CPU, Python 3.10.12, Torch 2.11.0 CPU |
| Public weights | No. Model weights remain in ignored local/private storage |

The application validates the model-weight hash against the calibration metadata at startup. It also validates label order, calibration provenance, threshold provenance and deployment-policy compatibility.

## Intended use

The intended user is an e-commerce insights analyst using the model to triage English-language review text for later human analysis.

Appropriate uses are:

- demonstrating a governed end-to-end classification workflow;
- comparing a majority baseline, TF-IDF and DistilBERT;
- showing calibrated probabilities and an explicit review decision;
- routing uncertain cases to an analyst in a local, non-production setting;
- supporting academic discussion of accuracy, calibration, robustness, efficiency and governance trade-offs.

The model output is decision support. It is not a verified fact about the reviewer, product, seller or underlying cause of the sentiment.

## Uses outside scope

Do not use this model for:

- automatic moderation, deletion, enforcement or customer sanctions;
- individual profiling, identity inference or fake-review detection;
- production deployment without prospective validation, monitoring and a stakeholder-approved service level;
- multilingual classification or claims of language-wide performance;
- aspect-level sentiment, product recommendation or customer-satisfaction measurement;
- safety-critical, legal, employment, credit or similarly consequential decisions;
- claims of demographic fairness;
- commercial redistribution of the source reviews or private model bundle.

## Training and evaluation data

The source is Amazon Reviews 2023, published by McAuley Lab at UCSD for research access. The project uses the All Beauty and Video Games categories for a non-commercial academic purpose. Public availability is not described as an open licence for the underlying Amazon content.

The governed preparation pipeline minimises fields, drops persistent reviewer identifiers, normalises and deduplicates review text, removes conflicting-label duplicates, joins product metadata using `parent_asin`, and uses a chronological split.

| Partition | Rows | Treatment |
|---|---:|---|
| Training | 30,000 | Balanced across category and class cells |
| Validation, model selection | 3,000 | Earlier validation half |
| Validation, policy calibration | 3,000 | Later validation half |
| Test | 12,000 | Natural prevalence, opened once after model and policy freezing |
| Total modelling dataset | 48,000 | Frozen governed dataset |

The preferred chronological split trains through 31 December 2021, validates during 2022 and tests from 1 January 2023. Validation and test preserve their natural prevalence. Ratings generate labels but are not model features.

## Evaluation results

The primary metric is macro-F1 because class prevalence is uneven and neutral performance matters to the triage decision.

| Model | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.6954 | 0.2734 | 0.0000 | 0.0000 | 0.8203 |
| TF-IDF logistic regression | 0.8380 | 0.6017 | 0.7622 | 0.1347 | 0.9083 |
| DistilBERT | 0.7976 | 0.6844 | 0.7957 | 0.3657 | 0.8917 |

DistilBERT minus TF-IDF macro-F1 was 0.0826. The paired bootstrap 95% interval was 0.0712 to 0.0947. This supports the scoped choice of DistilBERT despite TF-IDF having higher raw accuracy on the positive-heavy test distribution.

## Calibration and routing

Temperature scaling was fitted on the 3,000-row policy-calibration subset, not on test data. On the final test set, calibrated DistilBERT recorded:

- expected calibration error: 0.0623;
- multiclass Brier score: 0.2765;
- negative log-likelihood: 0.5023;
- frozen-threshold coverage: 73.16%;
- frozen-threshold selective accuracy: 90.81%.

The global threshold concealed weak class-specific behaviour. Among true neutral rows automatically routed under the original frozen threshold, selective accuracy was 69.76%. The interface therefore adds mandatory review for every predicted neutral result.

That mandatory-neutral rule is a post-test mitigation. It has not been evaluated on a new holdout or prospective sample and must not be described as proving a 90% production accuracy guarantee.

## Efficiency

The local CPU benchmark used batch size one, ten warm-up runs and one hundred measured runs over a deterministic 100-row sample.

| Model | Median latency | p95 latency | Serving size |
|---|---:|---:|---:|
| TF-IDF | 2.18 ms | 3.66 ms | 6.08 MiB |
| DistilBERT | 28.39 ms | 83.91 ms | 256.12 MiB |

DistilBERT was approximately 13 times slower at the median and 42 times larger, but remained interactive on the recorded machine. No stakeholder latency service level was available.

## Limitations and risks

- Star ratings are distant labels, not independently verified sentiment annotations.
- Neutral F1 is substantially lower than negative and positive F1.
- The source covers only two Amazon categories and a historical period ending in 2023.
- No explicit language-detection or multilingual evaluation was performed.
- Inputs longer than 256 tokens are truncated. The development-set token analysis measured 95.17% coverage at 256 tokens.
- Training prevalence is deliberately balanced, while validation and test preserve natural prevalence.
- Slice analysis covers operational data properties, not protected demographic groups.
- Manual qualitative error coding was excluded from the final scope.
- The global 90% selective-accuracy target was illustrative and not stakeholder validated.
- The revised mandatory-neutral UI policy lacks prospective validation.
- Latency describes one WSL2 CPU environment and does not generalise to other hardware.
- Accessibility was audited on 24 September 2026 in one environment only: Windows, Microsoft Edge 153, Windows Narrator and Lighthouse (tool versions not recorded). Keyboard, accessibility-tree, screen-reader, automated scan, 200% zoom and 320 CSS-pixel viewport checks passed after three fixes. NVDA, JAWS, VoiceOver, mobile devices and users with access needs were not tested, Gradio footer images lack suitable alternative text, and the project does not claim WCAG conformance.
- Model weights are deliberately absent from the public repository, so an authorised private bundle is required for real inference.

## Privacy, governance and logging

The preparation pipeline removes `user_id` and other prohibited fields before writing curated data. Raw reviews, processed rows, row-level predictions, operational logs and model weights are excluded from the public repository.

The local interface does not write submitted review text to its operational log. It records a generated request ID, timestamp, model/configuration/policy versions, input lengths, prediction, calibrated confidence, routing decision and latency. The server binds to `127.0.0.1` and does not create a public Gradio share link.

## Monitoring and review

For any future deployment, monitor:

- prediction and review-routing volumes by class;
- confidence and human-review rates;
- schema or source changes;
- temporal and category-level performance drift;
- calibration and selective-accuracy drift;
- latency and load failures;
- unexpected changes in neutral prevalence;
- privacy incidents or accidental raw-text retention.

Retraining or policy review should be triggered by a source/schema change, material temporal degradation, category expansion, failed calibration/routing targets or a change to the business decision.

## Evidence sources

- `config/experiment.yaml`
- `config/distilbert.yaml`
- `models/distilbert-run/{run_identity,training_summary,environment,calibration}.json`
- `artifacts/metrics/{policy_calibration_summary,review_thresholds,final_test_metrics,inference_benchmark,deployment_recommendation}.json`
- `artifacts/metrics/final_test_slice_metrics.csv`
- `artifacts/figures/`
- `artifacts/accessibility/accessibility-checklist.md`

## Contact and ownership

The academic artefact owner is responsible for the private model bundle, evaluation evidence, final report claims and any decision to retrain, replace or retire the model. No production support commitment is implied.
