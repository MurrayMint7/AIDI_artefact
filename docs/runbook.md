# Operational runbook: Amazon review sentiment triage

Version: 1.0
Date: 23 September 2026
Scope: existing local academic-demo workspace

## Purpose

This runbook explains how to validate, start, inspect and troubleshoot the local Gradio sentiment-triage application. It also defines the evidence boundary for accessibility and handover.

This is not a fresh-clone installation rehearsal. The documented commands were validated in the existing workspace. Model weights, processed data and row-level predictions are private or ignored and are not recoverable from the public repository alone.

## System overview

The serving path is:

1. `app.py` resolves configuration and private artefact paths.
2. `InferenceConfig` validates labels, calibration provenance, threshold provenance, the model hash and the deployment policy.
3. `SentimentPredictor.predict(text)` runs local DistilBERT inference, applies temperature scaling and constructs calibrated probabilities.
4. The routing policy requires review below the frozen threshold and for every predicted neutral result.
5. The Gradio adapter renders the result and writes privacy-safe metadata to an ignored JSONL log.

The server binds to `127.0.0.1`, uses `share=False`, does not open a browser automatically and does not require API credentials.

## Required local files

The real application requires these files:

- `config/experiment.yaml`;
- `models/distilbert-run/model/config.json`;
- `models/distilbert-run/model/model.safetensors`;
- the tokenizer files inside `models/distilbert-run/model/`;
- `models/distilbert-run/calibration.json`;
- `artifacts/metrics/review_thresholds.json`;
- `artifacts/metrics/deployment_recommendation.json`.

The `models/` directory is intentionally ignored. Obtain the authorised model bundle through the private handoff rather than committing it or downloading an arbitrary replacement.

## Environment preparation

Use the existing Python 3.10 virtual environment:

~~~bash
.venv/bin/python -m pip install -r requirements-interface.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
~~~

Expected dependency-check result:

~~~text
No broken requirements found.
~~~

Do not describe this as independent installation validation. A fresh-clone rehearsal is outside the final scope.

## Pre-start validation

Run the full automated suite:

~~~bash
.venv/bin/python -m pytest -q
~~~

Run the focused inference and interface suite:

~~~bash
.venv/bin/python -m pytest -q tests/test_inference.py
~~~

The focused suite verifies:

- calibrated probability calculation;
- confidence-threshold routing;
- mandatory review for neutral predictions;
- empty-input rejection;
- model/calibration hash compatibility;
- privacy-safe logging;
- semantic result content;
- Gradio label and control construction.

These tests verify implementation behaviour. They do not verify rendered keyboard order, screen-reader announcements, browser accessibility trees, zoom or narrow-viewport behaviour.

## Start and stop

Start the application:

~~~bash
.venv/bin/python app.py
~~~

Open:

~~~text
http://127.0.0.1:7860
~~~

Stop the server with `Ctrl+C` in its terminal.

To use a different port:

~~~bash
.venv/bin/python app.py --port 7861
~~~

## Runtime options

| Command option | Environment variable | Default |
|---|---|---|
| `--config` | `SENTIMENT_CONFIG` | `config/experiment.yaml` |
| `--model-dir` | `SENTIMENT_MODEL_DIR` | `models/distilbert-run` |
| `--thresholds` | `SENTIMENT_THRESHOLDS` | `artifacts/metrics/review_thresholds.json` |
| `--deployment-policy` | `SENTIMENT_DEPLOYMENT_POLICY` | `artifacts/metrics/deployment_recommendation.json` |
| `--log-path` | `SENTIMENT_LOG_PATH` | `artifacts/logs/inference.jsonl` |
| `--port` | `SENTIMENT_PORT` | `7860` |

Do not set a public bind address or enable Gradio sharing for this artefact.

## Five-minute smoke check

1. Start the app and confirm the terminal shows a local URL without a public share URL.
2. Open the page and confirm the heading, `Review text` field, `Predict sentiment` button and `Clear` button are present.
3. Submit an empty value. Confirm a visible `Input error` message appears.
4. Enter synthetic text and submit it. Confirm the page shows:
   - predicted sentiment;
   - calibrated confidence;
   - all three class probabilities;
   - model version;
   - either `Automatic route` or `Human review required`.
5. If the result is neutral, confirm it requires review regardless of confidence.
6. Select `Clear` and confirm the input, validation message and result return to their initial state.
7. Inspect the newest JSONL log entry and confirm the submitted text is absent.

Do not use a real identifiable review for demonstration. Use synthetic text or an already governed held-out example.

## Accessibility: what already exists

The following support is already implemented in `src/amazon_sentiment/interface.py`; it is not a proposed future addition.

| Implemented support | Where |
|---|---|
| Persistent visible input label | Gradio textbox label `Review text` |
| Native submit and clear controls | Gradio buttons |
| Text alternative to colour | `Automatic route` or `Human review required`, plus an icon |
| Decorative icon hidden from assistive technology | `aria-hidden="true"` |
| Dynamic result region | `aria-live="polite"` |
| Error announcement semantics | `role="alert"` |
| Semantic probabilities | Captioned table with column and row headers |
| Structured result details | Definition list |
| Visible focus styling | Three-pixel dark outline with offset |
| Calculated high-contrast states | Default, review, automatic and error colour pairs |
| Narrow-layout CSS | Definition list changes to one column below 35 rem |
| Local processing notice | Textbox help text |
| Local-only serving | Loopback address and `share=False` |

The automated tests assert the key labels, controls, alert markup, live-region markup, semantic table and non-colour routing text.

What is not yet evidenced is the behaviour generated by the browser and Gradio around that markup. Testing it adds evidence, not the underlying settings. A test failure may reveal that a code change is needed.

## Accessibility test setup

Use a graphical browser on the same machine that can reach `http://127.0.0.1:7860`. Record:

- date and tester;
- operating system;
- browser and version;
- Gradio version;
- screen reader and version, if used;
- zoom and viewport settings;
- exact result, pass or fail;
- defect and mitigation, if applicable.

Record results in `artifacts/accessibility/accessibility-checklist.md`. Do not convert an untested item into a pass.

### 1. Keyboard-only test

Reload the page and do not use the mouse.

1. Press `Tab` until focus reaches `Review text`.
2. Confirm a clearly visible focus indicator.
3. Type synthetic review text.
4. Press `Tab` to reach `Predict sentiment`; activate it with `Enter` or `Space`.
5. Confirm the result can be reached and read in a logical order.
6. Continue with `Tab` and `Shift+Tab`. Confirm focus is never trapped or lost.
7. Reach `Clear`, activate it from the keyboard and confirm the form resets.
8. Repeat with an empty input and confirm the error is visible and reachable.

Pass only if every mouse-available action is keyboard-operable, focus remains visible and the order is logical. W3C identifies keyboard availability and absence of keyboard traps as core checks.

### 2. Browser accessibility-tree test

In Chromium-based browsers:

1. Open Developer Tools with `F12`.
2. Use the Elements panel and select the textbox, both buttons, validation area and result area.
3. Open the Accessibility pane.
4. Check the computed name and role:
   - textbox name is `Review text`;
   - buttons are named `Predict sentiment` and `Clear`;
   - errors expose alert semantics;
   - the result exposes readable headings, status, details and table structure.
5. Confirm the decorative route icon is not exposed as meaningful content.
6. Inspect the document tree for duplicated, empty or misleading names introduced by Gradio.

Chrome's official guidance treats markup inspection and keyboard/screen-reader use as complementary checks, not substitutes for one another.

### 3. Screen-reader test

On Windows, NVDA is a suitable free option.

1. Start NVDA, then open the application in Chrome or Firefox.
2. Navigate from the page heading to the textbox and controls.
3. Confirm NVDA announces the textbox as `Review text` and announces both button names.
4. Submit an empty input. Confirm `Input error` is announced without moving unpredictably.
5. Submit synthetic text. Confirm the updated result is announced or can immediately be found and read.
6. Confirm the routing state, predicted sentiment, confidence and probability table are understandable without seeing colour or icons.
7. Activate `Clear` and confirm the reset does not leave stale result content announced.

Record the exact browser/NVDA versions and what was actually announced. The official NVDA user guide is available from NV Access.

### 4. Automated Lighthouse or axe scan

For Lighthouse in Chrome:

1. Open Developer Tools.
2. Open the Lighthouse panel.
3. Select Accessibility and analyse the local page.
4. Save the report or record every failing audit and affected element.
5. Repeat after rendering a successful result and an error state, because dynamic content may differ from the initial page.

Alternatively, install the axe DevTools extension for Chrome, Edge or Firefox, open its developer-tools panel and scan the page.

Automated results are not a conformance decision. Lighthouse excludes manual audits from its accessibility score, and Chrome explicitly states that keyboard and screen-reader testing must be performed manually.

### 5. Contrast verification

For default text, automatic-route, human-review, error and focus states:

1. Inspect the rendered element in Developer Tools.
2. Confirm the computed foreground and background colours match the checklist.
3. Use the browser contrast indicator or axe to record the observed ratio.
4. Check normal text, large status text, borders and focus indicators separately.
5. Record any browser or theme override.

The repository contains calculated ratios, but rendered-style inspection is needed to confirm Gradio does not override them.

### 6. Two-hundred-percent zoom

1. Set the browser viewport to a normal desktop width.
2. Set page zoom to 200%.
3. Confirm the input, both buttons, validation message, result status, details and probability table remain visible and usable.
4. Confirm text is not clipped or overlapped and no action becomes inaccessible.
5. Submit and clear a result while still at 200%.

W3C's published test is to zoom to 200% and confirm that all content and functionality remain available.

### 7. Narrow viewport

1. Open the browser's responsive/device toolbar.
2. Set the viewport width to 320 CSS pixels.
3. Test the initial, error and successful-result states.
4. Confirm there is no clipped text or control, no unusable horizontal overflow and no overlapping content.
5. Confirm the definition list becomes one column and the probability table remains readable or scrollable without losing context.
6. Complete submit and clear actions at that width.

### 8. Record the result

For each row in the accessibility checklist:

- replace `Not tested` only after performing the check;
- record Pass or Fail, never an inferred pass;
- add browser, operating system, tool/version and date;
- reference any saved report or screenshot;
- describe a mitigation for failures;
- keep the statement that this is a scoped audit and not a WCAG conformance claim.

Useful primary guidance:

- W3C WCAG 2.2 Quick Reference: https://www.w3.org/WAI/WCAG22/quickref/
- W3C Easy Checks: https://www.w3.org/WAI/test-evaluate/easy-checks/
- W3C 200% zoom technique: https://www.w3.org/WAI/WCAG21/Techniques/general/G142
- Chrome accessibility reference: https://developer.chrome.com/docs/devtools/accessibility/reference
- Chrome Lighthouse accessibility scoring: https://developer.chrome.com/docs/lighthouse/accessibility/scoring
- axe DevTools extension: https://www.deque.com/axe/devtools/extension/
- NVDA user guide: https://download.nvaccess.org/documentation/en/userGuide.html

## Logs and privacy

The default operational log is `artifacts/logs/inference.jsonl`. It is ignored by Git.

Permitted fields include:

- generated request ID and UTC timestamp;
- model, configuration and policy versions;
- input character and token counts;
- predicted label and probabilities;
- calibrated confidence;
- review decision and reason;
- threshold and latency.

The submitted review text must never appear. Check with a synthetic unique phrase:

~~~bash
rg -n "YOUR_UNIQUE_SYNTHETIC_PHRASE" artifacts/logs/inference.jsonl
~~~

Expected result: no output.

## Common failures

| Symptom | Likely cause | Action |
|---|---|---|
| Missing-file error at startup | Private model or evidence bundle absent | Verify every required local file and the configured paths |
| Model hash mismatch | Weights differ from the calibrated model | Restore the exact authorised bundle; do not edit the hash |
| Label mismatch | Model `id2label` differs from the frozen configuration | Restore the matching model/configuration pair |
| Invalid threshold bundle | Wrong stage, split or test flag | Use the frozen `review_thresholds.json` generated from policy calibration |
| Deployment policy rejected | Policy does not select DistilBERT or has no mandatory-review label | Restore the frozen deployment recommendation |
| Port already in use | Another process owns port 7860 | Stop it or start with `--port 7861` |
| Gradio import failure | Interface dependencies absent | Install `requirements-interface.txt` in the existing environment |
| Torch/Transformers import failure | Evaluation runtime dependencies absent | Install the pinned evaluation requirements in the existing environment |
| Prediction unavailable | Runtime/model exception | Read the terminal traceback; the browser intentionally receives a generic error |
| Log contains unexpected content | Logging contract regression | Stop the app, preserve the log privately and run `tests/test_inference.py` |
| Keyboard or screen-reader failure | Browser/Gradio behaviour differs from semantic intent | Record a failed checklist item, identify the generated element and patch/test the adapter where possible |

## Recovery and rollback

The application is fail-closed at startup when model, calibration, label or policy files do not match. Restore the last complete, hash-verified private DistilBERT bundle and its corresponding threshold/policy artefacts together.

TF-IDF is the documented efficiency fallback, but `app.py` does not currently expose a hot model switch or TF-IDF serving adapter. Do not claim an automatic rollback. Using TF-IDF in the UI would require a governed adapter implementing the same prediction contract and a separately approved deployment policy.

Do not overwrite `artifacts/metrics/final_test_metrics.json`; the protected test evaluation is intentionally one-shot.

## Retraining and policy-review triggers

Review or retrain when:

- the source schema or label policy changes;
- a new category or language is introduced;
- temporal monitoring shows material performance degradation;
- calibration or selective accuracy no longer meets an approved target;
- neutral prevalence or routing volume changes materially;
- the base model, tokenizer or dependency stack changes;
- privacy, licensing or governance assumptions change.

A retrained model is a new version. Refit calibration and thresholds on validation data, generate new hashes and evidence, and do not reuse the previous model's test claim or policy without reevaluation.

## Handover checklist

- [ ] Required private model files are present.
- [ ] Model/calibration/policy startup validation passes.
- [ ] Full current-workspace tests pass.
- [ ] Focused interface tests pass.
- [ ] Local endpoint smoke check passes.
- [ ] Submitted text is absent from the operational log.
- [ ] Server binds only to loopback and sharing is disabled.
- [ ] Accessibility evidence is described only to the level actually tested.
- [ ] Model card limitations are reflected in the report.
- [ ] Evidence-index paths and statuses are current.
- [ ] No raw data, row-level predictions, logs or weights are staged for Git.
- [ ] Owner knows how to restore the private bundle and stop the service.

Fresh-clone installation rehearsal, manual qualitative error coding, screenshots and browser-dependent accessibility evidence are outside the final committed scope unless the owner later chooses to perform and record them.
