# Scoped accessibility checklist

Audit date: 23 September 2026
Interface: local Gradio 6.28.0 app on `127.0.0.1`
Scope: Stage 5 implementation, automated checks and live local endpoint smoke test
Claim boundary: this checklist does not claim WCAG conformance.
Stage status: complete for the scoped local demonstration, with browser-dependent checks retained as explicit untested limitations.
Accessibility test procedure: `docs/runbook.md`, section “Accessibility test setup”.

| Check | Status | Evidence | Limitation or future action |
|---|---|---|---|
| Review input has a persistent visible label and programmatic component label | Pass (implementation) | `Review text` is the Gradio Textbox label; `tests/test_inference.py` inspects the generated app configuration | Confirm the accessible name in browser accessibility tools |
| Submit and clear controls are present | Pass (implementation) | Offline Gradio construction test finds `Predict sentiment` and `Clear` | Verify keyboard order and activation in a browser |
| Empty input returns a nearby text error | Pass (automated) | Interface test asserts a visible `role="alert"` input error | Confirm announcement with a screen reader |
| Result state does not rely on colour | Pass (implementation) | Result uses icon plus the text `Automatic route` or `Human review required` | Confirm rendered icon is ignored and route text is announced |
| Calibrated probabilities have a text alternative | Pass (implementation) | Semantic table with caption, column headers and row headers | Inspect the rendered accessibility tree |
| Dynamic result announcement | Pass (implementation) | Result region uses `aria-live="polite"`; errors use `role="alert"` | Confirm announcement timing with a screen reader |
| Default text contrast | Pass (calculated) | `#17202a` on `#ffffff`: 16.45:1 | Recheck computed browser styles after launch |
| Human-review state contrast | Pass (calculated) | `#542500` on `#fff4e5`: 11.77:1 | Recheck computed browser styles after launch |
| Automatic-route state contrast | Pass (calculated) | `#0b4725` on `#eefaf2`: 10.09:1 | Recheck computed browser styles after launch |
| Error-state contrast | Pass (calculated) | `#5f1410` on `#fff4f2`: 12.22:1 | Recheck computed browser styles after launch |
| Focus-indicator contrast | Pass (calculated) | `#111111` against `#ffffff`: 18.88:1; three-pixel outline with three-pixel offset | Confirm every interactive element receives this style |
| Local-only serving | Pass (functional) | App served at `127.0.0.1:7861`; Gradio client completed a real neutral prediction; `share=False` is explicit | Retain loopback binding and do not add public sharing |
| Keyboard-only input, submit, result reading and reset | Not tested | No graphical browser is installed in the workspace | Run manually in the target browser and retain focus screenshots |
| Visible focus throughout keyboard flow | Not tested | No graphical browser is installed in the workspace | Capture input, submit, result and clear focus states |
| Reading order and programmatic labels in accessibility tree | Not tested | No browser accessibility inspector is available | Inspect the rendered tree and record any Gradio-generated naming gaps |
| Screen-reader announcement | Not tested | No screen reader is available in the workspace | Test with NVDA or an available equivalent and record browser/version |
| Automated axe or Lighthouse scan | Not tested | Neither a supported browser nor axe/Lighthouse is installed | Run against the local server and save the exported report |
| 200% zoom | Not tested | Requires a graphical browser | Confirm input, actions and result remain present and usable |
| Narrow viewport | Not tested | Requires a graphical browser | Test at 320 CSS pixels; the custom definition list already collapses to one column |

## Functional evidence

The offline suite covers calibrated probability calculation, frozen threshold
application, mandatory review for neutral predictions, empty-input handling,
model/calibration hash compatibility, privacy-safe logging, semantic result
content and Gradio control construction. A live local Gradio client request also
completed against the real DistilBERT bundle and returned the required neutral
review state.

No browser screenshots or browser-based assistive-technology results are included in the repository. Stage 5 is complete for the scoped local demonstration on the strength of implementation, automated tests and the live local endpoint smoke check; the unperformed browser checks remain declared limitations.

## Known limitations

Gradio generates the outer document, component wrappers, queue status and other
markup. The application supplies persistent labels, semantic result HTML,
high-contrast states and live regions, but those choices do not establish that
all framework-generated markup is accessible. Keyboard behavior, accessible
names, reading order, live announcements, zoom and narrow-viewport behaviour
remain untested. Stage 5 is closed under a documented scope exception, so these
items must not be described as passed and this checklist must not be used to claim WCAG conformance.
