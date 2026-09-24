# Scoped accessibility checklist

Implementation audit date: 23 September 2026
Browser audit completed: 24 September 2026
Browser environment: Windows; Microsoft Edge 153.0.4234.32
Interface: local Gradio 6.28.0 app on `127.0.0.1`
Scope: Stage 5 implementation, automated checks, live local endpoint smoke test and completed browser checks
Claim boundary: this checklist does not claim WCAG conformance.
Stage status: complete for the scoped local demonstration, including the planned browser-dependent checks.
Accessibility test procedure: `docs/runbook.md`, section “Accessibility test setup”.

| Check | Status | Evidence | Limitation or future action |
|---|---|---|---|
| Review input has a persistent visible label and programmatic component label | Pass (manual, 24 September 2026) | Edge exposed role `textbox` with the accessible name `Review text Required. Review text is processed locally and is not written to the operational log.` | Retain the visible label and useful description |
| Submit and clear controls are present | Pass (manual, 24 September 2026) | Edge exposed role `button` with names `Predict sentiment` and `Clear`; keyboard activation also passed | Retain current names and order |
| Initial browser load and visible controls | Pass (manual, 24 September 2026) | Windows; Microsoft Edge 153.0.4234.32; page loaded successfully with review textbox, Predict, Clear and Prediction result visible; no initial problems observed | Continue the remaining checks in the same environment |
| Empty input returns a nearby text error | Pass (manual, 24 September 2026) | Windows Narrator automatically announced the visible role-`alert` empty-input error | Retain the alert markup |
| Result state does not rely on colour | Pass (manual, 24 September 2026) | Narrator announced the routing status and ignored the decorative icon; visual light/dark checks also passed | Retain visible routing text and hidden decorative icon |
| Calibrated probabilities have a text alternative | Pass (Narrator after fix, 24 September 2026) | After adding an atomic linear summary, Narrator automatically announced all three classes and percentages; the semantic table remained available for navigation | Retain the linear summary, atomic live region and semantic table |
| Dynamic result announcement | Pass (Narrator after fix, 24 September 2026) | Narrator automatically announced the error, result, routing status and complete probability summary without unexpected focus movement | Retain the atomic polite live region |
| Prediction-result contrast across light and dark themes | Pass (manual after fix, 24 September 2026) | Windows/Edge: light mode passed; the first dark-mode run exposed an unreadable nested automatic-route label; after the CSS inheritance fix, empty, completed and error states and both routing labels were readable in dark mode | Retain the regression test and theme-aware CSS |
| Human-review state contrast | Pass (manual, 24 September 2026) | `#542500` on `#fff4e5` is calculated at 11.77:1; the label was readable in Edge dark mode | Retain current colours |
| Automatic-route state contrast | Pass (manual after fix, 24 September 2026) | `#0b4725` on `#eefaf2` is calculated at 10.09:1; the initial nested-label failure was fixed and both label and icon were readable on retest in Edge dark mode | Retain the nested colour-inheritance regression assertion |
| Error-state contrast | Pass (manual, 24 September 2026) | `#5f1410` on `#fff4f2` is calculated at 12.22:1; the error was readable in Edge light and dark modes | Retain current colours |
| Focus indicator | Pass (manual, 24 September 2026) | Focus was visible throughout the tested control sequence in both light and dark modes | Retain the theme-aware three-pixel outline and offset |
| Local-only serving | Pass (functional) | App served at `127.0.0.1:7861`; Gradio client completed a real neutral prediction; `share=False` is explicit | Retain loopback binding and do not add public sharing |
| Keyboard-only input, submit, result reading and reset | Pass (manual, 24 September 2026) | Edge tab order was review input, Predict, Clear, then footer tabs left-to-right; typing, prediction and clearing all worked; Shift+Tab worked and no keyboard trap occurred | Retain this order and repeat after major Gradio upgrades |
| Visible focus throughout keyboard flow | Pass (manual, 24 September 2026) | Focus was visible in both light and dark modes for the tested keyboard flow | Retain theme-aware focus styling |
| Reading order and programmatic labels in accessibility tree | Pass (manual, 24 September 2026) | Edge exposed correct textbox/button roles and names, a polite live result region, logical result order, table caption/headers, hidden decorative icon and alert error | Retain semantic markup and repeat after major Gradio upgrades |
| Screen-reader announcement | Pass after fix (Windows Narrator, version not recorded, 24 September 2026) | Textbox/buttons, error, result, routing status and all probability values were announced understandably; the semantic table remained navigable, icon was ignored and focus stayed stable | Repeat with the supported assistive-technology matrix after major UI changes |
| Automated axe or Lighthouse scan | Pass after fix (Lighthouse, version not recorded, 24 September 2026) | The initial Edge Lighthouse score was 96 due to insufficient dark-mode contrast on the Predict button. After the button palette fix, the score was 100 with no failed audits or remaining contrast finding. Gradio footer images are still reported as lacking suitable alt text, but the footer is retained because its Settings link is required to switch themes. | Retain the footer image finding as a disclosed low-priority Gradio framework limitation; repeat the scan after major UI or dependency changes |
| 200% zoom | Pass (manual, 24 September 2026) | In Edge at 200% zoom, all controls remained visible, text was readable, Predict and Clear remained usable, and no content overlapped, was clipped or required horizontal scrolling | Repeat after major layout changes |
| Narrow viewport | Pass (manual, 24 September 2026) | In Edge at a 320 CSS-pixel responsive viewport, all controls remained visible, text wrapped correctly, the result table and buttons remained usable, and no horizontal scrolling, overlap or clipping was observed | Repeat after major layout changes |

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
