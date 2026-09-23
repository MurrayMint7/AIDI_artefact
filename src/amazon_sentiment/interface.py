"""Accessible presentation adapter for the local sentiment interface."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Protocol

from .inference import InferenceInputError, Prediction


class Predictor(Protocol):
    def predict(self, text: str) -> Prediction: ...


@dataclass(frozen=True)
class InterfaceResult:
    """Text-first content rendered in the Gradio result regions."""

    validation_html: str
    result_html: str

LOGGER = logging.getLogger(__name__)


_EMPTY_RESULT = """
<section id="prediction-result" class="result-card" aria-live="polite">
  <h2>Prediction result</h2>
  <p>Submit a review to see the predicted sentiment and routing decision.</p>
</section>
""".strip()


def present_prediction(text: str, predictor: Predictor) -> InterfaceResult:
    """Validate one submission and convert its typed prediction to semantic HTML."""

    try:
        prediction = predictor.predict(text)
    except InferenceInputError as error:
        message = html.escape(str(error))
        return InterfaceResult(
            validation_html=(
                '<p class="validation-error" role="alert">'
                f"<strong>Input error:</strong> {message}</p>"
            ),
            result_html=_EMPTY_RESULT,
        )
    except Exception:
        LOGGER.exception("Local sentiment prediction failed")
        return InterfaceResult(
            validation_html="",
            result_html=(
                '<section class="result-card error-card" role="alert">'
                "<h2>Prediction unavailable</h2>"
                "<p>The local model could not complete this request. "
                "Check the terminal for details and try again.</p></section>"
            ),
        )

    label = html.escape(prediction.label.capitalize())
    if prediction.review_required:
        route_icon = "&#9888;"
        route_text = "Human review required"
        if prediction.review_reason == "mandatory_review_for_predicted_label":
            explanation = (
                "Neutral predictions always require review because final-test evidence "
                "showed weaker reliability for this class."
            )
        else:
            explanation = (
                "Confidence is below the frozen automatic-routing threshold of "
                f"{prediction.threshold:.1%}."
            )
        route_class = "review-route"
    else:
        route_icon = "&#10003;"
        route_text = "Automatic route"
        explanation = (
            "The predicted class is eligible for automatic routing and confidence "
            "meets the frozen threshold."
        )
        route_class = "automatic-route"

    probability_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(name.capitalize())}</th>"
        f"<td>{probability:.1%}</td>"
        "</tr>"
        for name, probability in prediction.probabilities.items()
    )
    result_html = f"""
<section id="prediction-result" class="result-card" aria-live="polite" tabindex="-1">
  <h2>Prediction result</h2>
  <p class="route-status {route_class}">
    <span aria-hidden="true">{route_icon}</span> <strong>{route_text}</strong>
  </p>
  <p>{html.escape(explanation)}</p>
  <dl>
    <dt>Predicted sentiment</dt><dd>{label}</dd>
    <dt>Calibrated confidence</dt><dd>{prediction.confidence:.1%}</dd>
    <dt>Model version</dt><dd><code>{html.escape(prediction.model_version)}</code></dd>
  </dl>
  <table>
    <caption>Calibrated class probabilities</caption>
    <thead><tr><th scope="col">Sentiment</th><th scope="col">Probability</th></tr></thead>
    <tbody>{probability_rows}</tbody>
  </table>
  <p class="scope-note">This is a model prediction for review triage, not a fact about the reviewer or product.</p>
</section>
""".strip()
    return InterfaceResult(validation_html="", result_html=result_html)


def build_app(predictor: Predictor):
    """Build the local Gradio Blocks application without launching a server."""

    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The interface requires requirements-interface.txt"
        ) from error

    css = """
    :root {
      --body-text-color: #17202a;
      --body-background-fill: #ffffff;
      --primary-600: #174ea6;
    }
    .gradio-container { max-width: 58rem !important; }
    button:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
      outline: 3px solid #111111 !important;
      outline-offset: 3px !important;
    }
    .validation-error, .error-card {
      border-left: 0.4rem solid #b3261e;
      padding: 0.75rem 1rem;
      background: #fff4f2;
      color: #5f1410;
    }
    .result-card {
      border: 1px solid #536471;
      border-radius: 0.5rem;
      padding: 1rem 1.25rem;
      background: #ffffff;
      color: #17202a;
    }
    .route-status { font-size: 1.2rem; padding: 0.65rem; }
    .review-route { border: 2px solid #8a3b00; background: #fff4e5; color: #542500; }
    .automatic-route { border: 2px solid #176b3a; background: #eefaf2; color: #0b4725; }
    dl { display: grid; grid-template-columns: minmax(10rem, 1fr) 2fr; gap: 0.4rem 1rem; }
    dt { font-weight: 700; }
    dd { margin: 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    caption { text-align: left; font-weight: 700; margin-bottom: 0.4rem; }
    th, td { border: 1px solid #697680; padding: 0.55rem; text-align: left; }
    .scope-note { margin-top: 1rem; font-size: 0.95rem; }
    @media (max-width: 35rem) {
      dl { grid-template-columns: 1fr; }
    }
    """

    def submit(text: str) -> tuple[str, str]:
        rendered = present_prediction(text, predictor)
        return rendered.validation_html, rendered.result_html

    with gr.Blocks(
        title="Amazon review sentiment triage",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Amazon review sentiment triage\n"
            "Enter one English-language Amazon review. The local model predicts "
            "negative, neutral or positive sentiment and shows whether an analyst "
            "must review the result."
        )
        review_input = gr.Textbox(
            label="Review text",
            info="Required. Review text is processed locally and is not written to the operational log.",
            lines=7,
            placeholder="Enter the review body",
            elem_id="review-text",
        )
        validation = gr.HTML(value="", elem_id="input-validation")
        with gr.Row():
            submit_button = gr.Button("Predict sentiment", variant="primary")
            clear_button = gr.Button("Clear")
        result = gr.HTML(value=_EMPTY_RESULT, elem_id="result-region")

        submit_button.click(
            fn=submit,
            inputs=review_input,
            outputs=[validation, result],
            api_name="predict",
        )
        clear_button.click(
            fn=lambda: ("", "", _EMPTY_RESULT),
            inputs=None,
            outputs=[review_input, validation, result],
            api_name=False,
        )
    demo.stage5_css = css
    return demo
