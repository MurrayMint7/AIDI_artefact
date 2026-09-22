from __future__ import annotations

import json
from pathlib import Path


def test_datasets_fsspec_pin_is_compatible() -> None:
    """Datasets 5.0.1 declares fsspec[http] <= 2026.6.0."""

    requirements = Path("requirements-transformer.txt").read_text(encoding="utf-8")
    pins = {
        name: version
        for line in requirements.splitlines()
        if line and not line.startswith(("#", "-")) and "==" in line
        for name, version in [line.split("==", maxsplit=1)]
    }
    assert pins["datasets"] == "5.0.1"
    assert pins["fsspec"] == "2026.6.0"


def test_colab_overlay_does_not_replace_runtime_owned_packages() -> None:
    """The Colab overlay must not install the local environment lock."""

    colab_requirements = Path("requirements-colab.txt").read_text(encoding="utf-8")
    notebook = json.loads(
        Path("notebooks/train_distilbert_colab.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "-r requirements.txt" not in colab_requirements
    for runtime_package in ("pandas", "rich", "fsspec", "torch"):
        assert not any(
            line.lower().startswith(runtime_package)
            for line in colab_requirements.splitlines()
        )
    assert "requirements-colab.txt" in code
    assert "requirements-transformer.txt" not in code
    assert "capture_output=True" in code
    assert "critical_issues" in code
    assert "'pip', 'check'], check=True" not in code
