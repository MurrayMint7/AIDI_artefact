from __future__ import annotations

import ast
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


def test_colab_notebook_isolates_the_reproducible_environment() -> None:
    """Project pins must not replace packages in Colab's notebook process."""

    colab_requirements = Path("requirements-colab.txt").read_text(encoding="utf-8")
    notebook = json.loads(
        Path("notebooks/train_distilbert_colab.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "-r requirements-transformer.txt" in colab_requirements
    assert "requirements-colab.txt" in code
    assert "'venv', '--system-site-packages'" in code
    assert "VENV_PYTHON" in code
    assert "[sys.executable, '-m', 'pip'" not in code
    assert "'-m', 'pip', 'check'" not in code
    assert "if REPO_ROOT.exists():" in code
    assert "'pull', '--ff-only'" in code


def test_colab_notebook_code_cells_are_valid_python() -> None:
    notebook = json.loads(
        Path("notebooks/train_distilbert_colab.ipynb").read_text(encoding="utf-8")
    )

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            tree = ast.parse(source, filename=f"notebook-cell-{index}")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "preflight_code"
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    compile(node.value.value, "notebook-preflight", "exec")
