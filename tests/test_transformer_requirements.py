from __future__ import annotations

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
