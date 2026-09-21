"""Acquire declared source files without exposing them to version control."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from .data_pipeline import SchemaError


@dataclass(frozen=True)
class SourceFile:
    category: str
    kind: str
    url: str
    destination: Path


@dataclass(frozen=True)
class AcquisitionConfig:
    config_path: Path
    raw_dir: Path
    sources: tuple[SourceFile, ...]

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        raw_dir: str | Path,
    ) -> "AcquisitionConfig":
        resolved_config_path = Path(config_path)
        settings = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        if not isinstance(settings, Mapping):
            raise SchemaError("Experiment configuration must be a YAML mapping")

        resolved_raw_dir = Path(raw_dir)
        sources: list[SourceFile] = []
        for category in settings.get("data", {}).get("categories", []):
            for kind in ("reviews", "metadata"):
                url = str(category.get(f"{kind}_url", ""))
                filename = Path(urlparse(url).path).name
                if not url or not filename:
                    raise SchemaError(
                        f"Category {category.get('name', '<unknown>')} has no {kind}_url"
                    )
                sources.append(
                    SourceFile(
                        category=str(category["name"]),
                        kind=kind,
                        url=url,
                        destination=resolved_raw_dir / filename,
                    )
                )
        if not sources:
            raise SchemaError("Experiment configuration declares no downloadable sources")
        return cls(
            config_path=resolved_config_path,
            raw_dir=resolved_raw_dir,
            sources=tuple(sources),
        )


@dataclass(frozen=True)
class AcquisitionBundle:
    files: tuple[Path, ...]
    manifest_path: Path


def acquire(config: AcquisitionConfig) -> AcquisitionBundle:
    """Download every configured source atomically and record its checksum."""

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    files: list[Path] = []
    for source in config.sources:
        _download(source.url, source.destination)
        files.append(source.destination)
        records.append(
            {
                "category": source.category,
                "kind": source.kind,
                "source_url": source.url,
                "filename": source.destination.name,
                "bytes": source.destination.stat().st_size,
                "sha256": _sha256(source.destination),
            }
        )

    manifest_path = config.raw_dir / "acquisition_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AcquisitionBundle(files=tuple(files), manifest_path=manifest_path)


def _download(url: str, destination: Path) -> None:
    if destination.is_file():
        return

    partial = destination.with_name(destination.name + ".part")
    resume_at = partial.stat().st_size if partial.exists() else 0
    request = Request(url)
    if resume_at:
        request.add_header("Range", f"bytes={resume_at}-")

    with urlopen(request) as response:
        response_status = getattr(response, "status", None)
        can_resume = resume_at > 0 and response_status == 206
        mode = "ab" if can_resume else "wb"
        with partial.open(mode) as destination_file:
            shutil.copyfileobj(response, destination_file, length=1024 * 1024)
    os.replace(partial, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
