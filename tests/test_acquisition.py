from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from amazon_sentiment.acquisition import AcquisitionConfig, acquire


def test_acquire_places_complete_files_and_writes_a_checksum_manifest(
    tmp_path: Path,
) -> None:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    reviews_source = remote_dir / "Example.jsonl.gz"
    metadata_source = remote_dir / "meta_Example.jsonl.gz"
    reviews_source.write_bytes(b"synthetic compressed review bytes")
    metadata_source.write_bytes(b"synthetic compressed metadata bytes")

    settings = {
        "data": {
            "categories": [
                {
                    "name": "Example",
                    "reviews_url": reviews_source.as_uri(),
                    "metadata_url": metadata_source.as_uri(),
                }
            ]
        }
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    config = AcquisitionConfig.from_yaml(config_path, raw_dir=tmp_path / "raw")

    bundle = acquire(config)

    assert [path.name for path in bundle.files] == [
        "Example.jsonl.gz",
        "meta_Example.jsonl.gz",
    ]
    assert [path.read_bytes() for path in bundle.files] == [
        b"synthetic compressed review bytes",
        b"synthetic compressed metadata bytes",
    ]
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["files"] == [
        {
            "bytes": 33,
            "category": "Example",
            "filename": "Example.jsonl.gz",
            "kind": "reviews",
            "sha256": hashlib.sha256(b"synthetic compressed review bytes").hexdigest(),
            "source_url": reviews_source.as_uri(),
        },
        {
            "bytes": 35,
            "category": "Example",
            "filename": "meta_Example.jsonl.gz",
            "kind": "metadata",
            "sha256": hashlib.sha256(b"synthetic compressed metadata bytes").hexdigest(),
            "source_url": metadata_source.as_uri(),
        },
    ]
