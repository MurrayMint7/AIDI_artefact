# Public data-manifest evidence

These files are aggregate exports from the private preparation outputs used for the reported experiment:

- `data_manifest.json` records source and output filenames, sizes, row counts and SHA-256 checksums.
- `data_quality.json` records the curated schema, null counts and aggregate cleaning counts.
- `join_report.json` records aggregate metadata-join coverage and the diagnostic wrong-key comparison.
- `split_summary.json` records the chronological boundaries, sampling counts, seed and aggregate training class/category counts.

The exports contain no review bodies, titles, product identifiers, user identifiers, row-level record identifiers or row-level predictions. The private `data/generated/manifests/split_manifest.json` is not published because its `record_ids_by_split` field contains row-level identifiers. `split_summary.json` is an allowlisted aggregate projection of that file.

Raw downloads, the curated Parquet dataset and private row-level manifests remain excluded from Git under the repository's data-governance rules.
