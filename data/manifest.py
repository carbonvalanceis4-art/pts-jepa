"""Utilities for loading a prepared MUMDMC2025 manifest."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_image_path(manifest_path: str | Path, relative_path: str) -> Path:
    """Resolve a manifest path against the dataset root inferred from the manifest.

    The manifest is expected at data/manifests/mumdmc2025.csv and the images under
    data/raw/mumdmc2025/, but callers may place both elsewhere. If the first candidate
    does not exist, a sibling `mumdmc2025` directory is tried as a convenience.
    """
    manifest = Path(manifest_path).resolve()
    project_root = manifest.parent.parent.parent
    candidates = [
        project_root / "data" / "raw" / "mumdmc2025" / relative_path,
        manifest.parent / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    labels = sorted({row.get("label", "") for row in rows if row.get("label")})
    groups = {row.get("group_id", "") for row in rows if row.get("group_id")}
    missing_paths = []
    for row in rows:
        path = resolve_image_path(manifest_path, row["path"])
        if not path.exists():
            missing_paths.append(row["path"])
    return {
        "rows": len(rows),
        "labels": labels,
        "groups": len(groups),
        "missing_paths": missing_paths,
    }
