"""Audit a prepared MUMDMC manifest before any model training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from data.manifest import resolve_image_path, validate_manifest
from data.splits import add_research_columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("data/manifests/mumdmc2025.csv"), nargs="?")
    args = parser.parse_args()

    summary = validate_manifest(args.manifest)
    print("rows:", summary["rows"])
    print("labels:", summary["labels"])
    print("groups:", summary["groups"])
    print("missing paths:", len(summary["missing_paths"]))
    if summary["missing_paths"]:
        raise SystemExit("Audit failed: image paths are missing")

    df = add_research_columns(pd.read_csv(args.manifest))
    print("\nclass counts:\n", df["label"].value_counts().sort_index())
    print("\nresearch groups:", df["research_group"].nunique())
    print("angles:", sorted(x for x in df["angle"].unique() if x))

    bad = []
    for path in df["path"].head(50):
        image_path = resolve_image_path(args.manifest, path)
        with Image.open(image_path) as image:
            image.verify()
    print("checked first 50 images: OK")

    if len(df) != 2500:
        raise SystemExit(f"Expected 2,500 cropped images, found {len(df)}")
    if df["label"].isna().any() or (df["label"].astype(str).str.strip() == "").any():
        raise SystemExit("Audit failed: missing labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
