"""Download and prepare the public MUMDMC2025 cropped XPL subset.

Two supported workflows:
1. Download directly from the public Figshare file with --download.
2. Use a dataset ZIP/folder you already downloaded with --local-source PATH.

The script never stores dataset images in git. It extracts/scans them under data/raw/
and writes a reproducible manifest under data/manifests/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_URL = "https://figshare.com/ndownloader/files/55998200"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_NAMES = ["Biotite", "Hornblende", "Plagioclase", "Potassium-Feldspar", "Quartz"]
ALIASES = {
    "label": ["label", "class", "mineral", "mineral_class", "mineral_type", "target"],
    "group_id": ["group_id", "specimen_id", "sample_id", "crystal_id", "grain_id", "specimen", "sample"],
    "path": ["path", "filepath", "file", "filename", "image", "image_path"],
}


def download_file(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "pts-jepa/1.0"})
    with urlopen(req) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    return sha256(dest)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def pick_column(fieldnames: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize(name): name for name in fieldnames}
    for alias in aliases:
        if normalize(alias) in normalized:
            return normalized[normalize(alias)]
    return None


def find_metadata(root: Path) -> Path | None:
    candidates = list(root.rglob("*.csv"))
    preferred = [p for p in candidates if "metadata" in p.stem.lower()]
    return preferred[0] if preferred else (candidates[0] if candidates else None)


def class_from_path(path: Path) -> str | None:
    parts = {p.lower().replace("_", "-") for p in path.parts}
    for cls in CLASS_NAMES:
        if cls.lower() in parts or cls.lower().replace("-", "_") in parts:
            return cls
    stem = path.stem.lower().replace("_", "-")
    for cls in CLASS_NAMES:
        if cls.lower() in stem:
            return cls
    return None


def group_from_filename(path: Path) -> str:
    # Prefer obvious specimen/crystal identifiers when the filename contains them.
    for pattern in [r"(?:specimen|sample|crystal|grain)[-_]?([A-Za-z0-9]+)"]:
        match = re.search(pattern, path.stem, flags=re.I)
        if match:
            return match.group(0).lower()
    # Conservative fallback: every image becomes its own group. This avoids leakage,
    # at the cost of not exploiting rotational grouping until metadata is supplied.
    return path.stem


def build_manifest(root: Path, output: Path) -> dict:
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No image files found under {root}")

    metadata_path = find_metadata(root)
    metadata: dict[str, dict[str, str]] = {}
    metadata_keys = None
    if metadata_path:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                path_col = pick_column(reader.fieldnames, ALIASES["path"])
                label_col = pick_column(reader.fieldnames, ALIASES["label"])
                group_col = pick_column(reader.fieldnames, ALIASES["group_id"])
                if path_col:
                    for row in reader:
                        key = Path(row[path_col]).name
                        metadata[key] = {
                            "label": row.get(label_col, "") if label_col else "",
                            "group_id": row.get(group_col, "") if group_col else "",
                        }
                    metadata_keys = {"path": path_col, "label": label_col, "group_id": group_col}

    rows = []
    missing_labels = 0
    for image in images:
        record = metadata.get(image.name, {})
        label = record.get("label") or class_from_path(image)
        group_id = record.get("group_id") or group_from_filename(image)
        if not label:
            missing_labels += 1
        rows.append({
            "path": image.relative_to(root).as_posix(),
            "label": label or "",
            "group_id": group_id,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": len(rows),
        "metadata_file": str(metadata_path) if metadata_path else None,
        "metadata_columns": metadata_keys,
        "missing_labels": missing_labels,
        "labels": sorted({r["label"] for r in rows if r["label"]}),
        "root": str(root),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the public 2,500-image XPL subset from Figshare")
    parser.add_argument("--download-url", default=DEFAULT_URL)
    parser.add_argument("--local-source", type=Path, help="Existing ZIP archive or extracted dataset directory")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mumdmc2025"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/mumdmc2025.csv"))
    args = parser.parse_args()

    if not args.download and not args.local_source:
        parser.error("choose --download or --local-source PATH")
    if args.download and args.local_source:
        parser.error("choose only one data source")

    if args.download:
        archive = Path("data/raw/mumdmc2025_download.zip")
        print(f"Downloading MUMDMC2025 cropped subset from {args.download_url}")
        digest = download_file(args.download_url, archive)
        print(f"Downloaded SHA256: {digest}")
        if args.data_root.exists():
            shutil.rmtree(args.data_root)
        extract_archive(archive, args.data_root)
        dataset_root = args.data_root
    else:
        source = args.local_source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file() and source.suffix.lower() == ".zip":
            if args.data_root.exists():
                shutil.rmtree(args.data_root)
            extract_archive(source, args.data_root)
            dataset_root = args.data_root
        elif source.is_dir():
            dataset_root = source
        else:
            raise ValueError("--local-source must be a ZIP archive or directory")

    summary = build_manifest(dataset_root, args.manifest)
    print(json.dumps(summary, indent=2))
    if summary["missing_labels"]:
        print("WARNING: some images lack mineral labels. Supply metadata or organize images by class.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
