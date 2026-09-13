"""Leakage-safe MUMDMC group splitting utilities."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_mumdmc_filename(path: str) -> dict[str, str]:
    """Parse the documented MUMDMC filename convention when possible."""
    stem = Path(path).stem
    parts = stem.split("-")
    angle = parts[-1] if parts and re.fullmatch(r"\d+(?:deg)?", parts[-1], re.I) else ""
    return {
        "slide_id": parts[0] if len(parts) > 0 else "",
        "mineral_token": parts[1] if len(parts) > 1 else "",
        "polarization_token": parts[2] if len(parts) > 2 else "",
        "crystal_id": parts[3] if len(parts) > 3 else "",
        "photo_id": parts[4] if len(parts) > 4 else "",
        "angle": angle,
    }


def add_research_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out["path"].map(parse_mumdmc_filename).apply(pd.Series)
    for col in parsed.columns:
        out[col] = parsed[col]
    out["research_group"] = (
        out["slide_id"].fillna("").astype(str)
        + "|"
        + out["polarization_token"].fillna("").astype(str)
        + "|"
        + out["crystal_id"].fillna("").astype(str)
        + "|"
        + out["photo_id"].fillna("").astype(str)
    )
    return out


def group_split(frame: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """Return train/val/test frames with zero research-group overlap."""
    if "research_group" not in frame.columns:
        frame = add_research_columns(frame)
    if frame["research_group"].eq("|||").any():
        raise ValueError("Could not derive useful research groups from filenames")

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(frame, groups=frame["research_group"]))
    trainval = frame.iloc[trainval_idx].copy()
    test = frame.iloc[test_idx].copy()

    relative_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["research_group"]))
    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    assert set(train.research_group).isdisjoint(val.research_group)
    assert set(train.research_group).isdisjoint(test.research_group)
    assert set(val.research_group).isdisjoint(test.research_group)
    return train, val, test
