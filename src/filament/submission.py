"""Writing and validating the Kaggle submission CSV."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rle import decode


def to_dataframe(preds: dict[str, list[str]]) -> pd.DataFrame:
    """{image stem: [counts, ...]} -> DataFrame(filament_id, segmentation_rle).

    Ids are "<stem>_<k>" with k starting at 1. Images with no filament get no rows.
    """
    rows = [{"filament_id": f"{stem}_{k}", "segmentation_rle": c}
            for stem, counts in preds.items() for k, c in enumerate(counts, 1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def validate(df: pd.DataFrame, check_overlap: bool = True) -> list[str]:
    """Return a list of problems (empty = OK). Kaggle rejects overlapping masks."""
    problems = []
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        problems.append(f"columns are {list(df.columns)}")
    if df["filament_id"].duplicated().any():
        problems.append("duplicate filament_id values")
    if df["segmentation_rle"].str.contains("[\"']", regex=True).any():
        problems.append("RLE strings contain quotes")
    if check_overlap:
        stems = df["filament_id"].str.split("_", n=1).str[0]
        for stem, grp in df.groupby(stems):
            acc = None
            for c in grp["segmentation_rle"]:
                m = decode(c).astype(np.uint8)
                acc = m if acc is None else acc + m
            if acc is not None and acc.max() > 1:
                problems.append(f"overlapping masks in {stem}")
    return problems


def write(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)  # pandas does not quote these strings (no commas inside)
