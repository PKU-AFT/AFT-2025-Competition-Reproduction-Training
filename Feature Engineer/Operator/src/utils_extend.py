"""
Extended utilities for large-scale factor correlation filtering.

Designed to work on the merged features parquet without loading everything
into memory by sampling rows before computing correlations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from src.utils import DEFAULT_EXCLUDE_COLS


def _infer_factor_cols_lazy(columns: Iterable[str], exclude: Sequence[str]) -> list[str]:
    exclude_set = set(exclude)
    return [c for c in columns if c not in exclude_set]


def _drop_constant_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if not constant:
        return df, []
    return df.drop(columns=constant), constant


def _filter_high_corr(df: pd.DataFrame, corr_threshold: float) -> list[str]:
    """
    Greedy correlation filter: keep higher-variance columns first, drop later
    columns that are highly correlated (|corr| >= corr_threshold).
    """
    if df.empty or df.shape[1] <= 1:
        return df.columns.tolist()

    std_map = df.std(ddof=0).fillna(0.0).to_dict()
    ordered = sorted(df.columns, key=lambda c: std_map.get(c, 0.0), reverse=True)

    corr = df.corr(method="pearson").abs()
    keep: list[str] = []
    dropped: set[str] = set()

    for col in ordered:
        if col in dropped:
            continue
        keep.append(col)
        high_corr = corr.loc[col] >= corr_threshold
        to_drop = [c for c, flag in high_corr.items() if flag and c != col]
        dropped.update(to_drop)

    return keep


def sample_correlated_factors(
    parquet_path: str | Path,
    sample_frac: float = 0.05,
    max_rows: int = 200_000,
    corr_threshold: float = 0.8,
    seed: int = 42,
) -> list[str]:
    """
    Sample rows from a large parquet and perform correlation filtering.

    Parameters
    ----------
    parquet_path : str | Path
        Path to merged features parquet.
    sample_frac : float
        Fraction of rows to sample. Use <= 1.0.
    max_rows : int
        Upper bound of sampled rows to cap memory usage.
    corr_threshold : float
        Absolute correlation threshold for dropping redundant factors.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    list[str]
        Factor names kept after correlation filtering.
    """
    parquet_path = Path(parquet_path)
    lf = pl.scan_parquet(parquet_path)

    factor_cols = _infer_factor_cols_lazy(lf.columns, DEFAULT_EXCLUDE_COLS)
    if not factor_cols:
        raise ValueError("No factor columns found after excluding DEFAULT_EXCLUDE_COLS")

    total_rows = lf.select(pl.count()).collect().item()
    if total_rows == 0:
        raise ValueError("Parquet file is empty")

    sample_n = min(max_rows, max(1, int(total_rows * sample_frac)))

    sample_df = (
        lf.select(factor_cols)
        .sample(n=sample_n, with_replacement=False, shuffle=True, seed=seed)
        .collect(streaming=True)
        .to_pandas()
    )

    sample_df, constant_cols = _drop_constant_cols(sample_df)
    if constant_cols:
        print(f"[sample_correlated_factors] Dropped constant cols: {len(constant_cols)}")

    kept = _filter_high_corr(sample_df, corr_threshold=corr_threshold)
    print(
        f"[sample_correlated_factors] sampled_rows={len(sample_df)}, "
        f"factors_before={len(factor_cols)}, factors_after={len(kept)}"
    )
    return kept


if __name__ == "__main__":
    # Quick manual run example (adjust path as needed).
    selected = sample_correlated_factors(
        parquet_path=Path("E:/_Quant/内培/AFT/Feature_Engineer_Kaggle_2025/data/output/merged_features.parquet"),
        sample_frac=0.05,
        max_rows=200_000,
        corr_threshold=0.8,
        seed=42,
    )
    print(f"Kept factors: {len(selected)}")
