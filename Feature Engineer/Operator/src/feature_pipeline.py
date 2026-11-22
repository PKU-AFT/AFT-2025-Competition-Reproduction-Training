"""Deterministic preparation and assembly helpers for the operator notebook."""

from pathlib import Path
from collections.abc import Sequence

import polars as pl

KEYS = ("stock_id", "date_id", "seconds_in_bucket")
TRAIN_START = 0
TRAIN_END = 400


def sort_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Put observations in chronological order for each stock."""
    return df.sort(list(KEYS), maintain_order=True)


def training_subset(df: pl.DataFrame) -> pl.DataFrame:
    """Select the period used by both IC and correlation screening."""
    return df.filter(pl.col("date_id").is_between(TRAIN_START, TRAIN_END, closed="both"))


def _validate_keys(lf: pl.LazyFrame, label: str) -> int:
    columns = lf.collect_schema().names()
    missing = [key for key in KEYS if key not in columns]
    if missing:
        raise ValueError(f"{label}: missing keys {missing}")
    stats = lf.select(
        pl.len().alias("rows"),
        pl.struct(list(KEYS)).n_unique().alias("unique"),
        pl.any_horizontal([pl.col(key).is_null() for key in KEYS]).sum().alias("nulls"),
    ).collect().row(0, named=True)
    if stats["nulls"]:
        raise ValueError(f"{label}: null keys")
    if stats["rows"] != stats["unique"]:
        raise ValueError(f"{label}: duplicate keys")
    return stats["rows"]


def merge_operator_files(base: pl.DataFrame, paths: Sequence[Path]) -> pl.LazyFrame:
    """Join only this run's explicit outputs, after validating their row identities."""
    if not paths:
        raise ValueError("No operator files were generated in this run")
    base_lf = base.lazy()
    base_rows = _validate_keys(base_lf, "base")
    merged = base_lf
    seen = set(base.columns)
    for path in paths:
        path = Path(path)
        lf = pl.scan_parquet(path)
        rows = _validate_keys(lf, path.name)
        if rows != base_rows:
            raise ValueError(f"{path.name}: keys do not match the base panel")
        missing = base_lf.select(KEYS).join(lf.select(KEYS), on=list(KEYS), how="anti").select(pl.len()).collect().item()
        if missing:
            raise ValueError(f"{path.name}: keys do not match the base panel")
        new_cols = [col for col in lf.collect_schema().names() if col not in seen]
        if new_cols:
            merged = merged.join(lf.select([*KEYS, *new_cols]), on=list(KEYS), how="left", maintain_order="left")
            seen.update(new_cols)
    return merged
