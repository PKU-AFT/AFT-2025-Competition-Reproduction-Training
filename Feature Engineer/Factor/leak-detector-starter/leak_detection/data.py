"""Data loading helpers for the leak detection exercise."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_factors.csv"


@dataclass
class DatasetDescriptor:
    """Light-weight metadata about the simulated factor dataset."""

    rows: int
    symbols: List[str]
    start_date: str
    end_date: str
    factor_columns: List[str]
    target_columns: List[str]


def load_sample_factors(path: Path | None = None) -> pd.DataFrame:
    """Load the simulated factor dataset shipped with the project."""

    target_path = Path(path) if path else DEFAULT_DATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {target_path}. Run scripts/regenerate_data.py to create it."
        )
    df = pd.read_csv(target_path, parse_dates=["date"])
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def describe_dataset(df: pd.DataFrame) -> DatasetDescriptor:
    """Summarise the key characteristics of the dataset for reporting."""

    factors = [c for c in df.columns if c.startswith("factor_")]
    targets = [c for c in df.columns if c.startswith("target_")]
    symbols = sorted(df["symbol"].unique().tolist())
    start_date = df["date"].min().strftime("%Y-%m-%d") if not df.empty else ""
    end_date = df["date"].max().strftime("%Y-%m-%d") if not df.empty else ""
    return DatasetDescriptor(
        rows=len(df),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        factor_columns=factors,
        target_columns=targets,
    )


def split_train_validation(
    df: pd.DataFrame,
    validation_ratio: float = 0.2,
    groupby_cols: Iterable[str] = ("symbol",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split to help students test their detector."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    grouped = df.sort_values([*groupby_cols, "date"])
    train_parts: List[pd.DataFrame] = []
    valid_parts: List[pd.DataFrame] = []
    for _, part in grouped.groupby(list(groupby_cols), sort=False):
        split_idx = int(len(part) * (1 - validation_ratio))
        split_idx = max(1, min(len(part) - 1, split_idx))
        train_parts.append(part.iloc[:split_idx])
        valid_parts.append(part.iloc[split_idx:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(valid_parts).reset_index(drop=True)
