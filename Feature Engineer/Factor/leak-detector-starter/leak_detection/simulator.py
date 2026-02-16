"""Tools to (re)generate the simulated market dataset used for factor experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_factors.csv"


def generate_dataset(
    *,
    symbols: Iterable[str] = ("AAA", "BBB", "CCC"),
    start: str = "2021-01-04",
    periods: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a compact OHLCV dataset that students can feed into factor code."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="B")
    records: list[dict[str, object]] = []

    for symbol in symbols:
        base = 100 + rng.normal(0, 1, size=len(dates)).cumsum() + rng.normal(0, 0.5)
        close = pd.Series(base, index=dates)
        open_price = close * (1 + rng.normal(0, 0.002, size=len(dates)))
        high = np.maximum(close.values, open_price.values) * (1 + np.abs(rng.normal(0, 0.01, size=len(dates))))
        low = np.minimum(close.values, open_price.values) * (1 - np.abs(rng.normal(0, 0.01, size=len(dates))))
        volume = rng.integers(5_000, 20_000, size=len(dates))
        ret_1d = close.pct_change()

        for idx, date in enumerate(dates):
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": float(open_price.iloc[idx]),
                    "high": float(high[idx]),
                    "low": float(low[idx]),
                    "close": float(close.iloc[idx]),
                    "volume": int(volume[idx]),
                    "ret_1d": float(ret_1d.iloc[idx]) if pd.notna(ret_1d.iloc[idx]) else None,
                }
            )

    df = pd.DataFrame(records)
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_dataset(path: Path | None = None) -> Path:
    """Generate the dataset and persist it to ``path`` (defaults to ``data/sample_factors.csv``)."""

    target_path = Path(path) if path else DEFAULT_OUTPUT_PATH
    df = generate_dataset()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path


if __name__ == "__main__":
    location = write_dataset()
    print(f"Dataset written to {location}")
