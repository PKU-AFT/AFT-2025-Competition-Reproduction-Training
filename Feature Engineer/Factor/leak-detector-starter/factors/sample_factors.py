"""Sample factor definitions used for the onboarding exercise."""
from __future__ import annotations

import pandas as pd
import numpy as np


def factor_momentum_ema(df: pd.DataFrame) -> pd.Series:
    """Lagged exponential moving average of 1-day returns (uses only past data)."""

    returns = df.groupby("symbol", sort=False)["close"].pct_change()
    ema = returns.groupby(df["symbol"], sort=False).transform(lambda s: s.ewm(span=3, adjust=False).mean())
    return ema.groupby(df["symbol"], sort=False).shift(1)


def factor_volume_mean5(df: pd.DataFrame) -> pd.Series:
    """Five-day rolling mean of volume, shifted by one day to avoid look-ahead."""

    rolling = df.groupby("symbol", sort=False)["volume"].transform(
        lambda s: s.rolling(window=5, min_periods=3).mean()
    )
    return rolling.groupby(df["symbol"], sort=False).shift(1)


def factor_future_return(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: aligns tomorrow's return onto today's row."""

    forward_close = df.groupby("symbol", sort=False)["close"].shift(-1)
    return (forward_close - df["close"]) / df["close"]


# --- Additional safe factors (no look-ahead) ---


def factor_ret_1d_lag1(df: pd.DataFrame) -> pd.Series:
    """Yesterday's 1-day return (uses past only)."""

    return df.groupby("symbol", sort=False)["ret_1d"].shift(1)


def factor_sma_5(df: pd.DataFrame) -> pd.Series:
    """5-day simple moving average of close, lagged by 1 day."""

    sma = df.groupby("symbol", sort=False)["close"].transform(
        lambda s: s.rolling(window=5, min_periods=3).mean()
    )
    return sma.groupby(df["symbol"], sort=False).shift(1)


def factor_volatility_10(df: pd.DataFrame) -> pd.Series:
    """10-day rolling volatility (std of 1d returns), lagged by 1 day."""

    vol = df.groupby("symbol", sort=False)["ret_1d"].transform(
        lambda s: s.rolling(window=10, min_periods=5).std()
    )
    return vol.groupby(df["symbol"], sort=False).shift(1)


def factor_rsi_14(df: pd.DataFrame) -> pd.Series:
    """14-period RSI computed on close, lagged by 1 day."""

    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = df.groupby("symbol", sort=False)["close"].transform(_rsi)
    return rsi.groupby(df["symbol"], sort=False).shift(1)


def factor_obv_lag1(df: pd.DataFrame) -> pd.Series:
    """On-balance volume cumulative line, lagged by 1 day to avoid same-day info."""
    # Compute price change sign per symbol without introducing a MultiIndex
    delta = df.groupby("symbol", sort=False)["close"].diff()
    sign = np.sign(delta).fillna(0.0)
    obv = (sign * df["volume"]).groupby(df["symbol"], sort=False).cumsum().astype(float)
    return obv.groupby(df["symbol"], sort=False).shift(1)


# --- Additional intentional leaks (for testing the detector) ---


def factor_future_close(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: tomorrow's close aligned to today's row."""

    return df.groupby("symbol", sort=False)["close"].shift(-1)


def factor_future_volume(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: tomorrow's volume aligned to today."""

    return df.groupby("symbol", sort=False)["volume"].shift(-1)
