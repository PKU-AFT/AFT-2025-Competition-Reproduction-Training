"""Factory helpers to wrap DataFrame-based factor builders into Series factors.

Context
-------
In many codebases, factor construction functions mutate/return a DataFrame with
one or more new columns (e.g., Factor.auction_price_pressure adds
`auction_price_pressure`, `auction_price_pressure_pct`, ...). The leak detector
expects factors of the form `f(df) -> pd.Series` that return exactly one column
aligned to the input rows.

This module provides tiny helpers to bridge the two styles so you can plug your
existing DataFrame builders into the leak detector with minimum boilerplate.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import pandas as pd

from factors import FactorDefinition


DFBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def derive_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight derivations for frequently used intermediate columns.

    - `bid_ask_mid` = (bid_price + ask_price) / 2 when both exist
    - `bid_ask_spread` = ask_price - bid_price when both exist
    The function is idempotent and only fills missing columns.
    """
    if "bid_ask_mid" not in df.columns and {"bid_price", "ask_price"}.issubset(df.columns):
        df = df.copy()
        df["bid_ask_mid"] = (df["bid_price"] + df["ask_price"]) / 2
    if "bid_ask_spread" not in df.columns and {"bid_price", "ask_price"}.issubset(df.columns):
        if not isinstance(df, pd.DataFrame):  # defensive; should always be DataFrame
            return df
        df = df.copy()
        df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    return df


def make_series_factor(
    builders: DFBuilder | Sequence[DFBuilder],
    output_col: str,
    *,
    requires: Iterable[str] | None = None,
    auto_derive_common: bool = True,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Wrap one or more DataFrame builders into a `f(df)->Series` factor.

    Parameters
    ----------
    builders : DFBuilder | Sequence[DFBuilder]
        A single function or sequence of functions, each taking and returning a
        DataFrame. They are applied in order onto a copy of the input.
    output_col : str
        Column name to extract as the factor Series after all builders run.
    requires : Iterable[str] | None
        Optional set of required base columns. If any is missing, the wrapper
        raises a KeyError so the detector records a `factor_error` with details.
    auto_derive_common : bool
        When True, derive common intermediate columns like `bid_ask_mid` and
        `bid_ask_spread` before running the builders.
    """

    seq: List[DFBuilder] = list(builders) if isinstance(builders, (list, tuple)) else [builders]
    req_set = set(requires or ())

    def _fn(df: pd.DataFrame) -> pd.Series:
        missing = [c for c in req_set if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        tmp = df.copy()
        if auto_derive_common:
            tmp = derive_common_columns(tmp)
        for builder in seq:
            tmp = builder(tmp)
        if output_col not in tmp.columns:
            raise KeyError(f"Output column '{output_col}' not found after builders")
        s = tmp[output_col]
        if not isinstance(s, pd.Series) or len(s) != len(df):
            raise ValueError(
                f"Wrapped factor must return a Series aligned to input; got type={type(s)}, len={len(s)} vs {len(df)}"
            )
        return s

    return _fn


def make_factor_definition(
    name: str,
    builders: DFBuilder | Sequence[DFBuilder],
    output_col: str,
    *,
    description: str | None = None,
    requires: Iterable[str] | None = None,
    auto_derive_common: bool = True,
) -> FactorDefinition:
    """Build a FactorDefinition from DataFrame builders.

    Example
    -------
    >>> from Factor_test import Factor
    >>> fn = make_series_factor(lambda df: Factor.auction_price_pressure(df, 'stock_id'),
    ...                         output_col='auction_price_pressure',
    ...                         requires=['near_price','far_price','wap'])
    >>> fd = make_factor_definition('auction_price_pressure', fn, 'auction_price_pressure')
    """
    fn = builders if callable(builders) and not isinstance(builders, (list, tuple)) else make_series_factor(
        builders, output_col, requires=requires, auto_derive_common=auto_derive_common
    )
    desc = description or name
    return FactorDefinition(name=name, function=fn, description=desc)

