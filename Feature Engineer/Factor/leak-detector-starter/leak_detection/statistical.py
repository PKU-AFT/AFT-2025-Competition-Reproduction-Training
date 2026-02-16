"""Statistical leak suspicion detector for precomputed factor columns.

This module complements the forward-replay detector by providing a black-box
statistical check that requires only precomputed factor values and a target
column. It flags factors whose current values exhibit unusually high
lead-correlation with future targets, which is indicative of potential
look-ahead leakage or direct alignment with future labels.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .detector import LeakDetectionResult, LeakViolation


def _pick_factor_columns(
    df: pd.DataFrame,
    *,
    factor_cols: Iterable[str] | None = None,
    include_prefixes: Tuple[str, ...] | None = None,
    exclude_cols: Iterable[str] = (),
) -> List[str]:
    """Select candidate factor columns from a DataFrame.

    - If ``factor_cols`` is provided, it is used as-is (after filtering to numeric and not-excluded).
    - Else if ``include_prefixes`` is provided, numeric columns starting with any of the prefixes are used.
    - Else, all numeric columns excluding ``exclude_cols`` are used.
    """

    exclude_set = set(exclude_cols)

    def _is_numeric(col: str) -> bool:
        try:
            return pd.api.types.is_numeric_dtype(df[col])
        except Exception:
            return False

    if factor_cols is not None:
        items = [c for c in factor_cols if c in df.columns]
        return [c for c in items if c not in exclude_set and _is_numeric(c)]

    if include_prefixes:
        items = [
            c
            for c in df.columns
            if c not in exclude_set
            and any(c.startswith(p) for p in include_prefixes)
            and _is_numeric(c)
        ]
        return items

    # Fallback: any numeric column not excluded
    return [c for c in df.columns if c not in exclude_set and _is_numeric(c)]


def detect_statistical_suspicion(
    df: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    target_col: str = "target",
    factor_cols: Iterable[str] | None = None,
    include_prefixes: Tuple[str, ...] | None = None,
    exclude_cols: Iterable[str] = (),
    max_lead: int = 3,
    min_obs: int = 20,
    method: str = "pearson",
    threshold: float = 0.7,
) -> LeakDetectionResult:
    """Run a black-box statistical suspicion detector on precomputed factors.

    The detector computes correlation between each factor at time t and the
    future target at t+k (k=1..max_lead) within each entity's time series.
    A factor is flagged when the maximum absolute lead correlation exceeds
    ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing ``entity_col``, ``time_col``, ``target_col`` and factor columns.
    entity_col : str
        Entity grouping key (e.g., stock identifier).
    time_col : str
        Time ordering key. The DataFrame will be sorted by (entity_col, time_col).
    target_col : str, default "target"
        Target column used to assess lead correlations.
    factor_cols : Iterable[str] | None
        Explicit set of factor column names. If None, ``include_prefixes``/``exclude_cols`` are used.
    include_prefixes : tuple[str, ...] | None
        If provided, select numeric columns whose names start with any of the prefixes (after excluding ``exclude_cols``).
    exclude_cols : Iterable[str]
        Columns to exclude from consideration (e.g., keys and target columns).
    max_lead : int, default 3
        Maximum lead horizon k to compute correlations against target_{t+k}.
    min_obs : int, default 20
        Minimum number of overlapping observations required to compute a correlation.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation method.
    threshold : float, default 0.7
        Threshold on the maximum absolute lead correlation to flag a factor.
    """

    required = {entity_col, time_col, target_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    work = df.copy()
    work = work.sort_values([entity_col, time_col]).reset_index(drop=True)

    # Pick candidate factor columns
    excludes = set(exclude_cols) | {entity_col, time_col, target_col}
    candidates = _pick_factor_columns(
        work, factor_cols=factor_cols, include_prefixes=include_prefixes, exclude_cols=excludes
    )

    inspected: List[str] = []
    violations: List[LeakViolation] = []

    if len(candidates) == 0:
        return LeakDetectionResult(inspected_columns=inspected, violations=violations, notes="No candidate factors")

    # Precompute lead targets per k to avoid redoing groupby work for every factor
    for k in range(1, max_lead + 1):
        work[f"__lead_{k}"] = work.groupby(entity_col, sort=False)[target_col].shift(-k)

    # Compute correlations and collect violations
    for col in candidates:
        inspected.append(col)
        corrs: Dict[str, float] = {}
        best_k = None
        best_abs = -np.inf

        for k in range(1, max_lead + 1):
            x = work[col]
            y = work[f"__lead_{k}"]
            # Align valid pairs and require a minimum number of observations
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < min_obs:
                corrs[f"lead_{k}"] = np.nan
                continue
            try:
                r = float(x[mask].corr(y[mask], method=method))
            except Exception:
                r = np.nan
            corrs[f"lead_{k}"] = r
            if not np.isnan(r) and abs(r) > best_abs:
                best_abs = abs(r)
                best_k = k

        suspicion = float(best_abs if np.isfinite(best_abs) else 0.0)
        if suspicion >= threshold and best_k is not None:
            violations.append(
                LeakViolation(
                    column=col,
                    signal_type="statistical_suspicion",
                    confidence=suspicion,
                    details={
                        "method": method,
                        "max_lead": max_lead,
                        "best_k": int(best_k),
                        "correlations": corrs,
                        "min_obs": int(min_obs),
                    },
                )
            )

    # Clean up temporary columns
    drop_cols = [c for c in work.columns if c.startswith("__lead_")]
    if drop_cols:
        work.drop(columns=drop_cols, inplace=True)

    notes = (
        f"Statistical suspicion on {len(candidates)} factor columns using method={method}, "
        f"max_lead={max_lead}, threshold={threshold}."
    )
    return LeakDetectionResult(inspected_columns=inspected, violations=violations, notes=notes)

