"""
Utility helpers to time any operator implemented in operators.py.

The main entry point is ``operator_analysis`` which mirrors the calling
conventions used throughout examples such as ``examples/example_advanced.py``.
It loads the requested operator by name, executes it once, and prints the time
spent on the computation.
"""

import sys
import time
from math import ceil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl

_ROOT_DIR = Path(__file__).resolve().parent.parent # switch to root path
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
from operators import Operators

_WINDOW_PARAM_ALIASES = ("windows", "window", "periods", "period", "span", "spans", "weights")
_COL_PARAM_PRIORITY = ("cols", "col_pairs", "col")
_DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
_REL_EPS = 1e-12

# -------------------------------------------
#                   Tools                   |
# -------------------------------------------

def _save_for_check(expr: str, df):
    """
    Persist a 10% head sample for quick verification.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expr must be a non-empty string when saving a check file.")
    if df is None:
        raise ValueError("df must not be None when saving a check file.")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if isinstance(df, pl.DataFrame):
        total_rows = df.height
        take_n = max(1, ceil(total_rows / 10)) if total_rows else 0
        sample_df = df.head(take_n).to_pandas()
    elif isinstance(df, pd.DataFrame):
        total_rows = len(df)
        take_n = max(1, ceil(total_rows / 10)) if total_rows else 0
        sample_df = df.head(take_n)
    else:
        raise TypeError("df must be a pandas or polars DataFrame when saving a check file.")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _DATA_DIR / f"{expr}_check.parquet"
    sample_df.to_parquet(output_path, index=False)
    print(f"[operator_analysis] Saved verification sample to {output_path}")


def _accuracy_check(expr: str, df, rel_tol: float) -> bool:
    """
    Compare current results with the stored verification sample within the given relative tolerance.
    """
    if rel_tol <= 0:
        raise ValueError("rel_tol must be positive.")

    check_path = _DATA_DIR / f"{expr}_check.parquet"
    if not check_path.exists():
        raise FileNotFoundError(
            f"No verification file found for '{expr}'. Run operator_analysis(..., if_save=True) first."
        )

    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if isinstance(df, pl.DataFrame):
        current_df = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        current_df = df
    else:
        raise TypeError("df must be a pandas or polars DataFrame when checking accuracy.")

    baseline_df = pd.read_parquet(check_path)
    if baseline_df.empty:
        raise ValueError(f"The verification file for '{expr}' is empty; cannot run accuracy check.")
    if current_df.empty:
        raise ValueError("Current operator output is empty; cannot run accuracy check.")

    compare_len = min(len(baseline_df), len(current_df))
    baseline_df = baseline_df.head(compare_len).reset_index(drop=True)
    current_df = current_df.head(compare_len).reset_index(drop=True)

    baseline_numeric = baseline_df.select_dtypes(include=[np.number])
    current_numeric = current_df[baseline_numeric.columns.intersection(current_df.columns)]
    if current_numeric.empty:
        raise ValueError("No overlapping numeric columns were found for accuracy checking.")

    diff = (current_numeric - baseline_numeric[current_numeric.columns]).abs()
    denom = np.maximum(baseline_numeric[current_numeric.columns].abs(), _REL_EPS)
    rel_diff = diff / denom

    # Treat NaNs that match on both sides as zero diff
    both_nan_mask = baseline_numeric[current_numeric.columns].isna() & current_numeric.isna()
    rel_diff = rel_diff.mask(both_nan_mask, other=0)

    exceeds = rel_diff > rel_tol
    if exceeds.any().any():
        # Look across the full comparison window and report the single worst offender.
        masked_rel_diff = rel_diff.where(exceeds, other=np.nan)
        worst_col = masked_rel_diff.max(axis=0).idxmax()
        worst_row = masked_rel_diff[worst_col].idxmax()
        actual = current_numeric.at[worst_row, worst_col]
        expected = baseline_numeric.at[worst_row, worst_col]
        worst_diff = masked_rel_diff.at[worst_row, worst_col]
        raise AssertionError(
            f"Accuracy check failed for '{expr}' at row {worst_row}, column '{worst_col}': "
            f"expected {expected}, got {actual}, relative diff {worst_diff:.2e} "
            f"(tolerance {rel_tol})."
        )

    print(f"[operator_analysis] Accuracy check passed for '{expr}' within tolerance {rel_tol}.")
    return True


def _get_param_names(func: Callable) -> Sequence[str]:
    """
    Extract the positional argument names from a function object.
    """
    code = func.__code__
    return code.co_varnames[: code.co_argcount]


def _resolve_column_argument(param_names: Sequence[str], cols: Optional[Union[str, Sequence[str]]]):
    """
    Decide how to feed the ``cols`` argument based on the operator signature.
    """
    if cols is None:
        column_param = next((name for name in _COL_PARAM_PRIORITY if name in param_names), None)
        if not column_param:
            return {}
        raise ValueError(f"Operator expects argument '{column_param}' but no columns were provided.")

    def _is_col_pairs(value) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        return all(isinstance(it, (list, tuple)) and len(it) == 2 for it in value)

    if "col_pairs" in param_names and _is_col_pairs(cols):
        return {"col_pairs": cols}

    column_param = next((name for name in _COL_PARAM_PRIORITY if name in param_names), None)
    if not column_param:
        return {}

    if column_param == "col":
        if isinstance(cols, (list, tuple)):
            if len(cols) != 1:
                raise ValueError("This operator expects a single column name; pass cols=['col_name'].")
            value = cols[0]
        else:
            value = cols
        return {"col": value}

    return {column_param: cols}

# -------------------------------------------
#                  Functions                |
# -------------------------------------------

def available_operators() -> List[str]:
    """
    Return a sorted list of callable operator names defined on Operators.
    """
    operator_names: List[str] = []
    for name in dir(Operators):
        if name.startswith("_"):
            continue
        attr = getattr(Operators, name)
        if callable(attr):
            operator_names.append(name)
    return sorted(operator_names)


def operator_analysis(
    df: pd.DataFrame,
    cols: Optional[Union[str, Sequence[str]]],
    expr: str,
    windows: Optional[Union[int, Sequence[int]]] = None,
    price_col: Optional[str] = None,
    group_col: Optional[str] = None,
    if_save: bool = False,
    accuracy_tol: Optional[float] = 1e-5,
):
    """
    Run a single operator by name and report its execution time.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. The function copies it before benchmarking to avoid side effects.
    cols : str | Sequence[str] | None
        Column(s) passed to the operator. Acts as the price column if the operator requires one.
    expr : str
        Operator name as defined in operators.py (e.g. ``'ts_mean'``).
    windows : int | Sequence[int] | None
        Optional argument shared by ``windows``, ``periods``, ``span`` and similar parameters.
    price_col : str | None
        Explicit price column; falls back to ``cols`` when possible.
    group_col : str | None
        Optional group column for time-series or cross-section operations.
    if_save : bool
        When True, persist a 10% head sample of the result for later spot checking.
    accuracy_tol : float | None
        Relative tolerance for `_accuracy_check`. Pass ``None`` to skip the comparison.
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a string referencing a method defined on Operators.")
    if not hasattr(Operators, expr):
        raise ValueError(f"Operator '{expr}' was not found. Use available_operators() to inspect names.")

    operator = getattr(Operators, expr)
    param_names = _get_param_names(operator)

    call_kwargs = {}
    call_kwargs.update(_resolve_column_argument(param_names, cols))

    if "price_col" in param_names:
        if price_col:
            call_kwargs["price_col"] = price_col
        elif isinstance(cols, str):
            call_kwargs["price_col"] = cols
        else:
            raise ValueError("This operator needs a price column. Provide price_col or pass cols as a string.")

    if group_col is not None and "group_col" in param_names:
        call_kwargs["group_col"] = group_col

    if windows is not None:
        for alias in _WINDOW_PARAM_ALIASES:
            if alias in param_names:
                call_kwargs[alias] = windows

    # Shallow copy keeps timing fair without modifying the caller's frame.
    if isinstance(df, pd.DataFrame):
        work_df = df.copy(deep=True)
    elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        work_df = df.clone()
    else:
        raise TypeError("df must be a pandas or polars DataFrame/LazyFrame.")
    
    start = time.perf_counter()
    result_df = operator(work_df, **call_kwargs)
    elapsed = time.perf_counter() - start
    print(f"[operator_analysis] {expr} finished in {elapsed * 1000:.3f} ms")
    
    if if_save:
        _save_for_check(expr, result_df)
    if accuracy_tol is not None and if_save is not True:
        _accuracy_check(expr, result_df, accuracy_tol)

    return
