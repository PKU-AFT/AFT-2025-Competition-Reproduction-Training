"""Starter file that students will complete with their leak detection logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import numpy as np

from factors import FactorDefinition


def _classify_time_column(df: pd.DataFrame, col: str) -> str:
    """Classify a time column into date-only, time-only, or datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to classify.

    Returns
    -------
    str
        One of: 'datetime', 'date', 'time', 'unknown'
    """
    # Check data type
    dtype = df[col].dtype

    # datetime64 is a full datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        # Check if it has time component
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            has_time = any(
                pd.to_datetime(val).time() != pd.Timestamp('00:00:00').time()
                for val in sample
            )
            return 'datetime' if has_time else 'date'
        return 'datetime'

    # Try to infer from sample values
    sample = df[col].dropna().head(100)
    if len(sample) == 0:
        return 'unknown'

    # Convert sample to string for inspection
    try:
        first_val = str(sample.iloc[0])

        # Check for date-only format (YYYY-MM-DD, YYYY/MM/DD, etc.)
        if len(first_val) <= 10 and ('-' in first_val or '/' in first_val):
            return 'date'

        # Check for time-only format (HH:MM:SS, HH:MM)
        if ':' in first_val and len(first_val) <= 8:
            return 'time'

        # Check for datetime format
        if ' ' in first_val or 'T' in first_val:
            return 'datetime'

    except Exception:
        pass

    return 'unknown'


def _detect_complementary_time_cols(
    df: pd.DataFrame,
    time_cols: Sequence[str]
) -> tuple[str | None, str | None]:
    """Detect if there's a complementary pair of date and time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_cols : Sequence[str]
        Candidate time columns.

    Returns
    -------
    tuple[str | None, str | None]
        (date_col, time_col) if found, otherwise (None, None)
    """
    date_cols = []
    time_cols_only = []

    for col in time_cols:
        col_type = _classify_time_column(df, col)
        if col_type == 'date':
            date_cols.append(col)
        elif col_type == 'time':
            time_cols_only.append(col)

    # If we have exactly one date and one time column, they're complementary
    if len(date_cols) == 1 and len(time_cols_only) == 1:
        return date_cols[0], time_cols_only[0]

    return None, None


def _combine_date_time_columns(
    df: pd.DataFrame,
    date_col: str,
    time_col: str,
    combined_name: str = 'datetime_combined'
) -> tuple[pd.DataFrame, str]:
    """Combine separate date and time columns into a single datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.
    time_col : str
        Time column name.
    combined_name : str
        Name for the combined column.

    Returns
    -------
    tuple[pd.DataFrame, str]
        (DataFrame with combined column, combined column name)
    """
    df_copy = df.copy()

    # Combine date and time
    try:
        df_copy[combined_name] = pd.to_datetime(
            df_copy[date_col].astype(str) + ' ' + df_copy[time_col].astype(str)
        )
    except Exception as e:
        raise ValueError(
            f"Failed to combine date column '{date_col}' and time column '{time_col}': {e}"
        ) from e

    return df_copy, combined_name


def _detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect potential time columns in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    List[str]
        List of detected time column names, sorted by granularity (finest first).
    """
    time_cols = []

    for col in df.columns:
        # Check if column is datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append(col)
        # Check if column name suggests it's a time column
        elif any(keyword in col.lower() for keyword in ['time', 'date', 'datetime', 'timestamp', '时间', '日期']):
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].head(100))
                time_cols.append(col)
            except (ValueError, TypeError):
                pass

    if not time_cols:
        return []

    # Sort by granularity (finest first)
    # Estimate granularity by counting unique values per entity
    granularity_scores = {}
    for col in time_cols:
        # Higher unique count = finer granularity
        unique_count = df[col].nunique()
        granularity_scores[col] = unique_count

    # Sort descending (finest granularity first)
    time_cols_sorted = sorted(time_cols, key=lambda c: granularity_scores[c], reverse=True)

    return time_cols_sorted


def _select_best_time_column(
    df: pd.DataFrame,
    time_col: str | None = None,
    time_cols: Sequence[str] | None = None,
    auto_detect: bool = True,
    combine_time_cols: bool = True,
    combined_time_name: str = 'datetime_combined',
) -> tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Select or combine the most appropriate time column for leak detection.

    Strategy:
    1. If time_col is specified and exists, use it
    2. If time_cols is specified:
       a. Check for complementary date+time columns → combine if enabled
       b. Otherwise select the finest granularity one
    3. If auto_detect=True, automatically detect time columns

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str, optional
        Single time column name (legacy parameter).
    time_cols : Sequence[str], optional
        Multiple time column candidates.
    auto_detect : bool
        Enable automatic time column detection.
    combine_time_cols : bool, default True
        Allow combining complementary date and time columns.
    combined_time_name : str, default 'datetime_combined'
        Name for the combined column.

    Returns
    -------
    pd.DataFrame
        DataFrame (possibly with new combined column).
    str
        Selected/combined time column name.
    dict
        Selection/combination report with metadata.
    """
    report = {
        'method': None,
        'candidates': [],
        'selected': None,
        'reason': None,
    }

    # Priority 1: Use time_cols if specified
    if time_cols:
        # Validate all candidates exist
        missing = [c for c in time_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Specified time_cols not found in DataFrame: {missing}")

        # Classify all time columns
        time_classifications = {}
        for col in time_cols:
            time_classifications[col] = _classify_time_column(df, col)

        # Check if there's already a complete datetime column
        datetime_cols = [col for col, typ in time_classifications.items() if typ == 'datetime']

        # If there's already a datetime column, prefer it over combination
        if datetime_cols:
            # Select the datetime column with finest granularity
            granularity_scores = {col: df[col].nunique() for col in datetime_cols}
            selected = max(datetime_cols, key=lambda c: granularity_scores[c])

            report.update({
                'method': 'explicit_multi',
                'candidates': list(time_cols),
                'datetime_cols': datetime_cols,
                'granularity_scores': {col: df[col].nunique() for col in time_cols},
                'selected': selected,
                'reason': f'Selected existing datetime column "{selected}" (finest granularity: {granularity_scores[selected]} unique values)',
            })

            return df, selected, report

        # No complete datetime column, check for complementary date+time columns
        if combine_time_cols:
            date_col, time_col_only = _detect_complementary_time_cols(df, time_cols)
            if date_col and time_col_only:
                # Combine them
                df_combined, combined_col = _combine_date_time_columns(
                    df, date_col, time_col_only, combined_time_name
                )

                report.update({
                    'method': 'combination',
                    'candidates': list(time_cols),
                    'date_col': date_col,
                    'time_col': time_col_only,
                    'combined_col': combined_col,
                    'selected': combined_col,
                    'reason': f'Combined "{date_col}" (date) and "{time_col_only}" (time) into "{combined_col}"',
                })

                return df_combined, combined_col, report

        # No datetime column and no combination, select by granularity
        granularity_scores = {col: df[col].nunique() for col in time_cols}
        selected = max(time_cols, key=lambda c: granularity_scores[c])

        report.update({
            'method': 'explicit_multi',
            'candidates': list(time_cols),
            'granularity_scores': granularity_scores,
            'selected': selected,
            'reason': f'Selected "{selected}" as it has the finest granularity ({granularity_scores[selected]} unique values)',
        })

        return df, selected, report

    # Priority 2: Use time_col if specified
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"Specified time_col '{time_col}' not found in DataFrame columns")

        report.update({
            'method': 'explicit_single',
            'candidates': [time_col],
            'selected': time_col,
            'reason': f'Using explicitly specified time column: {time_col}',
        })

        return df, time_col, report

    # Priority 3: Auto-detect
    if auto_detect:
        detected = _detect_time_columns(df)

        if not detected:
            raise ValueError(
                "No time columns detected in DataFrame. Please specify time_col or time_cols explicitly."
            )

        # Check for complementary columns in auto-detected
        if combine_time_cols and len(detected) >= 2:
            date_col, time_col_only = _detect_complementary_time_cols(df, detected)
            if date_col and time_col_only:
                df_combined, combined_col = _combine_date_time_columns(
                    df, date_col, time_col_only, combined_time_name
                )

                report.update({
                    'method': 'auto_combination',
                    'candidates': detected,
                    'date_col': date_col,
                    'time_col': time_col_only,
                    'combined_col': combined_col,
                    'selected': combined_col,
                    'reason': f'Auto-detected and combined "{date_col}" + "{time_col_only}" into "{combined_col}"',
                })

                return df_combined, combined_col, report

        # No combination, select finest granularity
        selected = detected[0]  # Finest granularity

        report.update({
            'method': 'auto_detect',
            'candidates': detected,
            'selected': selected,
            'reason': f'Auto-detected {len(detected)} time column(s), selected "{selected}" with finest granularity',
        })

        return df, selected, report

    # Fallback: default to 'date'
    if 'date' in df.columns:
        report.update({
            'method': 'fallback',
            'candidates': ['date'],
            'selected': 'date',
            'reason': 'Using default column name "date"',
        })
        return df, 'date', report

    raise ValueError(
        "Could not determine time column. Please specify time_col or time_cols explicitly."
    )


@dataclass
class LeakViolation:
    """A single suspected leak signal."""

    column: str
    signal_type: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakDetectionResult:
    """Container for detector output used throughout the project."""

    inspected_columns: List[str]
    violations: List[LeakViolation] = field(default_factory=list)
    notes: str | None = None


def detect_leaks(
    df: pd.DataFrame,
    *,
    factor_definitions: Mapping[str, FactorDefinition],
    entity_col: str = "symbol",
    time_col: str | None = None,
    time_cols: Sequence[str] | None = None,
    auto_detect_time: bool = True,
    combine_time_cols: bool = True,
    combined_time_name: str = 'datetime_combined',
    tolerance: float = 1e-9,
    row_guard: int | None = None,
    replay_mode: str = "full",
    recent_window: int = 7,
    max_entities: int | None = None,
    replay_lookback: int | None = None,
    analysis_mode: str = "full",
    analysis_recent_days: int = 3,
    day_col: str | None = None,
) -> LeakDetectionResult:
    """Detect suspicious factor *code* that may leak future information.

    要求：遍历 ``factor_definitions`` 内的函数，构造足够的检验来识别未来信息泄露。
    建议思路：

    1. **时间前推重算 (forward-only replay)**：逐日滚动截取 ``df`` 的前缀，让因子函数在仅含 ``<= as_of`` 数据的情形下复现最后一行的取值，若结果与完整数据跑出来的不一致，说明函数依赖未来信息。
    2. **鲁棒性对比**：可对比训练/验证时间段、或不同切片下的统计量，找到与未来标签过于一致的因子。
    3. **自定义规则**：例如静态代码扫描、参数签名检查、或针对函数中 ``shift(-1)`` ``lead`` 等操作的简单 AST 分析。

    返回 ``LeakDetectionResult``，其中 ``violations`` 记录每个疑似泄露的因子函数。

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least entity/time columns plus the data needed by factors.
    factor_definitions : Mapping[str, FactorDefinition]
        Registry describing how to compute each factor.
    entity_col : str, default "symbol"
        Entity key column.
    time_col : str, optional
        Single time ordering column (legacy parameter for backward compatibility).
        If None and auto_detect_time=True, will auto-detect time column.
    time_cols : Sequence[str], optional
        Multiple time column candidates. If provided, the column with finest granularity
        (most unique values) will be selected automatically. This is useful when your
        data has multiple time columns like 'date' and 'datetime'.
    auto_detect_time : bool, default True
        Enable automatic time column detection. If True and neither time_col nor time_cols
        is specified, will search for columns with datetime type or time-related names.
    combine_time_cols : bool, default True
        Allow combining complementary date and time columns. If True and time_cols contains
        one date column (e.g., '2023-01-02') and one time column (e.g., '09:30:00'),
        they will be automatically combined into a single datetime column. This is useful
        when your data has separate date and time columns that need to be merged.
    combined_time_name : str, default 'datetime_combined'
        Name for the column created when combining date and time columns. Only used
        when combine_time_cols=True and complementary columns are detected.
    tolerance : float, default 1e-9
        Numeric tolerance for comparing replay values.
    row_guard : int | None, default None
        If provided, limit computations to the first ``row_guard`` rows after sorting. Useful
        when probing extremely large datasets.
    replay_mode : {"full", "latest", "recent"}, default "full"
        Controls which as-of timestamps are replayed.
    recent_window : int, default 7
        Number of timestamps to keep when ``replay_mode="recent"``.
    max_entities : int | None, default None
        If provided, restricts the detector to the first ``max_entities`` entities (after sorting).
    replay_lookback : int | None, default None
        If provided, each replay uses only the most recent ``replay_lookback`` timestamps (inclusive)
        up to ``as_of`` instead of the entire history.
    analysis_mode : {"full", "latest_day", "recent_days"}, default "full"
        Optional shortcut to limit the dataset internally before replay. ``"latest_day"`` keeps only
        the latest day/time slice and forces ``replay_mode='latest'``; ``"recent_days"`` keeps the last
        ``analysis_recent_days`` day slices while still checking only the final timestamp.
    analysis_recent_days : int, default 3
        Number of day slices retained when ``analysis_mode='recent_days'``.
    day_col : str | None, default None
        Optional column indicating the trading day. If supplied and present, it is used to compute
        latest/recent windows; otherwise the selected ``time_col`` values are used directly.

    Returns
    -------
    LeakDetectionResult
        Container with inspected columns, detected violations, and optional notes.

    Examples
    --------
    Scenario 1: Single time column (traditional usage)

    >>> result = detect_leaks(df, factor_definitions=factors, time_col='date')

    Scenario 2: Multiple time columns (auto-select finest granularity)

    >>> # df has both 'date' and 'datetime' columns
    >>> result = detect_leaks(df, factor_definitions=factors,
    ...                       time_cols=['date', 'datetime'])  # Will use 'datetime'

    Scenario 3: Auto-detection

    >>> # Automatically detect time columns from df
    >>> result = detect_leaks(df, factor_definitions=factors, auto_detect_time=True)

    Notes
    -----
    **Multi-Time Column Strategy:**

    When your data has multiple time columns (e.g., 'date' for daily aggregation and
    'datetime' for intraday timestamps), the detector will:

    1. If `time_cols` is provided: Select the column with finest granularity (most unique values)
    2. If `time_col` is provided: Use it directly (backward compatible)
    3. If neither: Auto-detect time columns if `auto_detect_time=True`

    **Granularity Selection:**

    For leak detection, finer granularity is always preferred because:
    - It can detect more subtle timing leaks (e.g., using current bar's close price)
    - It preserves all temporal information from the coarser column
    - Factor functions must work correctly at the finest time resolution
    """

    # ====== Multi-Time Column Support ======
    # Select the best time column based on user input and auto-detection
    if time_col is None and time_cols is None and not auto_detect_time:
        # Fallback to 'date' for backward compatibility
        time_col = 'date'

    work_df, selected_time_col, time_selection_report = _select_best_time_column(
        df,
        time_col=time_col,
        time_cols=time_cols,
        auto_detect=auto_detect_time,
        combine_time_cols=combine_time_cols,
        combined_time_name=combined_time_name,
    )

    # Use the selected/combined time column for all operations
    time_col = selected_time_col

    # Basic validations and sorting for deterministic behavior
    if entity_col not in work_df.columns:
        raise ValueError(f"DataFrame must contain entity column '{entity_col}'")

    if time_col not in work_df.columns:
        raise ValueError(
            f"Selected time column '{time_col}' not found in DataFrame columns. "
            f"Selection method: {time_selection_report['method']}, "
            f"Reason: {time_selection_report['reason']}"
        )
    work_df = work_df.sort_values([entity_col, time_col]).reset_index(drop=True)
    if row_guard is not None and row_guard > 0:
        work_df = work_df.head(row_guard).copy()
    if max_entities is not None and max_entities > 0:
        keep_entities = work_df[entity_col].drop_duplicates().head(max_entities).tolist()
        work_df = work_df[work_df[entity_col].isin(keep_entities)].copy()

    day_column = day_col if day_col and day_col in work_df.columns else None
    if analysis_mode not in {"full", "latest_day", "recent_days"}:
        raise ValueError("analysis_mode must be one of {'full','latest_day','recent_days'}")

    effective_replay_mode = replay_mode
    effective_replay_lookback = replay_lookback

    if analysis_mode == "latest_day":
        if day_column:
            latest_day = work_df[day_column].max()
            work_df = work_df[work_df[day_column] == latest_day].copy()
        else:
            latest_time = work_df[time_col].max()
            work_df = work_df[work_df[time_col] == latest_time].copy()
        effective_replay_mode = "latest"
    elif analysis_mode == "recent_days":
        days = max(1, int(analysis_recent_days))
        if day_column:
            ordered_days = pd.Index(work_df[day_column].unique()).sort_values()
            keep_days = ordered_days[-days:]
            work_df = work_df[work_df[day_column].isin(keep_days)].copy()
            ref_length = len(keep_days)
        else:
            ordered_times = pd.Index(work_df[time_col].unique()).sort_values()
            keep_times = ordered_times[-days:]
            work_df = work_df[work_df[time_col].isin(keep_times)].copy()
            ref_length = len(keep_times)
        effective_replay_mode = "latest"
        if effective_replay_lookback is None:
            effective_replay_lookback = ref_length

    # After all filtering steps re-index to keep positional logic consistent
    work_df = work_df.reset_index(drop=True)

    inspected = []
    violations: List[LeakViolation] = []

    # Build group index map (by entity) for split masks
    groups = list(work_df.groupby(entity_col, sort=False))

    # Helper for fuzzy equality with NaN handling
    def _is_close(a: float, b: float, tol: float) -> bool:
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
        return bool(abs(float(a) - float(b)) <= tol)

    # Pre-compute a mask for train/validation consistency stats
    # Split chronologically per entity with 80/20 ratio
    n = len(work_df)
    train_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)
    for _, part in groups:
        idxs = part.index.to_numpy()
        split_idx = int(len(idxs) * 0.8)
        split_idx = max(1, min(len(idxs) - 1, split_idx))
        train_mask[idxs[:split_idx]] = True
        valid_mask[idxs[split_idx:]] = True

    # Iterate over factors
    for name in sorted(factor_definitions.keys()):
        definition = factor_definitions[name]
        inspected.append(name)

        # Full run baseline on the entire dataset
        try:
            full_series = definition.function(work_df.copy())
        except Exception as exc:
            # If a factor crashes, mark as violation with zero confidence but include error details
            violations.append(
                LeakViolation(
                    column=name,
                    signal_type="factor_error",
                    confidence=0.0,
                    details={"error": repr(exc)},
                )
            )
            continue

        if not isinstance(full_series, pd.Series) or len(full_series) != len(work_df):
            violations.append(
                LeakViolation(
                    column=name,
                    signal_type="shape_mismatch",
                    confidence=1.0,
                    details={"message": "factor output must be a Series aligned to df index"},
                )
            )
            continue

        # Allocate per-row flags to support consistency stats later
        checked = np.zeros(n, dtype=bool)
        mismatched = np.zeros(n, dtype=bool)

        # Full-market as-of replay by unique dates (preserves cross-sectional context)
        all_times = pd.Index(work_df[time_col].unique()).sort_values()
        unique_times = all_times
        if effective_replay_mode not in {"full", "latest", "recent"}:
            raise ValueError(
                "replay_mode must be one of {'full','latest','recent'}, "
                f"got {effective_replay_mode!r}"
            )
        if effective_replay_mode == "latest":
            unique_times = unique_times[-1:]
        elif effective_replay_mode == "recent":
            tail = max(1, int(recent_window))
            unique_times = unique_times[-tail:]
        for as_of in unique_times:
            # Slice all rows up to and including as_of across the entire market
            if effective_replay_lookback and effective_replay_lookback > 0:
                try:
                    idx = all_times.get_loc(as_of)
                except KeyError:
                    idx = len(all_times) - 1
                start_idx = max(0, idx - effective_replay_lookback + 1)
                allowed = all_times[start_idx : idx + 1]
                asof_mask = work_df[time_col].isin(allowed)
            else:
                asof_mask = work_df[time_col] <= as_of
            df_asof = work_df.loc[asof_mask].copy()

            try:
                sub_series = definition.function(df_asof)
            except Exception as exc:
                violations.append(
                    LeakViolation(
                        column=name,
                        signal_type="replay_error",
                        confidence=0.0,
                        details={"as_of": str(as_of), "error": repr(exc)},
                    )
                )
                # Skip further checks for this as_of since factor couldn't be computed
                continue

            # Compare only the rows that occur exactly at as_of
            at_asof_mask = work_df[time_col] == as_of
            idxs_at_asof = work_df.index[at_asof_mask]

            for idx in idxs_at_asof:
                full_val = full_series.iloc[idx]

                # Strict index alignment: factor MUST preserve original indices
                if idx not in sub_series.index:
                    # Index not preserved - report as violation instead of guessing
                    violations.append(
                        LeakViolation(
                            column=name,
                            signal_type="index_not_preserved",
                            confidence=1.0,
                            details={
                                "message": (
                                    f"Factor did not preserve DataFrame index at position {idx}. "
                                    f"The factor must return a Series with the same index as the input DataFrame."
                                ),
                                "as_of": str(as_of),
                                "missing_index": int(idx),
                                "entity": str(work_df.iloc[idx][entity_col]),
                                "time": str(work_df.iloc[idx][time_col]),
                                "suggestion": (
                                    "Ensure your factor function returns a Series that preserves the original "
                                    "DataFrame index. Avoid operations like reset_index(drop=True), or ensure "
                                    "you restore the original index before returning."
                                )
                            }
                        )
                    )
                    # Skip further replay checks for this factor
                    break

                replay_val = sub_series.loc[idx]

                # Mark checks excluding both-NaN
                if not (pd.isna(full_val) and pd.isna(replay_val)):
                    checked[idx] = True
                    if not _is_close(full_val, replay_val, tolerance):
                        mismatched[idx] = True

        total_checked = int(checked.sum())
        total_mismatch = int(mismatched.sum())
        confidence = (total_mismatch / total_checked) if total_checked > 0 else 0.0

        # If any mismatch occurred, raise a forward-replay violation
        if total_mismatch > 0:
            # Train/validation split statistics for robustness
            train_checked = int((checked & train_mask).sum())
            valid_checked = int((checked & valid_mask).sum())
            train_mismatch = int((mismatched & train_mask).sum())
            valid_mismatch = int((mismatched & valid_mask).sum())
            train_rate = (train_mismatch / train_checked) if train_checked > 0 else 0.0
            valid_rate = (valid_mismatch / valid_checked) if valid_checked > 0 else 0.0

            # Prepare a few illustrative mismatches
            sample_rows: List[Dict[str, Any]] = []
            for idx in np.where(mismatched)[0][:5].tolist():
                sample_rows.append(
                    {
                        "index": int(idx),
                        entity_col: work_df.iloc[idx][entity_col],
                        time_col: str(work_df.iloc[idx][time_col]),
                        "full": None if pd.isna(full_series.iloc[idx]) else float(full_series.iloc[idx]),
                        "replay": None,  # filled below, requires re-evaluation in subset; costly to recompute
                    }
                )

            violations.append(
                LeakViolation(
                    column=name,
                    signal_type="forward_replay",
                    confidence=float(confidence),
                    details={
                        "total_checked": total_checked,
                        "total_mismatch": total_mismatch,
                        "train_checked": train_checked,
                        "train_mismatch": train_mismatch,
                        "train_rate": float(train_rate),
                        "valid_checked": valid_checked,
                        "valid_mismatch": valid_mismatch,
                        "valid_rate": float(valid_rate),
                        "examples": sample_rows,
                    },
                )
            )

            # Optional secondary signal to emphasise consistency across time windows
            if train_mismatch > 0 and valid_mismatch > 0:
                violations.append(
                    LeakViolation(
                        column=name,
                        signal_type="consistency_check",
                        confidence=float(min(train_rate, valid_rate)),
                        details={
                            "train_rate": float(train_rate),
                            "valid_rate": float(valid_rate),
                            "note": "Mismatches present in both train and validation windows",
                        },
                    )
                )

    # Build notes with time column selection/combination info
    notes_parts = []
    if time_selection_report['method'] in ('explicit_multi', 'auto_detect', 'combination', 'auto_combination'):
        if time_selection_report['method'] in ('combination', 'auto_combination'):
            notes_parts.append(
                f"时间列组合: {time_selection_report['reason']}"
            )
            notes_parts.append(
                f"组合详情: {time_selection_report['date_col']} (date) + "
                f"{time_selection_report['time_col']} (time) → "
                f"{time_selection_report['combined_col']}"
            )
        else:
            notes_parts.append(
                f"时间列选择: {time_selection_report['reason']}"
            )

        if time_selection_report.get('candidates'):
            notes_parts.append(
                f"候选列: {time_selection_report['candidates']}"
            )
        if time_selection_report.get('granularity_scores'):
            notes_parts.append(
                f"粒度评分: {time_selection_report['granularity_scores']}"
            )

    notes = '\n'.join(notes_parts) if notes_parts else None

    return LeakDetectionResult(inspected_columns=inspected, violations=violations, notes=notes)
