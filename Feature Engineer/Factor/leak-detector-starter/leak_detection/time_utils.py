"""Time series preprocessing utilities for leak detection framework.

This module provides integration between AutoTimeIndexer and the leak detection pipeline,
enabling automatic flattening of discrete time columns before running leak detection.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, List, Callable
from dataclasses import dataclass

import pandas as pd

# Import AutoTimeIndexer from the same directory
from .Timeindex import AutoTimeIndexer


@dataclass
class TimeIndexConfig:
    """Configuration for time index flattening.

    Parameters
    ----------
    enabled : bool, default True
        Whether to enable time flattening.
    column_mapping : dict, optional
        Manual column mapping, e.g., {'year': 'fiscal_year', 'month': 'report_month'}.
    tz : str, optional
        Target timezone, e.g., 'UTC' or 'Asia/Shanghai'.
    index_name : str, default 'timestamp'
        Name for the resulting time index.
    drop_originals : bool, default True
        Whether to drop original time columns after flattening.
    use_regex : bool, default True
        Enable regex matching for column names with prefixes/suffixes.
    use_inference : bool, default True
        Enable value-based inference for meaningless column names.
    custom_heuristics : dict, optional
        Custom heuristic rules for additional language support.
    verbose : bool, default False
        Output detailed detection logs.
    auto_detect : bool, default True
        Automatically detect if data needs flattening (skip if already has datetime index).
    custom_time_levels : list, optional
        Custom time level column names, e.g., ['date_id', 'second_in_buckets'].
        When using this, must also provide custom_converter.
    custom_converter : callable, optional
        Custom converter function: func(df) -> pd.Series of timestamps.
        Input is DataFrame with custom time columns, output is timestamp Series.

    Custom Time Levels Example
    --------------------------
    ```python
    def my_converter(df):
        base_date = pd.Timestamp('2020-01-01')
        timestamps = (base_date +
                     pd.to_timedelta(df['date_id'], unit='D') +
                     pd.to_timedelta(df['second_in_buckets'], unit='s'))
        return timestamps

    config = TimeIndexConfig(
        custom_time_levels=['date_id', 'second_in_buckets'],
        custom_converter=my_converter
    )
    ```
    """

    enabled: bool = True
    column_mapping: Optional[Dict[str, str]] = None
    tz: Optional[str] = None
    index_name: str = 'timestamp'
    drop_originals: bool = True
    use_regex: bool = True
    use_inference: bool = True
    custom_heuristics: Optional[Dict[str, Set[str]]] = None
    verbose: bool = False
    auto_detect: bool = True
    custom_time_levels: Optional[List[str]] = None
    custom_converter: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def needs_time_flattening(df: pd.DataFrame, time_col: str = 'date') -> bool:
    """Check if DataFrame needs time flattening.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str
        Expected time column name.

    Returns
    -------
    bool
        True if data has discrete time columns that need flattening.
    """
    # Check 1: If time_col exists and is already datetime type
    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return False

    # Check 2: If index is already a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return False

    # Check 3: Look for discrete time components
    time_indicators = {'year', 'month', 'day', 'yr', 'y', 'm', 'd', '年', '月', '日'}
    df_cols_lower = {col.lower() for col in df.columns}

    # If at least 2 time components are present, likely needs flattening
    found_components = time_indicators & df_cols_lower
    if len(found_components) >= 2:
        return True

    return False


def flatten_time_index(
    df: pd.DataFrame,
    config: TimeIndexConfig,
    entity_col: str = 'symbol',
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Flatten discrete time columns into a unified timestamp index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with discrete time columns.
    config : TimeIndexConfig
        Configuration for time flattening.
    entity_col : str
        Entity column name (will be preserved).

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened time index.
    dict
        Detection report containing detected columns and methods.
    """
    if not config.enabled:
        return df, {'status': 'disabled'}

    # Auto-detect if flattening is needed (skip for custom time levels)
    if config.custom_time_levels is None:
        if config.auto_detect and not needs_time_flattening(df):
            if config.verbose:
                print("[TimeUtils] 数据已有标准时间索引，跳过展平步骤")
            return df, {'status': 'skipped', 'reason': 'already_has_datetime_index'}

    # Create AutoTimeIndexer
    indexer = AutoTimeIndexer(
        column_mapping=config.column_mapping,
        tz=config.tz,
        index_name=config.index_name,
        drop_originals=config.drop_originals,
        use_regex=config.use_regex,
        use_inference=config.use_inference,
        custom_heuristics=config.custom_heuristics,
        verbose=config.verbose,
        custom_time_levels=config.custom_time_levels,
        custom_converter=config.custom_converter,
    )

    if config.verbose:
        print("[TimeUtils] 开始展平时间列...")

    # Transform
    try:
        result_df = indexer.transform(df)
    except ValueError as e:
        raise ValueError(
            f"时间展平失败: {e}\n\n"
            "建议:\n"
            "1. 检查数据是否包含必需的时间列（year, month, day）\n"
            "2. 使用 TimeIndexConfig(column_mapping={...}) 手动指定列映射\n"
            "3. 启用 verbose=True 查看详细检测过程"
        ) from e

    if config.verbose:
        print(indexer.get_detection_report())

    # Build detection report
    report = {
        'status': 'success',
        'detected_mapping': indexer.detected_mapping_,
        'detection_methods': indexer.detection_method_,
        'original_columns': indexer.original_columns_,
        'index_name': config.index_name,
    }

    return result_df, report


def prepare_for_leak_detection(
    df: pd.DataFrame,
    *,
    time_config: TimeIndexConfig | None = None,
    entity_col: str = 'symbol',
    time_col: str = 'date',
) -> tuple[pd.DataFrame, str, Dict[str, str]]:
    """Prepare DataFrame for leak detection by flattening time index if needed.

    This is the main entry point for integrating time flattening into the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_config : TimeIndexConfig, optional
        Time flattening configuration. If None, uses default config.
    entity_col : str
        Entity column name.
    time_col : str
        Time column name (used for auto-detection).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame ready for leak detection.
    str
        The actual time column name to use in leak detection.
    dict
        Processing report.
    """
    config = time_config or TimeIndexConfig()

    if not config.enabled:
        return df, time_col, {'status': 'disabled'}

    # Flatten time index
    processed_df, report = flatten_time_index(df, config, entity_col)

    # Determine the time column to use
    # After flattening, the time is in the index
    if report['status'] == 'success':
        # If index was set, we need to reset it to make it a column for the detector
        # OR we modify the detector to work with DatetimeIndex
        # For now, let's keep index as index but add a column for compatibility
        if config.index_name not in processed_df.columns:
            processed_df[config.index_name] = processed_df.index
        actual_time_col = config.index_name
    else:
        actual_time_col = time_col

    return processed_df, actual_time_col, report


# Convenience function for quick usage
def auto_flatten_time(
    df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Quick utility to auto-flatten time columns with default settings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    verbose : bool
        Show detailed logs.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened time index.
    """
    config = TimeIndexConfig(verbose=verbose)
    result_df, _ = flatten_time_index(df, config)
    return result_df
