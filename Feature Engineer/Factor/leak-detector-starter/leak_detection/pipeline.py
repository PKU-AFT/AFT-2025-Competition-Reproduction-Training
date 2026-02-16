"""Pipeline utilities to run a detector and persist its outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping

import pandas as pd

from factors import FactorDefinition, get_registered_factors

from .data import describe_dataset, load_sample_factors
from .detector import LeakDetectionResult, detect_leaks
from .reporting import export_all
from .time_utils import TimeIndexConfig, prepare_for_leak_detection

DetectorFn = Callable[..., LeakDetectionResult]


def run_pipeline(
    detector: DetectorFn = detect_leaks,
    *,
    data_path: Path | None = None,
    output_dir: Path | str = "outputs/manual",
    factor_definitions: Mapping[str, FactorDefinition] | None = None,
    tolerance: float = 1e-9,
    time_config: TimeIndexConfig | None = None,
    entity_col: str = "symbol",
    time_col: str = "date",
) -> Dict[str, Path]:
    """Run the leak detector end-to-end and persist reports.

    Parameters
    ----------
    detector : DetectorFn
        Detector function to use.
    data_path : Path, optional
        Path to data file. If None, uses default sample data.
    output_dir : Path or str
        Output directory for reports.
    factor_definitions : Mapping[str, FactorDefinition], optional
        Factor definitions to test. If None, uses registered factors.
    tolerance : float
        Numerical tolerance for leak detection.
    time_config : TimeIndexConfig, optional
        Configuration for automatic time index flattening.
        If None, uses default config (auto-detection enabled).
        Set to TimeIndexConfig(enabled=False) to disable time flattening.
    entity_col : str
        Entity column name (default: "symbol").
    time_col : str
        Time column name (default: "date").

    Returns
    -------
    dict
        Mapping with keys ``summary``/``json``/``markdown`` pointing to generated artefacts.

    Examples
    --------
    # Basic usage (auto time flattening)
    >>> artefacts = run_pipeline()

    # With custom time configuration
    >>> time_cfg = TimeIndexConfig(
    ...     column_mapping={'year': 'fiscal_year'},
    ...     verbose=True
    ... )
    >>> artefacts = run_pipeline(time_config=time_cfg)

    # Disable time flattening
    >>> artefacts = run_pipeline(time_config=TimeIndexConfig(enabled=False))
    """

    # Load data
    df = load_sample_factors(data_path)

    # Prepare time index (flatten if needed)
    df_processed, actual_time_col, time_report = prepare_for_leak_detection(
        df,
        time_config=time_config,
        entity_col=entity_col,
        time_col=time_col,
    )

    # Get descriptor (use processed data if flattening occurred)
    descriptor = describe_dataset(df_processed if time_report['status'] == 'success' else df)

    factor_definitions = factor_definitions or get_registered_factors()

    try:
        result = detector(
            df_processed,
            factor_definitions=factor_definitions,
            entity_col=entity_col,
            time_col=actual_time_col,
            tolerance=tolerance,
        )
    except NotImplementedError as exc:  # pragma: no cover - clarity for students
        raise RuntimeError(
            "检测逻辑尚未实现。请先在 leak_detection/detector.py 中完成 detect_leaks 函数。"
        ) from exc

    if result.notes is None:
        span = f"{descriptor.symbols[0]}…{descriptor.symbols[-1]}" if descriptor.symbols else "N/A"
        notes = (
            f"Inspected {len(result.inspected_columns)} factor functions across {descriptor.rows} rows "
            f"({span}) from {descriptor.start_date} to {descriptor.end_date}."
        )

        # Add time processing info to notes
        if time_report['status'] == 'success':
            detected = time_report.get('detected_mapping', {})
            notes += f"\n\n时间展平: 成功展平 {len(detected)} 个时间列: {list(detected.keys())}"
        elif time_report['status'] == 'skipped':
            notes += "\n\n时间展平: 已跳过（数据已有标准时间索引）"

        result.notes = notes

    output_dir = Path(output_dir)
    artefacts = export_all(result, output_dir)

    # Save time processing report
    if time_report['status'] in ('success', 'skipped'):
        time_report_path = output_dir / "time_processing.txt"
        with open(time_report_path, 'w', encoding='utf-8') as f:
            f.write("时间索引处理报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"状态: {time_report['status']}\n")
            if time_report['status'] == 'success':
                f.write(f"索引名称: {time_report.get('index_name', 'N/A')}\n")
                f.write(f"检测到的时间列: {time_report.get('detected_mapping', {})}\n")
                f.write(f"检测方法: {time_report.get('detection_methods', {})}\n")
                f.write(f"原始列: {time_report.get('original_columns', [])}\n")
        artefacts['time_report'] = time_report_path

    return artefacts


def preview_dataset(rows: int = 5, *, data_path: Path | None = None) -> pd.DataFrame:
    """Return the first ``rows`` rows of the dataset for quick inspection."""

    df = load_sample_factors(data_path)
    return df.head(rows).copy()
