"""CLI to run the statistical leak suspicion detector on precomputed factors.

Example:
    python scripts/run_stat_suspicion.py \
        --data /path/to/factors.parquet \
        --entity stock_id \
        --time date_id \
        --target target \
        --factor-prefix factor_ --factor-prefix feat_
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leak_detection.reporting import export_all, to_summary_frame
from leak_detection.statistical import detect_statistical_suspicion


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run statistical leak suspicion detector on precomputed factors")
    p.add_argument("--data", type=Path, required=True, help="Path to a CSV or Parquet with factors")
    p.add_argument("--entity", required=True, help="Entity column name (e.g., stock_id or symbol)")
    p.add_argument("--time", dest="time_col", required=True, help="Time column name (monotonic per entity)")
    p.add_argument("--target", default="target", help="Target column (default: target)")
    p.add_argument("--factor-prefix", dest="factor_prefixes", action="append", default=None,
                   help="Prefix for factor columns (can be passed multiple times)")
    p.add_argument("--factor-cols", dest="factor_cols", nargs="*", default=None,
                   help="Explicit factor columns (space-separated)")
    p.add_argument("--exclude", dest="exclude_cols", action="append", default=[],
                   help="Columns to exclude from factor candidates (can be repeated)")
    p.add_argument("--max-lead", type=int, default=3, help="Max lead horizon k (default 3)")
    p.add_argument("--min-obs", type=int, default=20, help="Min overlapping obs for correlation (default 20)")
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson", help="Correlation method")
    p.add_argument("--threshold", type=float, default=0.7, help="Suspicion threshold (default 0.7)")
    p.add_argument("--output", type=Path, default=Path("outputs/statistical"), help="Output directory")
    args = p.parse_args()

    df = _load_frame(args.data)

    result = detect_statistical_suspicion(
        df,
        entity_col=args.entity,
        time_col=args.time_col,
        target_col=args.target,
        factor_cols=args.factor_cols,
        include_prefixes=tuple(args.factor_prefixes) if args.factor_prefixes else None,
        exclude_cols=args.exclude_cols,
        max_lead=args.max_lead,
        min_obs=args.min_obs,
        method=args.method,
        threshold=args.threshold,
    )

    # Persist artefacts (CSV/JSON/Markdown) reusing existing reporting helpers
    out_dir = args.output
    artefacts = export_all(result, out_dir)

    print("Generated artefacts:")
    for name, path in artefacts.items():
        print(f"- {name}: {path}")

    # Also print a quick summary to stdout
    frame = to_summary_frame(result)
    if frame.empty:
        print("No suspicious factors above the threshold.")
    else:
        print("\nTop suspicious factors:")
        print(frame.sort_values("confidence", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

