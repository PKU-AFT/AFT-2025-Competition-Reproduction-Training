"""CLI wrapper around the leak detection pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leak_detection.pipeline import run_pipeline
from leak_detection.time_utils import TimeIndexConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leak detection pipeline with automatic time index flattening"
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to the factors CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manual"),
        help="Directory where reports will be written",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numerical tolerance passed to the detector",
    )

    # Time flattening options
    time_group = parser.add_argument_group('time flattening options')
    time_group.add_argument(
        "--disable-time-flatten",
        action="store_true",
        help="Disable automatic time index flattening",
    )
    time_group.add_argument(
        "--time-verbose",
        action="store_true",
        help="Show detailed time flattening logs",
    )
    time_group.add_argument(
        "--time-tz",
        type=str,
        default=None,
        help="Timezone for time index (e.g., 'UTC', 'Asia/Shanghai')",
    )
    time_group.add_argument(
        "--time-index-name",
        type=str,
        default="timestamp",
        help="Name for the time index column",
    )
    time_group.add_argument(
        "--keep-time-columns",
        action="store_true",
        help="Keep original time columns after flattening",
    )

    args = parser.parse_args()

    # Build time configuration
    if args.disable_time_flatten:
        time_config = TimeIndexConfig(enabled=False)
    else:
        time_config = TimeIndexConfig(
            verbose=args.time_verbose,
            tz=args.time_tz,
            index_name=args.time_index_name,
            drop_originals=not args.keep_time_columns,
        )

    artefacts = run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        tolerance=args.tolerance,
        time_config=time_config,
    )

    print("\n" + "=" * 60)
    print("因子泄露检测完成")
    print("=" * 60)
    print("\n生成的报告文件:")
    for name, path in artefacts.items():
        print(f"  {name:15s}: {path}")
    print()


if __name__ == "__main__":
    main()
