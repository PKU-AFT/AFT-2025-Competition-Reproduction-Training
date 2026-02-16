"""Print a sample of the dataset for manual inspection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leak_detection.pipeline import preview_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the simulated dataset")
    parser.add_argument("-n", "--rows", type=int, default=5, help="Rows to display")
    args = parser.parse_args()

    sample = preview_dataset(rows=args.rows)
    print(sample.to_markdown(index=False))


if __name__ == "__main__":
    main()
