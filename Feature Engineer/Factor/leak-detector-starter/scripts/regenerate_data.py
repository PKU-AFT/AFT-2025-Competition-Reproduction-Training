"""Regenerate the simulated factor dataset."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leak_detection.simulator import write_dataset


def main(target: str | None = None) -> Path:
    """Write the dataset to ``target`` (defaults to data/sample_factors.csv)."""

    path = write_dataset(Path(target) if target else None)
    print(f"Dataset saved to {path}")
    return path


if __name__ == "__main__":
    main()
