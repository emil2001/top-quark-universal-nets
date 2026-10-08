#!/usr/bin/env python
"""Prepare the bundled eight-event sample for training and inference."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pat_embedder.example import save_example_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/example.pkl"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new path")
    save_example_cache(REPO_ROOT / "examples/mini_events.npz", args.output)


if __name__ == "__main__":
    main()
