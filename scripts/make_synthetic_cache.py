#!/usr/bin/env python
"""Create an illustrative cache without downloading any dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pat_embedder.config import SEED  # noqa: E402
from pat_embedder.synthetic import save_synthetic_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic.pkl"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new path")
    save_synthetic_cache(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
