#!/usr/bin/env python3
"""Rebuild a contract-v2 cache from cached raw-equivalent targets and write provenance."""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pat_embedder.data import rebuild_cache, write_cache_provenance  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-cache", type=Path, required=True)
    return parser.parse_args()


def _code_version():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _ws_slot3_eta_std(path):
    with path.open("rb") as cache_file:
        cache = pickle.load(cache_file)
    return float(cache["reg_stds"]["ws"][3, 1])


def main():
    args = _parse_args()
    output = args.output.resolve()
    previous_cache = args.previous_cache.resolve()
    provenance_path = Path(f"{output}.provenance.json")
    if output.exists() or provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite cache or provenance at {output}")
    if not previous_cache.is_file():
        raise FileNotFoundError(f"Previous cache does not exist: {previous_cache}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {output.parent}")

    old_ws_slot3_eta_std = _ws_slot3_eta_std(previous_cache)
    cache_path, build_metadata = rebuild_cache(str(previous_cache), str(output))
    with open(cache_path, "rb") as cache_file:
        new_cache = pickle.load(cache_file)
    new_ws_slot3_eta_std = float(new_cache["reg_stds"]["ws"][3, 1])
    provenance = write_cache_provenance(
        cache_path,
        code_version=_code_version(),
        previous_cache_path=str(previous_cache),
        differences=(
            "Rebuilt train/test targets and their normalization statistics together from raw targets.",
            "Discarded events with non-finite or abs(target) above 1e6 regression entries.",
            "Replaces the legacy reg_stats_fix sidecar path with cache contract v2.",
            "Uses the code version containing the signed V5 energy contract.",
        ),
        build_metadata={
            **build_metadata,
            "ws_slot3_eta_std_before": old_ws_slot3_eta_std,
            "ws_slot3_eta_std_after": new_ws_slot3_eta_std,
        },
    )
    print(json.dumps({"cache": cache_path, "provenance": provenance}, sort_keys=True))


if __name__ == "__main__":
    main()
