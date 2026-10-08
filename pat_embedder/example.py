"""Build a demonstrator cache from the eight supplied simulated events."""

from pathlib import Path

import numpy as np

from .data import collate_fn, compute_stats, make_collate_and_scale, save_cache


def make_example_data(sample_path):
    """Return six training and two test events with train-only statistics.

    Empty dimensions have neutral statistics; singly observed dimensions use
    unit standard deviation. These conventions are for the miniature demo.
    """
    with np.load(sample_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays["features"].shape != (8, 130):
        raise ValueError("Expected the supplied eight-event sample with 130 input features")
    samples = []
    for i in range(8):
        targets = {}
        for kind in ("regression", "classification", "assignments"):
            prefix = kind + "_"
            targets[kind] = {key[len(prefix):]: value[i] for key, value in arrays.items()
                             if key.startswith(prefix)}
        targets["high_level_regression"] = arrays["high_level_regression"][i]
        samples.append({"features": arrays["features"][i], "targets": targets})
    raw_train = collate_fn(samples[:6])
    with np.errstate(invalid="ignore", divide="ignore"):
        statistics = list(compute_stats([raw_train]))

    def complete(mean, std, observations):
        count = np.isfinite(observations).sum(axis=0)
        return np.where(count == 0, 0.0, mean).astype(np.float32), np.where(
            count < 2, 1.0, std
        ).astype(np.float32)

    statistics[0], statistics[1] = complete(*statistics[:2], raw_train["features"])
    for key, observations in raw_train["targets"]["regression"].items():
        statistics[2][key], statistics[3][key] = complete(
            statistics[2][key], statistics[3][key], observations
        )
    statistics[4], statistics[5] = complete(
        *statistics[4:], raw_train["targets"]["high_level_regression"]
    )
    collate = make_collate_and_scale(*statistics)
    return {
        "train_batches": [collate(samples[:3]), collate(samples[3:6])],
        "test_batches": [collate(samples[6:])],
        **dict(zip(
            ["feat_mean", "feat_std", "reg_means", "reg_stds", "hlr_mean", "hlr_std"],
            statistics,
        )),
    }


def save_example_cache(sample_path, output):
    """Write a contract-v2 cache from the bundled miniature sample."""
    output = Path(output)
    return save_cache(
        **make_example_data(sample_path),
        cache_dir=str(output.parent), filename=output.name,
        build_metadata={"kind": "miniature_research_sample", "train_events": 6, "test_events": 2},
    )
