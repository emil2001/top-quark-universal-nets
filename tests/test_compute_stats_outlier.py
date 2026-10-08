"""Normalization statistics exclude non-finite and corrupted outlier values."""

import numpy as np
import pytest
import torch

from pat_embedder.data import compute_stats


def _make_batches(values, n_batches=4, batch_size=8):
    """values: 1D array of length n_batches*batch_size for a single
    (particle=1, column=1) regression target "ws"; features are unrelated
    small values, present in every event."""
    batches = []
    for i in range(n_batches):
        chunk = values[i * batch_size:(i + 1) * batch_size]
        batches.append({
            "features": torch.zeros(batch_size, 3),
            "targets": {
                "regression": {
                    "ws": torch.from_numpy(chunk.reshape(batch_size, 1, 1)).float(),
                },
                "classification": {},
            },
        })
    return batches


class TestComputeStatsOutlierGuard:
    def test_clean_data_unaffected(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(loc=0.0, scale=1.8, size=32).astype(np.float64)
        loader = _make_batches(clean)

        _, _, reg_means, reg_stds, _, _ = compute_stats(loader)

        assert reg_stds["ws"][0, 0] == pytest.approx(clean.std(), abs=0.05)

    def test_rare_huge_outlier_does_not_dominate_std(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(loc=0.0, scale=1.8, size=31).astype(np.float64)
        poisoned = np.concatenate([clean, [1e11]])  # 1 garbage value among 32
        loader = _make_batches(poisoned)

        _, _, reg_means, reg_stds, _, _ = compute_stats(loader)

        std = reg_stds["ws"][0, 0]
        assert std < 10.0, f"outlier not filtered: std={std}"
        assert std == pytest.approx(clean.std(), abs=0.3)

    def test_negative_huge_outlier_also_filtered(self):
        rng = np.random.default_rng(1)
        clean = rng.normal(loc=0.0, scale=1.8, size=31).astype(np.float64)
        poisoned = np.concatenate([clean, [-9.99e10]])
        loader = _make_batches(poisoned)

        _, _, reg_means, reg_stds, _, _ = compute_stats(loader)

        assert reg_stds["ws"][0, 0] < 10.0
