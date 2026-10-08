"""Regression plots support selecting a single variable."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_embedder.evaluation import plot_regression_distribution_all_particles
from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from infer_and_plot import GATrEvalWrapper  # noqa: E402


def test_single_variable_does_not_crash(synthetic_batch, fixture_data, reg_means, reg_stds):
    device = torch.device("cpu")
    module = PatEmbedderLightningVXFull(
        feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
        embed_dim=16, ga_channels=4, ga_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    ).eval()
    model = GATrEvalWrapper(module, device)

    try:
        pred, tgt, mask = plot_regression_distribution_all_particles(
            model, [synthetic_batch], device, reg_means, reg_stds, "tops",
            variables_to_plot=["pt"],
        )
    finally:
        plt.close("all")

    assert pred.shape[-1] == tgt.shape[-1]
    assert mask.any()


def test_histogram_of_neighboring_float32_predictions(synthetic_batch, monkeypatch):
    """Two nearly identical predictions must still produce finite histogram bins."""
    # NumPy 2 raises "Too many bins for data range" on these float32 values.
    values = torch.tensor([-1.0, np.nextafter(np.float32(-1), np.float32(0))])

    class Model(torch.nn.Module):
        max_tops = 4

        def forward(self, inputs):
            pred = torch.zeros(inputs.shape[0], 4, 7)
            pred[:, :, 0] = values.repeat(inputs.shape[0] // 2)[:, None]
            return (pred,) * 5

    batch = {**synthetic_batch, "targets": {**synthetic_batch["targets"]}}
    batch["targets"]["regression"] = {
        **synthetic_batch["targets"]["regression"],
        "tops": torch.zeros_like(synthetic_batch["targets"]["regression"]["tops"]),
    }
    stats = {"tops": np.ones((4, 7), dtype=np.float32)}
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    pred, _, _ = plot_regression_distribution_all_particles(
        Model(), [batch], torch.device("cpu"), stats, stats, "tops",
        variables_to_plot=["pt"], denormalize=False,
    )
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
