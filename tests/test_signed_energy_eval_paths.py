"""Signed-energy decoding reaches regression plots, quality metrics and mass-shell checks."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from pat_embedder.evaluation import plot_regression_distribution_all_particles
from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull

from .test_mass_shell_sanity import _load_mass_shell_sanity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from infer_and_plot import (  # noqa: E402
    GATrEvalWrapper,
    compute_regression_quality_metrics,
)

# The predicted energy under test: negative, one value per event so the
# published plots still have a histogram to draw. Both contracts un-z-score it
# to the same ``signed_log1p`` value and then disagree: v3 reads it as
# sign(x)*expm1(|x|) = -100 GeV and deeper, v2 as expm1(x) = -0.99 GeV.
_ENERGY_GEV_FIRST_EVENT = -100.0
# Momentum pinned alongside the energy where the test needs an exact mass.
_PT_GEV = 10.0

# Energy column of a denormalised prediction: (pt, eta, phi, energy).
_DENORM_ENERGY_DIM = 3


def _predicted_energy_gev(batch_size):
    """The negative energies pinned into the prediction, one per event."""
    return _ENERGY_GEV_FIRST_EVENT - np.arange(batch_size, dtype=np.float64)


def _legacy_energy_gev(energy_gev):
    """What the pre-v3 (unsigned log1p) decode makes of the same model output."""
    return np.expm1(-np.log1p(-np.asarray(energy_gev, dtype=np.float64)))


def _vx_full_module(fixture_data, reg_means, reg_stds):
    return PatEmbedderLightningVXFull(
        feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
        embed_dim=16, ga_channels=4, ga_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    ).eval()


def _pin_predicted_energy(module, reg_means, reg_stds, *, pt_gev=None):
    """Make the module predict :func:`_predicted_energy_gev` for tops/ws.

    A freshly initialised module predicts energies of an arbitrary sign, and the
    contracts differ only where the prediction is negative — so the event under
    test is built instead of hoped for. ``pt_gev`` additionally pins the
    momentum, for the mass-shell path where the decoded mass has to be exact.
    Only the predicted numbers are replaced: the module, the eval wrapper and
    the contract buffers the decode is keyed off stay the real ones.
    """
    inner = module.model
    forward = inner.forward
    # Raw cache columns are (pT, eta, phi, px, py, pz, E) — E is column 6, not 3
    # (=px); these are the statistics every eval path denormalises with, see
    # sin/cos phi are actual values, not z-scored.
    stats = {}
    for key, output_idx in (("tops", 3), ("ws", 4)):
        stats[output_idx] = (
            torch.as_tensor(np.asarray(reg_means[key]), dtype=torch.float32),
            torch.as_tensor(np.asarray(reg_stds[key]), dtype=torch.float32),
        )

    def _forward(*args, **kwargs):
        outputs = list(forward(*args, **kwargs))
        for output_idx, (mean, std) in stats.items():
            reg = outputs[output_idx].clone()
            energy = _predicted_energy_gev(reg.shape[0])
            signed_log = torch.as_tensor(-np.log1p(-energy), dtype=torch.float32).unsqueeze(-1)
            reg[..., 4] = ((signed_log - mean[:, 6]) / std[:, 6]).to(reg.device, reg.dtype)
            if pt_gev is not None:
                log_pt = float(np.log1p(pt_gev))
                reg[..., 0] = ((log_pt - mean[:, 0]) / std[:, 0]).to(reg.device, reg.dtype)
                reg[..., 1] = ((0.0 - mean[:, 1]) / std[:, 1]).to(reg.device, reg.dtype)
                reg[..., 2] = 0.0  # sin(phi)
                reg[..., 3] = 1.0  # cos(phi)
            outputs[output_idx] = reg
        return tuple(outputs)

    inner.forward = _forward


def _downgrade_to_contract_v2(module):
    module.model._vx_full_contract_version.fill_(2)


def test_regression_quality_energy_follows_the_checkpoint_contract(
    synthetic_batch, fixture_data, reg_means, reg_stds
):
    """``infer_and_plot.compute_regression_quality_metrics`` — the published
    ``metrics.json`` energy numbers (720 §6.1) must change with the contract."""
    device = torch.device("cpu")
    module = _vx_full_module(fixture_data, reg_means, reg_stds)
    _pin_predicted_energy(module, reg_means, reg_stds)
    model = GATrEvalWrapper(module, device)

    def _energy_metrics():
        return compute_regression_quality_metrics(
            model, [synthetic_batch], device, reg_means, reg_stds,
            particles=["tops"], variables=["energy"], max_samples=None,
        )["tops"]["energy"]

    v3 = _energy_metrics()
    _downgrade_to_contract_v2(module)
    v2 = _energy_metrics()

    # Every prediction sits below every (positive) target energy, so each MAE is
    # target + |decoded energy| and the gap between them is exactly the gap
    # between the two decodes of the same model output, averaged over the slots
    # the metric keeps (permuting them does not move an event's energy).
    mask = synthetic_batch["masks"]["tops"].numpy().astype(bool)
    signed = np.broadcast_to(_predicted_energy_gev(mask.shape[0])[:, None], mask.shape)
    expected_gap = float(np.mean((_legacy_energy_gev(signed) - signed)[mask]))
    assert v3["mae"] - v2["mae"] == pytest.approx(expected_gap, rel=1e-4)
    assert v3["n"] == v2["n"] == int(mask.sum())


def test_regression_distribution_energy_follows_the_checkpoint_contract(
    synthetic_batch, fixture_data, reg_means, reg_stds
):
    """``evaluation.plot_regression_distribution_all_particles`` — the energy
    histograms/scatter of the published plots must change with the contract."""
    device = torch.device("cpu")
    module = _vx_full_module(fixture_data, reg_means, reg_stds)
    _pin_predicted_energy(module, reg_means, reg_stds)
    model = GATrEvalWrapper(module, device)

    def _predicted_energy():
        try:
            pred, _, _ = plot_regression_distribution_all_particles(
                model, [synthetic_batch], device, reg_means, reg_stds, "tops",
            )
        finally:
            plt.close("all")
        return pred[:, :, _DENORM_ENERGY_DIM]

    v3 = _predicted_energy()
    _downgrade_to_contract_v2(module)
    v2 = _predicted_energy()

    expected = np.broadcast_to(_predicted_energy_gev(v3.shape[0])[:, None], v3.shape)
    np.testing.assert_allclose(v3, expected, rtol=1e-4)
    np.testing.assert_allclose(v2, _legacy_energy_gev(expected), rtol=1e-4)


def test_mass_shell_sanity_masses_follow_the_checkpoint_contract(
    synthetic_batch, fixture_data, reg_means, reg_stds
):
    """``scripts/mass_shell_sanity.py`` — the 810 mass-shell numbers (827 M4)
    must change with the contract, not just the flag echoed into the JSON."""
    sanity = _load_mass_shell_sanity()
    device = torch.device("cpu")
    module = _vx_full_module(fixture_data, reg_means, reg_stds)
    _pin_predicted_energy(module, reg_means, reg_stds, pt_gev=_PT_GEV)

    v3 = sanity.compute_mass_shell(module, "vx-full", [synthetic_batch], device, None)
    _downgrade_to_contract_v2(module)
    v2 = sanity.compute_mass_shell(module, "vx-full", [synthetic_batch], device, None)

    # M^2 = E^2 - |p|^2 on the pinned 4-momentum: 100 GeV of energy against
    # 10 GeV of momentum is on shell, 0.99 GeV of energy cannot be.
    assert v3["signed_energy"] is True and v2["signed_energy"] is False
    for particle in ("tops", "ws"):
        mask = synthetic_batch["masks"][particle].numpy().astype(bool)
        signed = np.broadcast_to(_predicted_energy_gev(mask.shape[0])[:, None], mask.shape)
        assert v3[particle]["n_slots"] == v2[particle]["n_slots"] == int(mask.sum())
        assert v3[particle]["frac_physical"] == 1.0
        assert v3[particle]["mass_median_gev"] == pytest.approx(
            float(np.median(np.sqrt(signed[mask] ** 2 - _PT_GEV ** 2))), rel=1e-4
        )
        assert v2[particle]["frac_physical"] == 0.0
        assert v2[particle]["neg_m_sq_median_gev2"] == pytest.approx(
            float(np.median(_legacy_energy_gev(signed[mask]) ** 2 - _PT_GEV ** 2)), rel=1e-4
        )
