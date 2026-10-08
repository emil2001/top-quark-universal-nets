"""Mass-shell evaluation follows signed-energy decoding for LGaTr contract v3."""

import importlib.util
from pathlib import Path

import torch

from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull


def _load_mass_shell_sanity():
    """Import scripts/mass_shell_sanity.py (not a package)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "mass_shell_sanity.py"
    spec = importlib.util.spec_from_file_location("mass_shell_sanity_for_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mass_shell_energy_decode_follows_the_model_contract(
    synthetic_batch, fixture_data, reg_means, reg_stds
):
    sanity = _load_mass_shell_sanity()
    device = torch.device("cpu")
    module = PatEmbedderLightningVXFull(
        feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
        embed_dim=16, ga_channels=4, ga_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    ).eval()

    signed = sanity.compute_mass_shell(module, "vx-full", [synthetic_batch], device, None)
    assert signed["signed_energy"] is True
    for particle in ("tops", "ws"):
        assert signed[particle]["n_slots"] > 0
        assert signed[particle]["n_nonfinite_m_sq"] == 0

    # A pre-contract (v2) model keeps the legacy unsigned decode — the flag has
    # to be read off the model, never assumed by the script.
    module.model._vx_full_contract_version.fill_(2)
    legacy = sanity.compute_mass_shell(module, "vx-full", [synthetic_batch], device, None)
    assert legacy["signed_energy"] is False
