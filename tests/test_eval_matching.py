"""Matching and physical decoding preserve slot-specific normalization."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pat_embedder.evaluation import (
    _DEFAULT_DIMENSION_NAMES,
    _DIMENSION_NAMES_BY_RD,
    _get_decoder_perms,
    _uses_signed_v5_energy,
)
from pat_embedder.losses import _M_TOP_GEV, reg_target_to_v5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from infer_and_plot import (  # noqa: E402
    DetrEvalWrapper,
    GATrEvalWrapper,
    _denorm_v5_regression,
)


def _make_outputs(top_reg, w_reg, top_cls_logits, w_cls_logits):
    """Assemble the model-output tuple consumed by ``_get_decoder_perms``.

    Only indices 3/4 (top/W regression) and 8/9 (top/W classification logits)
    are read by the matching code.
    """
    outs = [None] * 11
    outs[3] = top_reg
    outs[4] = w_reg
    outs[8] = top_cls_logits
    outs[9] = w_cls_logits
    return outs


# --------------------------------------------------------------------------- #
# F1 — eval matching uses the train (sin/cos) representation
# --------------------------------------------------------------------------- #

def test_reg_target_to_v5_recovers_sincos():
    """The shared helper recovers the physical angle from z-scored phi and emits
    ``(pT_n, eta_n, sin_phi, cos_phi, E_n)`` — the documented train target."""
    # raw 7-dim cache layout: (pT_n, eta_n, phi_n, px_n, py_n, pz_n, E_n).
    raw = torch.tensor([[[0.3, -0.7, 1.0, 9.0, 0.0, 0.0, 1.2]]])  # (1, 1, 7)
    phi_mean = torch.tensor([0.5])
    phi_std = torch.tensor([2.0])
    out = reg_target_to_v5(raw, phi_mean, phi_std)
    phi_actual = 1.0 * 2.0 + 0.5  # recover physical angle before sin/cos
    assert out.shape[-1] == 5
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(0.3))   # pT_n untouched
    torch.testing.assert_close(out[0, 0, 1], torch.tensor(-0.7))  # eta_n untouched
    torch.testing.assert_close(out[0, 0, 2], torch.sin(torch.tensor(phi_actual)))
    torch.testing.assert_close(out[0, 0, 3], torch.cos(torch.tensor(phi_actual)))
    torch.testing.assert_close(out[0, 0, 4], torch.tensor(1.2))   # E_n (col 6) -> dim 4


def _swap_inducing_case():
    """One event engineered so the buggy raw-truncation matching prefers a slot
    swap while the train-consistent V5 matching is identity.

    7-dim cache layout (pT, eta, phi, E, px, py, pz); identity stats (mean 0,
    std 1) so phi_n == phi_actual.  Both slots: pT=eta=phi=0.  The garbage cost
    term (pred energy vs cache px) drives the swap:
      slot0: E=0,  px=10 ;  slot1: E=10, px=0.
    """
    raw = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ]])  # (1, 2, 7)
    # Perfect V5 prediction = train target representation (pT,eta,sin,cos,E).
    perfect = torch.tensor([[
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 10.0],
    ]])  # (1, 2, 5)
    cls_logits = torch.zeros(1, 2, 3)
    cls_logits[..., 1] = 1.0  # argmax -> class idx 1 (matches real top/W targets)
    top_pid = torch.tensor([[6.0, 6.0]])    # _top_map: 6 -> 1
    w_pid = torch.tensor([[24.0, 24.0]])    # _w_map: 24 -> 1
    targets = {
        "regression": {"tops": raw.clone(), "ws": raw.clone()},
        "classification": {"tops": top_pid, "ws": w_pid},
    }
    masks_raw = {
        "tops": torch.ones(1, 2, dtype=torch.bool),
        "ws": torch.ones(1, 2, dtype=torch.bool),
    }
    outputs = _make_outputs(perfect.clone(), perfect.clone(), cls_logits.clone(), cls_logits.clone())
    return targets, masks_raw, outputs


def _identity_stats(n=2):
    z = np.zeros((n, 7), np.float32)
    o = np.ones((n, 7), np.float32)
    return {"tops": z, "ws": z.copy()}, {"tops": o, "ws": o.copy()}


def test_eval_matching_identity_for_perfect_pred_with_stats():
    """With phi stats the eval matching is computed in the train (sin/cos) space,
    so a perfect predictor pairs each slot to itself (identity perm)."""
    targets, masks_raw, outputs = _swap_inducing_case()
    reg_means, reg_stds = _identity_stats()
    _, top_perm, w_perm = _get_decoder_perms(
        None, torch.zeros(1, 1), targets, masks_raw, torch.device("cpu"),
        max_tops=2, max_ws=2, outputs=outputs,
        reg_means=reg_means, reg_stds=reg_stds,
    )
    identity = np.arange(2)[None, :]
    np.testing.assert_array_equal(top_perm, identity)
    np.testing.assert_array_equal(w_perm, identity)


def _query_normalization_case():
    """Two unequal query normalizations where source-slot matching is wrong.

    The prediction slots encode target slots in their receiving query
    normalizations.  A matcher that compares against source-normalized targets
    chooses identity; the physical pairwise cost must choose the swap.
    """
    raw = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    pred = torch.tensor([[
        [0.5, 0.0, 0.0, 1.0, 0.0],
        [9.5, 0.0, 0.0, 1.0, 0.0],
    ]])
    cls_logits = torch.zeros(1, 2, 3)
    cls_logits[..., 1] = 1.0
    pid = torch.tensor([[6.0, 6.0]])
    targets = {
        "regression": {"tops": raw.clone(), "ws": raw.clone()},
        "classification": {"tops": pid, "ws": torch.tensor([[24.0, 24.0]])},
    }
    masks_raw = {
        "tops": torch.ones(1, 2, dtype=torch.bool),
        "ws": torch.ones(1, 2, dtype=torch.bool),
    }
    means = np.zeros((2, 7), np.float32)
    means[1, 0] = -9.5
    stats = {"tops": means, "ws": means.copy()}
    stds = {"tops": np.ones((2, 7), np.float32), "ws": np.ones((2, 7), np.float32)}
    outputs = _make_outputs(pred.clone(), pred.clone(), cls_logits, cls_logits)
    return targets, masks_raw, outputs, stats, stds


def test_eval_matching_renormalizes_targets_for_receiving_query():
    """Eval matcher reproduces the training query-normalized assignment."""
    targets, masks_raw, outputs, reg_means, reg_stds = _query_normalization_case()
    _, top_perm, w_perm = _get_decoder_perms(
        None, torch.zeros(1, 1), targets, masks_raw, torch.device("cpu"),
        max_tops=2, max_ws=2, outputs=outputs,
        reg_means=reg_means, reg_stds=reg_stds,
    )
    expected = np.array([[1, 0]])
    np.testing.assert_array_equal(top_perm, expected)
    np.testing.assert_array_equal(w_perm, expected)


def test_eval_matching_rejects_missing_stats():
    """No evaluation path may silently fall back to identity normalization."""
    targets, masks_raw, outputs = _swap_inducing_case()
    with pytest.raises(ValueError, match="reg_means and reg_stds are required"):
        _get_decoder_perms(
            None, torch.zeros(1, 1), targets, masks_raw, torch.device("cpu"),
            max_tops=2, max_ws=2, outputs=outputs,
            reg_means=None, reg_stds=None,
        )


# --------------------------------------------------------------------------- #
# F2 — V5 energy denorm uses energy stats, not px stats
# --------------------------------------------------------------------------- #

def test_denorm_v5_energy_uses_energy_stats_not_px():
    """``_denorm_v5_regression`` must scale the predicted energy (pred dim 4) by
    the energy stats (raw cache column 6), not the px stats (column 3), and must
    invert log1p (``expm1``) after un-z-scoring — see
    the regression contract."""
    # raw 7-dim cache layout: (pT_n, eta_n, phi_n, px_n, py_n, pz_n, E_n).
    # Distinct px (col 3) vs energy (col 6) stats so the bug is observable.
    reg_means = {"tops": np.array([[0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 3.0]], np.float32)}
    reg_stds = {"tops": np.array([[1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0]], np.float32)}
    # pred (pT_n, eta_n, sin_phi, cos_phi, E_n); E_n = 1.0.
    pred = np.array([[[0.0, 0.0, 0.0, 1.0, 1.0]]], np.float32)  # (1, 1, 5)
    target = np.zeros((1, 1, 7), np.float32)
    pred_out, _ = _denorm_v5_regression(pred, target, reg_means, reg_stds, "tops")
    # Energy is output dim 3.
    expected = float(np.expm1(1.0 * 1.0 + 3.0))   # energy stats (col 6) + expm1 -> ~53.6
    wrong = 1.0 * 20.0 + 50.0                       # old bug: px stats (col 3), no expm1 -> 70.0
    np.testing.assert_allclose(pred_out[0, 0, 3], expected, rtol=1e-6)
    assert not np.isclose(pred_out[0, 0, 3], wrong)


def test_denorm_v5_signed_energy_preserves_negative_prediction():
    reg_means = {"tops": np.zeros((1, 7), np.float32)}
    reg_stds = {"tops": np.ones((1, 7), np.float32)}
    pred = np.array([[[0.0, 0.0, 0.0, 1.0, -np.log1p(100.0)]]], np.float32)
    target = np.zeros((1, 1, 7), np.float32)

    pred_out, _ = _denorm_v5_regression(
        pred, target, reg_means, reg_stds, "tops", signed_energy=True
    )

    np.testing.assert_allclose(pred_out[0, 0, 3], -100.0, rtol=1e-6)


def test_signed_energy_routing_is_limited_to_vx_full_contract_v3(
    fixture_data, reg_means, reg_stds
):
    """Every object the eval pipeline holds a model through must route alike.

    ``infer_and_plot.py::_eval_model`` hands the helpers an eval wrapper, not
    the LightningModule, and the two keep the inner ``nn.Module`` under
    different attribute names (``._model`` vs ``.model``). A stub mirroring one
    of those names cannot fail when the lookup misses the other one, which is
    how a lookup that disabled signed decoding on both repository eval paths
    passed the suite (the regression contract B1, B2).
    """
    from types import SimpleNamespace

    from pat_embedder.lightning import PatEmbedderLightningV5
    from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull

    device = torch.device("cpu")
    vx_full = PatEmbedderLightningVXFull(
        feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
        embed_dim=16, ga_channels=4, ga_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    ).eval()

    # infer_and_plot.py:1008 and evaluation.py:260 see the wrapper.
    assert _uses_signed_v5_energy(GATrEvalWrapper(vx_full, device)) is True
    # mass_shell_sanity.py sees the LightningModule; v1/v2/v3 eval the bare module.
    assert _uses_signed_v5_energy(vx_full) is True
    assert _uses_signed_v5_energy(vx_full.model) is True

    # V5 (DETR) carries no contract buffer — it keeps the legacy unsigned decode.
    v5 = PatEmbedderLightningV5(
        embed_dim=16, n_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    ).eval()

    assert _uses_signed_v5_energy(DetrEvalWrapper(v5, device)) is False
    assert _uses_signed_v5_energy(v5) is False
    assert _uses_signed_v5_energy(SimpleNamespace()) is False


# --------------------------------------------------------------------------- #
# MLP regression (seven coordinates) recovers physical energy from col 6
# --------------------------------------------------------------------------- #

def test_denorm_v5_regression_rd7_generic_branch_mass_shell(synthetic_batch, reg_means, reg_stds):
    """V1 (rd==7): ``_denorm_v5_regression`` must apply ``expm1`` to pT/E and read
    energy from raw col 6 (not col 3/px) — the previously-unfixed V1 eval path
   ."""
    raw = synthetic_batch["targets"]["regression"]["tops"]  # (B, N, 7), z-scored raw cache target
    pred_out, target_out = _denorm_v5_regression(
        raw.numpy(), raw.numpy(), reg_means, reg_stds, "tops"
    )
    assert pred_out.shape[-1] == 4
    np.testing.assert_allclose(pred_out, target_out, rtol=1e-5)  # perfect predictor

    mask = synthetic_batch["masks"]["tops"].numpy()
    pt, eta, phi, energy = (pred_out[:, :, i] for i in range(4))
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    m_sq = energy ** 2 - (px ** 2 + py ** 2 + pz ** 2)
    m = np.sqrt(np.clip(m_sq[mask], 0, None))
    assert np.isfinite(m).all()
    assert abs(m.mean() - _M_TOP_GEV) < 5.0


def test_dimension_names_by_rd_matches_actual_column_order():
    """Plot labels match the supported models' denormalized output columns."""
    assert _DIMENSION_NAMES_BY_RD[5] == ['pt', 'eta', 'phi', 'energy']
    assert _DIMENSION_NAMES_BY_RD[7] == ['pt', 'eta', 'phi', 'px', 'py', 'pz', 'energy']
    assert 6 not in _DIMENSION_NAMES_BY_RD
    assert _DEFAULT_DIMENSION_NAMES == ['pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz']
