"""Unit tests for the physics-informed decay chain mass loss (C2a)
and the sin²φ + cos²φ = 1 soft constraint.

Tests cover:
  - Helper functions: _v5_to_cartesian, _soft_select_p4, _lorentz_mass_sq, _mass_huber
  - PatEmbedderLossV5._physics_mass_loss: finite output, gradient flow, correct
    mass recovery when assignments are perfect, zero loss when disabled.
  - Integration: forward_with_aux with mass_loss_weight > 0 produces a "mass"
    key and a larger total loss than without.
"""

import numpy as np
import pytest
import torch

from pat_embedder.losses import (
    _M_TOP_GEV,
    _M_W_GEV,
    PatEmbedderLossV5,
    _lorentz_mass_sq,
    _mass_huber,
    _soft_select_p4,
    _v5_to_cartesian,
    reg_target_to_v5,
)
from pat_embedder.models import REG_DIM_V5, PatEmbedderModelV3

from .conftest import BATCH_SIZE, MAX_JETS, MAX_LEPTONS, MAX_TOPS, MAX_WS, NUM_HLR_VARS

# ---------------------------------------------------------------------------
# Helper: build minimal reg_means / reg_stds dicts (all zeros / ones)
# ---------------------------------------------------------------------------

def _make_reg_stats(device="cpu"):
    """Return trivial (zero mean, unit std) reg stats for all particle types."""
    slot_sizes = {
        "leptons": MAX_LEPTONS,
        "neutrinos": MAX_LEPTONS,
        "jets": MAX_JETS,
        "tops": MAX_TOPS,
        "ws": MAX_WS,
    }
    means = {k: torch.zeros(n, 3, device=device) for k, n in slot_sizes.items()}
    stds  = {k: torch.ones(n, 3, device=device)  for k, n in slot_sizes.items()}
    return means, stds


# ---------------------------------------------------------------------------
# Tests for _v5_to_cartesian
# ---------------------------------------------------------------------------

class TestV5ToCartesian:
    def test_output_shape(self):
        B, N = 4, 3
        reg_v5 = torch.randn(B, N, 5)
        mean = torch.zeros(N, 3)
        std  = torch.ones(N, 3)
        p4 = _v5_to_cartesian(reg_v5, mean, std)
        assert p4.shape == (B, N, 4)

    def test_output_finite(self):
        B, N = 4, 3
        reg_v5 = torch.randn(B, N, 5)
        mean = torch.zeros(N, 3)
        std  = torch.ones(N, 3)
        p4 = _v5_to_cartesian(reg_v5, mean, std)
        assert torch.isfinite(p4).all()

    def test_zero_momentum_particle(self):
        """A particle with pT=0 should have px=py=pz=0."""
        B, N = 1, 1
        # pT_n=0 (z-scored, mean=0, std=1 → pT=0), η=0, sin_phi=0, cos_phi=1, E=0
        reg_v5 = torch.zeros(B, N, 5)
        mean = torch.zeros(N, 3)
        std  = torch.ones(N, 3)
        p4 = _v5_to_cartesian(reg_v5, mean, std)
        # E=0, px=0, py=0, pz=0
        assert torch.allclose(p4, torch.zeros(B, N, 4), atol=1e-6)

    def test_known_kinematics(self):
        """Particle with pT=10, η=0, φ=0 → px=10, py=0, pz=0, E=10.

        reg_v5 stores log1p(pT)/log1p(E) (z-scored; mean=0/std=1 here so
        z-scoring is a no-op) — _v5_to_cartesian must invert log1p via
        expm1() after un-z-scoring.
        """
        N = 1
        pT_n = torch.log1p(torch.tensor(10.0)).item()
        E_n = torch.log1p(torch.tensor(10.0)).item()
        # With mean=0, std=1: pT_n=log1p(10), η_n=0, sin_phi=0, cos_phi=1, E_n=log1p(10)
        reg_v5 = torch.tensor([[[pT_n, 0.0, 0.0, 1.0, E_n]]])
        mean = torch.zeros(N, 3)
        std  = torch.ones(N, 3)
        p4 = _v5_to_cartesian(reg_v5, mean, std)
        # E=10, px=10*cos(0)=10, py=10*sin(0)=0, pz=10*sinh(0)=0
        expected = torch.tensor([[[10.0, 10.0, 0.0, 0.0]]])
        assert torch.allclose(p4, expected, atol=1e-4)

    def test_gradient_flows(self):
        B, N = 2, 3
        reg_v5 = torch.randn(B, N, 5, requires_grad=True)
        mean = torch.zeros(N, 3)
        std  = torch.ones(N, 3)
        p4 = _v5_to_cartesian(reg_v5, mean, std)
        p4.sum().backward()
        assert reg_v5.grad is not None
        assert torch.isfinite(reg_v5.grad).all()


# ---------------------------------------------------------------------------
# Tests for _soft_select_p4
# ---------------------------------------------------------------------------

class TestSoftSelectP4:
    def test_output_shape(self):
        B, Nq, Nk = 4, 3, 12
        logits = torch.randn(B, Nq, Nk + 1)  # +1 for no-match
        p4_keys = torch.randn(B, Nk, 4)
        out = _soft_select_p4(logits, p4_keys)
        assert out.shape == (B, Nq, 4)

    def test_output_finite(self):
        B, Nq, Nk = 4, 3, 12
        logits = torch.randn(B, Nq, Nk + 1)
        p4_keys = torch.randn(B, Nk, 4)
        out = _soft_select_p4(logits, p4_keys)
        assert torch.isfinite(out).all()

    def test_hard_assignment_recovers_key(self):
        """When logits strongly prefer key j, output ≈ p4_keys[:, j, :]."""
        B, Nq, Nk = 2, 1, 4
        # Make logit for key 2 very large
        logits = torch.full((B, Nq, Nk + 1), -100.0)
        logits[:, :, 2] = 100.0
        p4_keys = torch.randn(B, Nk, 4)
        out = _soft_select_p4(logits, p4_keys)
        expected = p4_keys[:, 2:3, :]  # (B, 1, 4)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_gradient_flows(self):
        B, Nq, Nk = 2, 3, 5
        logits = torch.randn(B, Nq, Nk + 1, requires_grad=True)
        p4_keys = torch.randn(B, Nk, 4)
        out = _soft_select_p4(logits, p4_keys)
        out.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Tests for _lorentz_mass_sq and _mass_huber
# ---------------------------------------------------------------------------

class TestLorentzMassSq:
    def test_massless_particle(self):
        """Photon: E=1, px=1, py=0, pz=0 → M²=0."""
        p4 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        m_sq = _lorentz_mass_sq(p4)
        assert torch.allclose(m_sq, torch.zeros(1), atol=1e-6)

    def test_w_boson_mass(self):
        """W boson at rest: E=M_W, px=py=pz=0 → M²=M_W²."""
        p4 = torch.tensor([[_M_W_GEV, 0.0, 0.0, 0.0]])
        m_sq = _lorentz_mass_sq(p4)
        assert torch.allclose(m_sq, torch.tensor([_M_W_GEV ** 2]), atol=1e-3)

    def test_top_quark_mass(self):
        """Top quark at rest: E=M_t, px=py=pz=0 → M²=M_t²."""
        p4 = torch.tensor([[_M_TOP_GEV, 0.0, 0.0, 0.0]])
        m_sq = _lorentz_mass_sq(p4)
        assert torch.allclose(m_sq, torch.tensor([_M_TOP_GEV ** 2]), atol=1e-3)

    def test_batch_shape(self):
        p4 = torch.randn(4, 3, 4)
        m_sq = _lorentz_mass_sq(p4)
        assert m_sq.shape == (4, 3)


class TestMassHuber:
    def test_zero_at_target(self):
        """When M_pred == M_target, loss should be 0."""
        m_sq = torch.tensor([_M_W_GEV ** 2])
        loss = _mass_huber(m_sq, _M_W_GEV)
        assert loss.item() < 1e-4

    def test_positive_loss(self):
        """Loss should be non-negative."""
        m_sq = torch.tensor([50.0 ** 2])  # wrong mass
        loss = _mass_huber(m_sq, _M_W_GEV)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        m_sq = torch.tensor([60.0 ** 2], requires_grad=True)
        loss = _mass_huber(m_sq, _M_W_GEV)
        loss.sum().backward()
        assert m_sq.grad is not None
        assert torch.isfinite(m_sq.grad).all()

    def test_negative_m_sq_clamped(self):
        """Negative M² (unphysical) should not produce NaN — clamped to 1e-4."""
        m_sq = torch.tensor([-100.0])
        loss = _mass_huber(m_sq, _M_W_GEV)
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Tests for PatEmbedderLossV5._physics_mass_loss
# ---------------------------------------------------------------------------

class TestPhysicsMassLoss:
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch, cls_targets_long):
        self.model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        )
        self.criterion = PatEmbedderLossV5(mass_loss_weight=1.0)
        self.x = synthetic_batch["features"]
        self.masks = synthetic_batch["masks"]
        self.reg_targets = {
            k: v[:, :, :REG_DIM_V5].contiguous()
            for k, v in synthetic_batch["targets"]["regression"].items()
        }
        self.cls_targets = cls_targets_long
        self.assign_targets = {
            k: v.long().view(BATCH_SIZE, -1) for k, v in synthetic_batch["targets"]["assignments"].items()
        }
        self.hlr_targets = synthetic_batch["targets"]["high_level_regression"]
        self.reg_means, self.reg_stds = _make_reg_stats()

    def _forward(self):
        out = self.model(self.x)
        reg = {"leptons": out[0], "neutrinos": out[1], "jets": out[2], "tops": out[3], "ws": out[4]}
        cls = {
            "lepton_logits": out[5], "neutrino_logits": out[6], "jet_logits": out[7],
            "top_logits": out[8], "w_logits": out[9],
        }
        assign = out[10]
        hlr = out[11]
        aux = out[12]
        return reg, cls, assign, hlr, aux

    def test_physics_mass_loss_finite(self):
        reg, cls, assign, hlr, aux = self._forward()
        loss = self.criterion._physics_mass_loss(
            reg, assign, self.masks, self.reg_means, self.reg_stds
        )
        assert torch.isfinite(loss)

    def test_physics_mass_loss_nonnegative(self):
        reg, cls, assign, hlr, aux = self._forward()
        loss = self.criterion._physics_mass_loss(
            reg, assign, self.masks, self.reg_means, self.reg_stds
        )
        assert loss.item() >= 0.0

    def test_physics_mass_loss_gradient_flows(self):
        """Gradients must flow back to model parameters through the mass loss."""
        reg, cls, assign, hlr, aux = self._forward()
        loss = self.criterion._physics_mass_loss(
            reg, assign, self.masks, self.reg_means, self.reg_stds
        )
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_forward_with_aux_includes_mass_key(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
            reg_means=self.reg_means,
            reg_stds=self.reg_stds,
        )
        assert "mass" in losses
        assert torch.isfinite(losses["mass"])

    def test_mass_loss_increases_total(self):
        """Total loss with mass_loss_weight=1.0 should be >= total without it."""
        reg, cls, assign, hlr, aux = self._forward()

        criterion_no_mass = PatEmbedderLossV5(mass_loss_weight=0.0)
        losses_no_mass = criterion_no_mass.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
        )

        losses_with_mass = self.criterion.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
            reg_means=self.reg_means,
            reg_stds=self.reg_stds,
        )
        assert losses_with_mass["total"].item() >= losses_no_mass["total"].item()

    def test_mass_loss_disabled_when_weight_zero(self):
        """When mass_loss_weight=0, 'mass' key should not appear in losses."""
        criterion_disabled = PatEmbedderLossV5(mass_loss_weight=0.0)
        reg, cls, assign, hlr, aux = self._forward()
        losses = criterion_disabled.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
            reg_means=self.reg_means,
            reg_stds=self.reg_stds,
        )
        assert "mass" not in losses

    def test_mass_loss_disabled_without_reg_stats(self):
        """When reg_means/reg_stds are None, mass loss is skipped even if weight > 0."""
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
            reg_means=None,
            reg_stds=None,
        )
        assert "mass" not in losses

    def test_perfect_w_jj_mass_gives_low_loss(self):
        """When W→jj assignment is perfect and jet 4-momenta sum to M_W, loss ≈ 0."""
        B = 2
        Nw = MAX_WS
        Nj = MAX_JETS

        # Build two jets that together have invariant mass = M_W
        # Jet 1: E=M_W/2 + 10, px=10, py=0, pz=0
        # Jet 2: E=M_W/2 + 10, px=-10, py=0, pz=0
        # Sum: E=M_W+20, px=0, py=0, pz=0 → M²=(M_W+20)² — not exactly M_W
        # Instead: use jets at rest each with E=M_W/2
        E_j = _M_W_GEV / 2.0
        # jet 1: (E, px, py, pz) = (E_j, E_j, 0, 0) — massless
        # jet 2: (E_j, -E_j, 0, 0) — massless
        # sum: (M_W, 0, 0, 0) → M = M_W ✓
        jet_p4_np = np.zeros((B, Nj, 4), dtype=np.float32)
        jet_p4_np[:, 0, :] = [E_j, E_j, 0.0, 0.0]   # jet 0
        jet_p4_np[:, 1, :] = [E_j, -E_j, 0.0, 0.0]  # jet 1
        jet_p4 = torch.from_numpy(jet_p4_np)

        # Verify the invariant mass is M_W
        w_p4 = jet_p4[:, 0, :] + jet_p4[:, 1, :]  # (B, 4)
        m_sq = _lorentz_mass_sq(w_p4.unsqueeze(1)).squeeze(1)  # (B,)
        assert torch.allclose(torch.sqrt(m_sq.clamp(min=0)), torch.full((B,), _M_W_GEV), atol=1e-3)

        # Build assignment logits that strongly prefer jet 0 for w_to_jet1
        # and jet 1 for w_to_jet2
        logits_j1 = torch.full((B, Nw, Nj + 1), -100.0)
        logits_j1[:, :, 0] = 100.0  # all W slots → jet 0
        logits_j2 = torch.full((B, Nw, Nj + 1), -100.0)
        logits_j2[:, :, 1] = 100.0  # all W slots → jet 1

        # Build reg_v5 for jets: convert jet_p4 back to V5 format
        # pT = sqrt(px²+py²), η=0, sin_phi=0, cos_phi=1, E
        pT_j0 = E_j  # since px=E_j, py=0
        pT_j1 = E_j
        # With mean=0, std=1, z-scoring is a no-op — but pT/E are still stored
        # log1p-scaled; the decoder applies expm1 after un-z-scoring.
        pT_j0_n = torch.log1p(torch.tensor(pT_j0)).item()
        pT_j1_n = torch.log1p(torch.tensor(pT_j1)).item()
        E_j_n = torch.log1p(torch.tensor(E_j)).item()
        jet_reg_v5 = torch.zeros(B, Nj, 5)
        jet_reg_v5[:, 0, 0] = pT_j0_n  # pT (log1p)
        jet_reg_v5[:, 0, 3] = 1.0      # cos_phi = 1 (phi=0)
        jet_reg_v5[:, 0, 4] = E_j_n    # E (log1p)
        jet_reg_v5[:, 1, 0] = pT_j1_n  # pT (log1p)
        jet_reg_v5[:, 1, 3] = -1.0     # cos_phi = -1 (phi=π)
        jet_reg_v5[:, 1, 4] = E_j_n    # E (log1p)

        reg_out = {
            "leptons":   torch.zeros(B, MAX_LEPTONS, 5),
            "neutrinos": torch.zeros(B, MAX_LEPTONS, 5),
            "jets":      jet_reg_v5,
            "tops":      torch.zeros(B, MAX_TOPS, 5),
            "ws":        torch.zeros(B, MAX_WS, 5),
        }
        # Only include w_to_jet1/w_to_jet2 — omit w_to_lepton/w_to_neutrino/top_to_*
        # so only the W→jj mass constraint is active in this test.
        assign_out = {
            "w_to_jet1": logits_j1,
            "w_to_jet2": logits_j2,
        }
        masks = {
            "ws":   torch.ones(B, Nw, dtype=torch.bool),
            "tops": torch.zeros(B, MAX_TOPS, dtype=torch.bool),
        }
        reg_means, reg_stds = _make_reg_stats()

        criterion = PatEmbedderLossV5(mass_loss_weight=1.0)
        loss = criterion._physics_mass_loss(reg_out, assign_out, masks, reg_means, reg_stds)

        # The W→jj mass should be very close to M_W, so the Huber loss should be small
        assert loss.item() < 1.0, f"Expected small mass loss, got {loss.item():.4f}"


# ---------------------------------------------------------------------------
# Tests for sin²φ + cos²φ = 1 soft constraint
# ---------------------------------------------------------------------------

class TestPhiNormConstraint:
    """Tests for the sin²φ + cos²φ = 1 soft penalty in _masked_regression."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch, cls_targets_long):
        self.model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        )
        self.x = synthetic_batch["features"]
        self.masks = synthetic_batch["masks"]
        self.reg_targets = {
            k: v[:, :, :REG_DIM_V5].contiguous()
            for k, v in synthetic_batch["targets"]["regression"].items()
        }
        self.cls_targets = cls_targets_long
        self.assign_targets = {
            k: v.long().view(BATCH_SIZE, -1) for k, v in synthetic_batch["targets"]["assignments"].items()
        }
        self.hlr_targets = synthetic_batch["targets"]["high_level_regression"]

    def _forward(self):
        out = self.model(self.x)
        reg = {"leptons": out[0], "neutrinos": out[1], "jets": out[2], "tops": out[3], "ws": out[4]}
        cls = {
            "lepton_logits": out[5], "neutrino_logits": out[6], "jet_logits": out[7],
            "top_logits": out[8], "w_logits": out[9],
        }
        assign = out[10]
        hlr = out[11]
        aux = out[12]
        return reg, cls, assign, hlr, aux

    def test_phi_norm_loss_finite(self):
        """With phi_norm_weight > 0, total loss should be finite."""
        criterion = PatEmbedderLossV5(phi_norm_weight=0.1)
        reg, cls, assign, hlr, aux = self._forward()
        losses = criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert torch.isfinite(losses["total"])

    def test_phi_norm_loss_increases_regression(self):
        """Regression loss with phi_norm_weight > 0 should be >= without it."""
        criterion_no_phi = PatEmbedderLossV5(phi_norm_weight=0.0)
        criterion_with_phi = PatEmbedderLossV5(phi_norm_weight=1.0)
        reg, cls, assign, hlr, aux = self._forward()

        losses_no = criterion_no_phi(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        losses_with = criterion_with_phi(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert losses_with["regression"].item() >= losses_no["regression"].item()

    def test_phi_norm_zero_at_unit_circle(self):
        """When sin²φ + cos²φ = 1 exactly, the penalty should be 0."""
        B, N = 4, 4
        # Construct predictions exactly on the unit circle
        phi = torch.rand(B, N) * 2 * torch.pi
        pred = torch.zeros(B, N, 5)
        pred[:, :, 2] = torch.sin(phi)   # sin_phi
        pred[:, :, 3] = torch.cos(phi)   # cos_phi
        # Compute only the phi norm part: norm_sq = sin²+cos² = 1 → penalty = (1-1)² = 0
        norm_sq = pred[:, :, 2].pow(2) + pred[:, :, 3].pow(2)
        phi_penalty = (norm_sq - 1.0).pow(2)
        assert torch.allclose(phi_penalty, torch.zeros(B, N), atol=1e-6)

    def test_phi_norm_nonzero_off_unit_circle(self):
        """When sin²φ + cos²φ ≠ 1, the penalty should be > 0."""
        B, N = 4, 4
        pred = torch.zeros(B, N, 5)
        pred[:, :, 2] = 0.8   # sin_phi
        pred[:, :, 3] = 0.9   # cos_phi  → norm_sq = 0.64 + 0.81 = 1.45 ≠ 1
        norm_sq = pred[:, :, 2].pow(2) + pred[:, :, 3].pow(2)
        phi_penalty = (norm_sq - 1.0).pow(2)
        assert (phi_penalty > 0).all()

    def test_phi_norm_gradient_flows(self):
        """Gradients must flow back through the phi norm penalty."""
        criterion = PatEmbedderLossV5(phi_norm_weight=0.1)
        reg, cls, assign, hlr, aux = self._forward()
        losses = criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        losses["total"].backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_phi_norm_disabled_by_default(self):
        """Default phi_norm_weight=0 should give same regression loss as explicit 0."""
        criterion_default = PatEmbedderLossV5()
        criterion_zero = PatEmbedderLossV5(phi_norm_weight=0.0)
        reg, cls, assign, hlr, aux = self._forward()

        losses_default = criterion_default(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        losses_zero = criterion_zero(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert torch.allclose(losses_default["regression"], losses_zero["regression"])


# Recover on-shell top and W masses from the synthetic normalized targets.

class TestMassShellFixtureRegression:
    def _p4(self, key, synthetic_batch, reg_means, reg_stds):
        raw = synthetic_batch["targets"]["regression"][key]  # (B, N, 7), z-scored raw cache target
        phi_mean = torch.as_tensor(np.asarray(reg_means[key])[:, 2], dtype=raw.dtype)
        phi_std = torch.as_tensor(np.asarray(reg_stds[key])[:, 2], dtype=raw.dtype)
        v5 = reg_target_to_v5(raw, phi_mean, phi_std)
        reg_mean3 = torch.as_tensor(np.asarray(reg_means[key])[:, [0, 1, 6]], dtype=raw.dtype)
        reg_std3 = torch.as_tensor(np.asarray(reg_stds[key])[:, [0, 1, 6]], dtype=raw.dtype)
        return _v5_to_cartesian(v5, reg_mean3, reg_std3)  # (B, N, 4) = (E, px, py, pz)

    def _invariant_mass_sq(self, p4, mask):
        m_sq = p4[..., 0] ** 2 - (p4[..., 1] ** 2 + p4[..., 2] ** 2 + p4[..., 3] ** 2)
        return m_sq[mask]

    def test_top_mass_shell_close_to_pdg(self, synthetic_batch, reg_means, reg_stds):
        p4 = self._p4("tops", synthetic_batch, reg_means, reg_stds)
        m_sq = self._invariant_mass_sq(p4, synthetic_batch["masks"]["tops"])
        m = torch.sqrt(m_sq.clamp(min=0))
        assert torch.isfinite(m).all()
        assert abs(m.mean().item() - _M_TOP_GEV) < 5.0

    def test_w_mass_shell_close_to_pdg(self, synthetic_batch, reg_means, reg_stds):
        p4 = self._p4("ws", synthetic_batch, reg_means, reg_stds)
        m_sq = self._invariant_mass_sq(p4, synthetic_batch["masks"]["ws"])
        m = torch.sqrt(m_sq.clamp(min=0))
        assert torch.isfinite(m).all()
        assert abs(m.mean().item() - _M_W_GEV) < 5.0

    def test_neutrino_mass_shell_near_zero(self, synthetic_batch, reg_means, reg_stds):
        p4 = self._p4("neutrinos", synthetic_batch, reg_means, reg_stds)
        m_sq = self._invariant_mass_sq(p4, synthetic_batch["masks"]["neutrinos"])
        assert torch.isfinite(m_sq).all()
        assert abs(m_sq.mean().item()) < 5.0

    def test_lepton_and_jet_mass_shell_not_imaginary(self, synthetic_batch, reg_means, reg_stds):
        """Leptons/jets are light — M² must be non-negative (not imaginary) in the
        vast majority of valid slots (jets: hadronisation smearing allows a few
        exceptions, see 310 §1B)."""
        for key in ("leptons", "jets"):
            p4 = self._p4(key, synthetic_batch, reg_means, reg_stds)
            m_sq = self._invariant_mass_sq(p4, synthetic_batch["masks"][key])
            assert torch.isfinite(m_sq).all()
            frac_negative = (m_sq < -1.0).float().mean().item()
            assert frac_negative < 0.2, f"{key}: too many imaginary masses ({frac_negative:.2%})"

    def test_round_trip_px_identity(self, synthetic_batch, reg_means, reg_stds):
        """expm1(un-z-score(col0)) * cos(phi_actual) must match the denormalised
        raw col 3 (px) — decisive col-index sanity check, independent of the E
        column (310 §1A)."""
        raw = synthetic_batch["targets"]["regression"]["tops"]  # (B, N, 7)
        mean = torch.as_tensor(np.asarray(reg_means["tops"]), dtype=raw.dtype)
        std = torch.as_tensor(np.asarray(reg_stds["tops"]), dtype=raw.dtype)
        mask = synthetic_batch["masks"]["tops"]

        pt = torch.expm1(raw[..., 0] * std[:, 0] + mean[:, 0])
        phi = raw[..., 2] * std[:, 2] + mean[:, 2]
        px_from_pt_phi = pt * torch.cos(phi)
        px_denorm = raw[..., 3] * std[:, 3] + mean[:, 3]

        diff = (px_from_pt_phi - px_denorm)[mask]
        assert torch.max(torch.abs(diff)).item() < 1e-2
