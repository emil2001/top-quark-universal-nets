"""Loss function tests on synthetic data: finite values, gradients, backward pass."""

import pytest
import torch

from pat_embedder.losses import PatEmbedderLoss, PatEmbedderLossV5, focal_ce_loss
from pat_embedder.models import REG_DIM_V5, PatEmbedderModel, PatEmbedderModelV3

from .conftest import BATCH_SIZE, NUM_HLR_VARS


def _run_v1_model(model, x):
    out = model(x)
    reg = {"leptons": out[0], "neutrinos": out[1], "jets": out[2], "tops": out[3], "ws": out[4]}
    cls = {
        "lepton_logits": out[5], "neutrino_logits": out[6], "jet_logits": out[7],
        "top_logits": out[8], "w_logits": out[9],
    }
    return reg, cls, out[10], out[11]


class TestFocalCELoss:
    def test_returns_finite(self):
        logits = torch.randn(16, 7)
        targets = torch.randint(0, 7, (16,))
        loss = focal_ce_loss(logits, targets)
        assert loss.shape == (16,)
        assert torch.isfinite(loss).all()

    def test_reduction_mean(self):
        logits = torch.randn(16, 7)
        targets = torch.randint(0, 7, (16,))
        loss = focal_ce_loss(logits, targets, reduction="mean")
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))
        loss = focal_ce_loss(logits, targets, reduction="mean")
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestLossV1:
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch, cls_targets_long):
        self.model = PatEmbedderModel(num_hlr_vars=NUM_HLR_VARS)
        self.criterion = PatEmbedderLoss()
        self.x = synthetic_batch["features"]
        self.masks = synthetic_batch["masks"]
        self.reg_targets = {
            k: v.view(BATCH_SIZE, -1, 7) for k, v in synthetic_batch["targets"]["regression"].items()
        }
        self.cls_targets = cls_targets_long
        self.assign_targets = {
            k: v.long().view(BATCH_SIZE, -1) for k, v in synthetic_batch["targets"]["assignments"].items()
        }
        self.hlr_targets = synthetic_batch["targets"]["high_level_regression"]

    def test_total_finite(self):
        reg, cls, assign, hlr = _run_v1_model(self.model, self.x)
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert torch.isfinite(losses["total"])

    def test_all_subtasks_present(self):
        reg, cls, assign, hlr = _run_v1_model(self.model, self.x)
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        for key in ("regression", "classification", "assignment", "hlr", "total"):
            assert key in losses

    def test_backward(self):
        reg, cls, assign, hlr = _run_v1_model(self.model, self.x)
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        losses["total"].backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestLossV5:
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch, cls_targets_long):
        self.model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        )
        self.criterion = PatEmbedderLossV5()
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

    def test_total_finite(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert torch.isfinite(losses["total"])

    def test_presence_loss(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
        )
        assert "top_presence_bce" in losses
        assert "w_presence_bce" in losses

    def test_aux_forward(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
        )
        assert "aux" in losses
        assert torch.isfinite(losses["total"])

    def test_forward_accepts_aux_kwargs(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
        )
        assert "aux" in losses
        assert torch.isfinite(losses["total"])

    def test_backward(self):
        reg, cls, assign, hlr, aux = self._forward()
        losses = self.criterion.forward_with_aux(
            reg, cls, self.reg_targets, self.cls_targets, self.masks,
            assign, self.assign_targets, hlr, self.hlr_targets,
            aux_head_outputs=aux,
        )
        losses["total"].backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


