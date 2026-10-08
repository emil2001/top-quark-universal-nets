"""Model forward-pass tests on deterministic synthetic events."""

import pytest
import pytorch_lightning as pl
import torch

from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull
from pat_embedder.models import REG_DIM_V5, PatEmbedderModel, PatEmbedderModelV3
from pat_embedder.models_gatr import PatEmbedderModelVXFull

from .conftest import BATCH_SIZE, MAX_JETS, MAX_LEPTONS, MAX_TOPS, MAX_WS, NUM_HLR_VARS

ASSIGN_KEYS = {"top_to_w", "top_to_bjet", "w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2"}


class TestPatEmbedderV1:
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch):
        self.x = synthetic_batch["features"]
        self.model = PatEmbedderModel(num_hlr_vars=NUM_HLR_VARS)

    def test_output_count(self):
        out = self.model(self.x)
        assert len(out) == 12

    def test_regression_shapes(self):
        out = self.model(self.x)
        assert out[0].shape == (BATCH_SIZE, MAX_LEPTONS, 7)
        assert out[1].shape == (BATCH_SIZE, MAX_LEPTONS, 7)
        assert out[2].shape == (BATCH_SIZE, MAX_JETS, 7)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, 7)
        assert out[4].shape == (BATCH_SIZE, MAX_TOPS, 7)

    def test_classification_shapes(self):
        out = self.model(self.x)
        assert out[5].shape == (BATCH_SIZE, MAX_LEPTONS, 7)
        assert out[7].shape == (BATCH_SIZE, MAX_JETS, 3)
        assert out[8].shape == (BATCH_SIZE, MAX_TOPS, 3)

    def test_assignment_keys(self):
        out = self.model(self.x)
        assert set(out[10].keys()) == ASSIGN_KEYS

    def test_hlr_shape(self):
        out = self.model(self.x)
        assert out[11].shape == (BATCH_SIZE, NUM_HLR_VARS)

    def test_no_nan(self):
        out = self.model(self.x)
        for t in out[:10]:
            assert not torch.isnan(t).any()


class TestPatEmbedderV3:
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_batch):
        self.x = synthetic_batch["features"]
        self.model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        )

    def test_output_count(self):
        out = self.model(self.x)
        assert len(out) == 14

    def test_regression_dim_v5(self):
        out = self.model(self.x)
        assert out[0].shape == (BATCH_SIZE, MAX_LEPTONS, REG_DIM_V5)
        assert out[2].shape == (BATCH_SIZE, MAX_JETS, REG_DIM_V5)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)

    def test_presence_logits(self):
        out = self.model(self.x)
        assign = out[10]
        assert "top_presence" in assign
        assert "w_presence" in assign
        assert assign["top_presence"].shape == (BATCH_SIZE, MAX_TOPS)

    def test_aux_outputs(self):
        out = self.model(self.x)
        aux = out[12]
        assert isinstance(aux, list)
        assert len(aux) == 1  # n_dec_layers - 1

    def test_no_nan(self):
        out = self.model(self.x)
        for t in out[:10]:
            assert not torch.isnan(t).any()


def _make_4mom_from_raw(x, feat_mean, feat_std, max_leptons=4, max_jets=12):
    """Reconstruct (B, 17, 4) raw 4-momenta from z-scored features."""
    x_raw = x * feat_std + feat_mean
    B = x_raw.size(0)

    met_pt  = x_raw[:, 0]
    met_phi = x_raw[:, 1]
    met_4mom = torch.stack(
        [met_pt, met_pt * torch.cos(met_phi), met_pt * torch.sin(met_phi),
         torch.zeros_like(met_pt)], dim=-1
    ).unsqueeze(1)

    lep_raw = x_raw[:, 2:2 + max_leptons * 8].view(B, max_leptons, 8)
    lep_4mom = lep_raw[:, :, [6, 3, 4, 5]]

    jet_raw = x_raw[:, 2 + max_leptons * 8:].view(B, max_jets, 8)
    jet_4mom = jet_raw[:, :, [6, 3, 4, 5]]

    return torch.cat([met_4mom, lep_4mom, jet_4mom], dim=1)  # (B, 17, 4)


class TestPatEmbedderVXFull:
    @pytest.fixture(autouse=True)
    def setup(self, fixture_data):
        self.x = fixture_data["features"].float()
        feat_mean = torch.from_numpy(fixture_data["feat_mean"]).float().unsqueeze(0)
        feat_std  = torch.from_numpy(fixture_data["feat_std"]).float().unsqueeze(0)
        self.x_raw = self.x * feat_std + feat_mean
        self.x_4mom = _make_4mom_from_raw(self.x, feat_mean, feat_std)
        self.model = PatEmbedderModelVXFull(
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=2,
            num_hlr_vars=NUM_HLR_VARS,
            reg_dim=REG_DIM_V5,
        )

    def test_output_count(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        assert len(out) == 12

    def test_regression_shapes(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        assert out[0].shape == (BATCH_SIZE, MAX_LEPTONS, REG_DIM_V5)
        assert out[2].shape == (BATCH_SIZE, MAX_JETS, REG_DIM_V5)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)
        assert out[4].shape == (BATCH_SIZE, MAX_WS, REG_DIM_V5)

    def test_classification_shapes(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        assert out[5].shape == (BATCH_SIZE, MAX_LEPTONS, 7)
        assert out[7].shape == (BATCH_SIZE, MAX_JETS, 3)
        assert out[8].shape == (BATCH_SIZE, MAX_TOPS, 3)
        assert out[9].shape == (BATCH_SIZE, MAX_WS, 3)

    def test_assignment_shapes(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        assign = out[10]
        assert "top_presence" in assign
        assert "w_presence" in assign
        assert assign["top_presence"].shape == (BATCH_SIZE, MAX_TOPS)
        assert assign["w_presence"].shape == (BATCH_SIZE, MAX_WS)
        assert assign["top_to_w"].shape    == (BATCH_SIZE, MAX_TOPS, MAX_WS + 1)
        assert assign["top_to_bjet"].shape == (BATCH_SIZE, MAX_TOPS, MAX_JETS + 1)
        assert assign["w_to_lepton"].shape == (BATCH_SIZE, MAX_WS, MAX_LEPTONS + 1)

    def test_hlr_shape(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        assert out[11].shape == (BATCH_SIZE, NUM_HLR_VARS)

    def test_no_nan(self):
        out = self.model(self.x, self.x_4mom, self.x_raw)
        for t in out[:10]:
            assert not torch.isnan(t).any()

    def test_spurions_forward(self):
        model = PatEmbedderModelVXFull(
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=1,
            num_hlr_vars=NUM_HLR_VARS,
            reg_dim=REG_DIM_V5,
            spurions=True,
        )
        assert model.n_spurions == 2
        out = model(self.x, self.x_4mom, self.x_raw)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)
        assert out[10]["top_presence"].shape == (BATCH_SIZE, MAX_TOPS)

    @pytest.mark.parametrize("head_type", ["strict-equivariant", "hybrid-residual"])
    def test_parton_regression_head_variants(self, head_type):
        model = PatEmbedderModelVXFull(
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=1,
            num_hlr_vars=NUM_HLR_VARS,
            reg_dim=REG_DIM_V5,
            parton_reg_head_type=head_type,
            parton_reg_channels=3,
            parton_reg_residual_scale=0.2,
        )
        out = model(self.x, self.x_4mom, self.x_raw)
        assert len(out) == 12
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)
        assert out[4].shape == (BATCH_SIZE, MAX_WS, REG_DIM_V5)
        assert not torch.isnan(out[3]).any()
        assert not torch.isnan(out[4]).any()

    @pytest.mark.parametrize("head_type", ["strict-equivariant", "hybrid-residual"])
    def test_parton_regression_head_lightning_checkpoint_roundtrip(
        self, fixture_data, reg_means, reg_stds, tmp_path, head_type
    ):
        model = PatEmbedderLightningVXFull(
            feat_mean=fixture_data["feat_mean"],
            feat_std=fixture_data["feat_std"],
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=1,
            max_leptons=MAX_LEPTONS,
            max_jets=MAX_JETS,
            max_tops=MAX_TOPS,
            max_ws=MAX_WS,
            num_hlr_vars=NUM_HLR_VARS,
            reg_means=reg_means,
            reg_stds=reg_stds,
            parton_reg_head_type=head_type,
            parton_reg_channels=3,
            parton_reg_residual_scale=0.2,
        )
        ckpt_path = tmp_path / f"{head_type}.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            ckpt_path,
        )

        loaded = PatEmbedderLightningVXFull.load_from_checkpoint(
            ckpt_path,
            feat_mean=fixture_data["feat_mean"],
            feat_std=fixture_data["feat_std"],
            reg_means=reg_means,
            reg_stds=reg_stds,
            weights_only=False,
        )
        assert loaded.model.parton_reg_head_type == head_type
        out = loaded.model(self.x, self.x_4mom, self.x_raw)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)

    def test_fully_connected_subgroup_adds_parameters(self):
        base = PatEmbedderModelVXFull(
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=1,
            num_hlr_vars=0,
            reg_dim=REG_DIM_V5,
            fully_connected_subgroup=False,
        )
        fc = PatEmbedderModelVXFull(
            ga_channels=8,
            ga_heads=2,
            n_enc_layers=1,
            num_hlr_vars=0,
            reg_dim=REG_DIM_V5,
            fully_connected_subgroup=True,
        )
        assert sum(p.numel() for p in fc.parameters()) > sum(p.numel() for p in base.parameters())
