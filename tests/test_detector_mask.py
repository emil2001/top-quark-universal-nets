"""Detector-derived encoder padding mask.

The encoder padding mask must come from a single inference-able source — the
detector inputs (an object is present iff its feature slot is non-empty) — and
be applied identically in training and evaluation for both the DETR and GATr
architectures. It must never depend on parton-level regression targets (which
are unavailable at inference and leak truth into the encoder).
"""

import importlib.util
import types
from pathlib import Path

import torch

from pat_embedder.data import detector_enc_padding_mask, detector_slot_presence
from pat_embedder.models import REG_DIM_V5, PatEmbedderModelV3
from pat_embedder.models_gatr import PatEmbedderModelVXFull

from .conftest import BATCH_SIZE, MAX_JETS, MAX_LEPTONS, MAX_TOPS, MAX_WS, NUM_HLR_VARS
from .test_models import _make_4mom_from_raw

INPUT_DIM = 2 + MAX_LEPTONS * 8 + MAX_JETS * 8


def _load_infer_module():
    """Import scripts/infer_and_plot.py (not a package) to reach the eval wrappers."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "infer_and_plot.py"
    spec = importlib.util.spec_from_file_location("infer_and_plot_for_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── The detector-mask source ────────────────────────────────────────────────


class TestDetectorMaskSource:
    def test_padded_slot_is_masked_present_slot_is_not(self):
        torch.manual_seed(0)
        x = torch.randn(2, INPUT_DIM)
        # Event 0: absent lepton slot 2 and absent jet slot 5 (collate zeroes pads).
        x[0, 2 + 2 * 8:2 + 3 * 8] = 0.0
        x[0, 2 + MAX_LEPTONS * 8 + 5 * 8:2 + MAX_LEPTONS * 8 + 6 * 8] = 0.0

        lep_present, jet_present = detector_slot_presence(x, MAX_LEPTONS, MAX_JETS)
        assert not lep_present[0, 2]
        assert not jet_present[0, 5]
        # Every other slot has random (non-zero) features → present.
        assert lep_present[0, [0, 1, 3]].all()
        assert jet_present[0, [i for i in range(MAX_JETS) if i != 5]].all()
        assert lep_present[1].all() and jet_present[1].all()

        mask = detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS)
        assert mask[0, 1 + 2]  # lepton slot 2 → padding
        assert mask[0, 1 + MAX_LEPTONS + 5]  # jet slot 5 → padding
        assert not mask[1].any()  # event 1: nothing masked

    def test_met_and_query_tokens_never_masked(self):
        torch.manual_seed(1)
        # All-zero detector slots are the worst case for "always present" tokens.
        x = torch.zeros(3, INPUT_DIM)
        n_query = MAX_TOPS + MAX_WS
        mask = detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS, n_query=n_query)
        assert mask.shape == (3, 1 + MAX_LEPTONS + MAX_JETS + n_query)
        assert not mask[:, 0].any()  # MET token always present
        assert not mask[:, 1 + MAX_LEPTONS + MAX_JETS:].any()  # query tokens always present
        # With everything zero, all detector slots are padding.
        assert mask[:, 1:1 + MAX_LEPTONS + MAX_JETS].all()

    def test_shapes(self):
        x = torch.randn(4, INPUT_DIM)
        assert detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS).shape == (
            4, 1 + MAX_LEPTONS + MAX_JETS,
        )
        assert detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS, n_query=8).shape == (
            4, 1 + MAX_LEPTONS + MAX_JETS + 8,
        )

    def test_presence_ignores_categorical_feature(self):
        torch.manual_seed(2)
        x = torch.randn(2, INPUT_DIM)
        # Padding zeroes the seven kinematic features, but z-scoring turns its
        # categorical zero into a non-zero value.  The latter must not revive
        # the slot.
        x[0, 2:2 + 7] = 0.0
        x[0, 2 + 7] = -2.5
        lep_present, _ = detector_slot_presence(x, MAX_LEPTONS, MAX_JETS)
        lep = x[:, 2:2 + MAX_LEPTONS * 8].reshape(2, MAX_LEPTONS, 8)
        assert not lep_present[0, 0]
        assert lep_present[0, 1:].all()
        assert lep_present[1].all()
        assert torch.equal(lep_present, (lep[..., :7] != 0.0).any(dim=-1))


# ── No parton-truth leakage: mask depends on detector inputs only ────────────


class TestNoPartonTruthLeakage:
    def test_encoder_mask_does_not_mirror_parton_truth(self, synthetic_batch):
        feats = synthetic_batch["features"].float()
        masks = synthetic_batch["masks"]
        det_jet_pad = detector_enc_padding_mask(feats, MAX_LEPTONS, MAX_JETS)[
            :, 1 + MAX_LEPTONS:1 + MAX_LEPTONS + MAX_JETS
        ]
        parton_jet_pad = ~masks["jets"].bool()
        # The old encoder mask equalled the parton-truth pad; the detector mask
        # must diverge from it (real detector jets without a parton target are
        # now visible to the encoder).
        assert (det_jet_pad != parton_jet_pad).any(), (
            "encoder mask still mirrors parton-truth target validity"
        )

    def test_mask_unchanged_when_targets_change(self, synthetic_batch):
        """Mutating the regression targets must not change the encoder mask."""
        feats = synthetic_batch["features"].float()
        m1 = detector_enc_padding_mask(feats, MAX_LEPTONS, MAX_JETS)
        # Detector inputs are unchanged; targets are irrelevant to the source.
        feats_same = feats.clone()
        m2 = detector_enc_padding_mask(feats_same, MAX_LEPTONS, MAX_JETS)
        assert torch.equal(m1, m2)

    def test_fixture_masks_exactly_kinematically_empty_detector_slots(self, synthetic_batch):
        feats = synthetic_batch["features"].float()
        batch_size = feats.size(0)
        leptons = feats[:, 2:2 + MAX_LEPTONS * 8].reshape(batch_size, MAX_LEPTONS, 8)
        jets = feats[:, 2 + MAX_LEPTONS * 8:].reshape(batch_size, MAX_JETS, 8)
        expected = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool),
                ~(leptons[..., :7] != 0.0).any(dim=-1),
                ~(jets[..., :7] != 0.0).any(dim=-1),
            ],
            dim=1,
        )
        mask = detector_enc_padding_mask(feats, MAX_LEPTONS, MAX_JETS)
        assert expected[:, 1:1 + MAX_LEPTONS].any()
        assert expected[:, 1 + MAX_LEPTONS:].any()
        assert torch.equal(mask, expected)


# ── train == eval at the model level (in-model fallback == explicit mask) ────


class TestTrainEvalConsistency:
    def test_v3_detr_fallback_equals_explicit_mask(self, synthetic_batch):
        x = synthetic_batch["features"].float()
        model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        ).eval()
        mask = detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS)
        with torch.no_grad():
            out_explicit = model(x, enc_padding_mask=mask)
            out_fallback = model(x)  # enc_padding_mask=None → detector fallback
        for a, b in zip(out_explicit[:5], out_fallback[:5]):
            assert torch.allclose(a, b, atol=1e-6)


    def test_vxfull_runs_with_detector_mask(self, fixture_data):
        x = fixture_data["features"].float()
        feat_mean = torch.from_numpy(fixture_data["feat_mean"]).float().unsqueeze(0)
        feat_std = torch.from_numpy(fixture_data["feat_std"]).float().unsqueeze(0)
        x_raw = x * feat_std + feat_mean
        x_4mom = _make_4mom_from_raw(x, feat_mean, feat_std)
        model = PatEmbedderModelVXFull(
            ga_channels=8, ga_heads=2, n_enc_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        ).eval()
        mask = detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS)
        with torch.no_grad():
            out = model(x, x_4mom, x_raw, enc_padding_mask=mask)
        assert out[3].shape == (BATCH_SIZE, MAX_TOPS, REG_DIM_V5)
        assert not torch.isnan(out[3]).any()


# ── Evaluation wrappers: accepts_masks threaded, parton masks ignored ────────


class TestEvalWrappers:
    def test_detr_wrapper_threads_accepts_masks_and_ignores_parton(self, synthetic_batch):
        infer = _load_infer_module()
        x = synthetic_batch["features"].float()
        model = PatEmbedderModelV3(
            embed_dim=64, n_heads=2, n_enc_layers=2, n_dec_layers=2,
            num_hlr_vars=NUM_HLR_VARS, reg_dim=REG_DIM_V5,
        ).eval()
        pl_stub = types.SimpleNamespace(model=model)
        wrapper = infer.DetrEvalWrapper(pl_stub, torch.device("cpu"))
        assert wrapper.accepts_masks is True

        garbage = {
            "leptons": torch.zeros(x.size(0), MAX_LEPTONS, dtype=torch.bool),
            "jets": torch.zeros(x.size(0), MAX_JETS, dtype=torch.bool),
        }
        out_none = wrapper(x)
        out_garbage = wrapper(x, masks=garbage)
        # Parton masks must be ignored — the wrapper builds the detector mask.
        assert torch.allclose(out_none[3], out_garbage[3], atol=1e-6)

        # The wrapper output matches the raw model run with the explicit detector mask.
        mask = detector_enc_padding_mask(x, MAX_LEPTONS, MAX_JETS)
        with torch.no_grad():
            out_ref = model(x, enc_padding_mask=mask)
        assert torch.allclose(out_none[3], out_ref[3], atol=1e-6)

    def test_gatr_wrapper_threads_accepts_masks_and_ignores_parton(
        self, fixture_data, reg_means, reg_stds
    ):
        from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull

        infer = _load_infer_module()
        x = fixture_data["features"].float()
        pl = PatEmbedderLightningVXFull(
            feat_mean=fixture_data["feat_mean"],
            feat_std=fixture_data["feat_std"],
            ga_channels=8, ga_heads=2, n_enc_layers=1,
            max_leptons=MAX_LEPTONS, max_jets=MAX_JETS,
            max_tops=MAX_TOPS, max_ws=MAX_WS,
            num_hlr_vars=NUM_HLR_VARS, reg_means=reg_means, reg_stds=reg_stds,
        ).eval()
        wrapper = infer.GATrEvalWrapper(pl, torch.device("cpu"))
        assert wrapper.accepts_masks is True

        garbage = {
            "leptons": torch.zeros(x.size(0), MAX_LEPTONS, dtype=torch.bool),
            "jets": torch.zeros(x.size(0), MAX_JETS, dtype=torch.bool),
        }
        out_none = wrapper(x)
        out_garbage = wrapper(x, masks=garbage)
        # The previously-broken masks=None fallback and parton masks must all
        # resolve to the same detector-derived mask.
        assert torch.allclose(out_none[3], out_garbage[3], atol=1e-6)
        assert not torch.isnan(out_none[3]).any()
