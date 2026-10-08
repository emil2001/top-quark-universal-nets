"""Focused tests for the GATr physical p4 / V5 representation contract."""

from types import SimpleNamespace

import torch

from pat_embedder.data import detector_enc_padding_mask
from pat_embedder.lightning_gatr import PatEmbedderLightningVXFull
from pat_embedder.losses import _v5_to_cartesian
from pat_embedder.models_gatr import _cartesian_to_v5


def _prepare_fixture_p4(fixture_data):
    wrapper = SimpleNamespace(
        device=torch.device("cpu"),
        feat_mean_buf=torch.as_tensor(fixture_data["feat_mean"]).unsqueeze(0),
        feat_std_buf=torch.as_tensor(fixture_data["feat_std"]).unsqueeze(0),
        model=SimpleNamespace(max_leptons=4, max_jets=12),
    )
    return PatEmbedderLightningVXFull._prepare_4mom(wrapper, {"features": fixture_data["features"]})


def test_prepare_4mom_restores_fixture_energy_and_met_pt(fixture_data):
    p4, x_raw = _prepare_fixture_p4(fixture_data)

    assert torch.allclose(p4[:, 0, 0], x_raw[:, 0].expm1(), rtol=1e-6, atol=1e-6)
    occupied = ~detector_enc_padding_mask(fixture_data["features"], 4, 12)[:, 1]
    assert torch.allclose(p4[occupied, 1, 0], x_raw[occupied, 2 + 6].expm1(), rtol=1e-6, atol=1e-6)
    assert torch.allclose(p4[occupied, 1, 1:], x_raw[occupied, 2 + 3:2 + 6], rtol=0, atol=0)
    features = fixture_data["features"]
    batch_size = features.size(0)
    leptons = features[:, 2:34].reshape(batch_size, 4, 8)
    jets = features[:, 34:].reshape(batch_size, 12, 8)
    expected_padding = torch.cat(
        [
            torch.zeros(batch_size, 1, dtype=torch.bool),
            ~(leptons[..., :7] != 0.0).any(dim=-1),
            ~(jets[..., :7] != 0.0).any(dim=-1),
        ],
        dim=1,
    )
    padding = detector_enc_padding_mask(fixture_data["features"], 4, 12)
    assert 0 < expected_padding.sum().item() < expected_padding.numel()
    assert torch.equal(padding, expected_padding)
    assert torch.equal(p4[expected_padding], torch.zeros_like(p4[expected_padding]))


def test_physical_p4_v5_roundtrip_and_zero_vectors_are_finite():
    p4 = torch.tensor(
        [[[10.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    v5 = _cartesian_to_v5(p4)
    recovered = _v5_to_cartesian(
        v5[:, :2], torch.zeros(2, 3), torch.ones(2, 3), signed_energy=True
    )

    assert torch.isfinite(v5).all()
    assert torch.isfinite(recovered).all()
    assert torch.allclose(recovered[0, 0], p4[0, 0], rtol=1e-5, atol=2e-5)
    assert torch.allclose(recovered[0, 1], torch.zeros(4), atol=2e-5)
    assert torch.allclose(v5[0, 2, 4], -torch.log1p(torch.tensor(3.0)))

    v5.sum().backward()
    assert torch.isfinite(p4.grad).all()
    assert p4.grad[0, 2, 0].abs() > 0


def test_physical_p4_v5_preserves_energy_sign_in_roundtrip_and_gradient():
    energies = torch.tensor([-100.0, 100.0], requires_grad=True)
    p4 = torch.stack(
        [
            energies,
            torch.tensor([30.0, 30.0]),
            torch.tensor([40.0, 40.0]),
            torch.tensor([0.0, 0.0]),
        ],
        dim=-1,
    ).unsqueeze(0)

    v5 = _cartesian_to_v5(p4)
    expected_energy = torch.sign(energies) * torch.log1p(energies.abs())
    torch.testing.assert_close(v5[0, :, 4], expected_energy)
    assert v5[0, 0, 4] < 0 < v5[0, 1, 4]

    recovered = _v5_to_cartesian(
        v5, torch.zeros(2, 3), torch.ones(2, 3), signed_energy=True
    )
    torch.testing.assert_close(recovered[0, :, 0], energies)

    positive_target = torch.log1p(torch.tensor(100.0))
    torch.square(v5[0, 0, 4] - positive_target).backward()
    assert energies.grad[0] < 0


def test_signed_energy_encoding_uses_unit_subgradient_at_zero():
    energy = torch.tensor(0.0, requires_grad=True)
    p4 = torch.stack(
        [
            energy,
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ]
    ).reshape(1, 1, 4)

    _cartesian_to_v5(p4)[0, 0, 4].backward()

    torch.testing.assert_close(energy.grad, torch.tensor(1.0))


def test_low_pt_bfloat16_conversion_keeps_direction():
    p4 = torch.tensor([[[1.0, 1e-3, 2e-3, 0.0]]], dtype=torch.bfloat16)
    v5 = _cartesian_to_v5(p4)
    assert v5.dtype == torch.float32
    assert torch.allclose(v5[..., 2], torch.tensor([[2.0 / 5.0**0.5]]), atol=2e-3)
    assert torch.allclose(v5[..., 3], torch.tensor([[1.0 / 5.0**0.5]]), atol=2e-3)


def test_vx_full_contract_marker_rejects_legacy_checkpoint(fixture_data, reg_means, reg_stds):
    module = PatEmbedderLightningVXFull(
        feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
        embed_dim=16, ga_channels=4, ga_heads=1, n_enc_layers=1, n_dec_layers=1,
        reg_means=reg_means, reg_stds=reg_stds,
    )
    current_state = module.state_dict()
    assert int(current_state["_gatr_output_contract_version"]) == 3
    assert int(current_state["model._vx_full_contract_version"]) == 3
    module.on_load_checkpoint({"state_dict": current_state})

    legacy_state = current_state.copy()
    legacy_state["_gatr_output_contract_version"] = torch.tensor(2)
    try:
        module.on_load_checkpoint({"state_dict": legacy_state})
    except RuntimeError as err:
        assert "signed-energy output contract v3" in str(err)
    else:
        raise AssertionError("legacy checkpoint was silently accepted")
