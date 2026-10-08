"""Regression tests for scalar-only VX-full parton queries."""

import math

import pytest
import torch

from pat_embedder.models_gatr import PatEmbedderModelVXFull


def _rotate_z(p4: torch.Tensor, angle: float) -> torch.Tensor:
    rotated = p4.clone()
    rotated[..., 1] = math.cos(angle) * p4[..., 1] - math.sin(angle) * p4[..., 2]
    rotated[..., 2] = math.sin(angle) * p4[..., 1] + math.cos(angle) * p4[..., 2]
    return rotated


@pytest.mark.parametrize("head_type", ["grade1", "strict-equivariant", "hybrid-residual"])
def test_parton_queries_stay_scalar_and_equivariant_after_optimizer_step(head_type):
    torch.manual_seed(5)
    model = PatEmbedderModelVXFull(
        ga_channels=4, ga_heads=1, n_enc_layers=1, dropout=0.0,
        parton_reg_head_type=head_type, parton_reg_channels=2,
    )
    model.eval()
    p4 = torch.randn(2, 17, 4) * 5
    p4[..., 0] = torch.sqrt(p4[..., 1:].square().sum(-1) + 1)
    features = torch.zeros(2, 130)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    outputs = model(features, p4, features)
    loss = outputs[3].square().mean() + outputs[4].square().mean()
    loss.backward()
    assert torch.isfinite(model.top_mv_q.grad).all()
    assert torch.isfinite(model.w_mv_q.grad).all()
    optimizer.step()

    assert model.top_mv_q.shape[-1] == 1
    assert model.w_mv_q.shape[-1] == 1
    assert torch.isfinite(model.top_mv_q).all()
    assert torch.isfinite(model.w_mv_q).all()

    angle = 0.7
    with torch.no_grad():
        original = model(features, p4, features)[3]
        rotated = model(features, _rotate_z(p4, angle), features)[3]

    expected_sin = math.cos(angle) * original[..., 2] + math.sin(angle) * original[..., 3]
    expected_cos = math.cos(angle) * original[..., 3] - math.sin(angle) * original[..., 2]
    assert torch.allclose(rotated[..., :2], original[..., :2], atol=3e-5, rtol=3e-5)
    assert torch.allclose(rotated[..., 4], original[..., 4], atol=3e-5, rtol=3e-5)
    assert torch.allclose(rotated[..., 2], expected_sin, atol=4e-5, rtol=4e-5)
    assert torch.allclose(rotated[..., 3], expected_cos, atol=4e-5, rtol=4e-5)
