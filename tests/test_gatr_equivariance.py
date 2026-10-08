"""Lorentz-equivariance tests for the retained geometric algebra layers."""

import numpy as np
import pytest
import torch

from pat_embedder.gatr import BlockConfig, GATrBlock, apply_rotor, random_rotor
from pat_embedder.gatr.algebra import GRADE1_INDICES


def _apply_rotor_to_batch(mv: torch.Tensor, rotor: np.ndarray) -> torch.Tensor:
    """Apply Lorentz rotor to a batch of multivectors (B, T, C, 16)."""
    r = torch.tensor(rotor, dtype=mv.dtype, device=mv.device)
    # Sandwich each channel independently: mv_out[b,t,c] = R mv[b,t,c] R~
    B, T, C, _ = mv.shape
    mv_flat = mv.reshape(B * T * C, 16)
    r_exp = r.unsqueeze(0).expand(B * T * C, -1)
    out = torch.tensor(
        apply_rotor(mv_flat.numpy(), r_exp.numpy()), dtype=mv.dtype
    )
    return out.reshape(B, T, C, 16)


class TestGATrBlockEquivariance:
    """A single GATrBlock must satisfy block(R x R~) ≈ R block(x) R~."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        self.rotor = random_rotor(boost_max=0.4, rotation_max=1.0, rng=rng)

        cfg = BlockConfig(channels=8, n_heads=2, gp_mid_channels=4, attn_channels=4)
        self.block = GATrBlock(cfg)
        self.block.eval()

        # Random input multivectors: (2, 5, 8, 16)
        self.x = torch.randn(2, 5, 8, 16) * 0.3

    def test_grade1_equivariance(self):
        """Grade-1 output components transform as 4-vectors under the rotor."""
        x_boosted = _apply_rotor_to_batch(self.x, self.rotor)

        with torch.no_grad():
            out_orig   = self.block(self.x)
            out_boosted = self.block(x_boosted)

        # Expected boosted output
        out_orig_boosted = _apply_rotor_to_batch(out_orig, self.rotor)

        # Compare grade-1 components (indices 1..4)
        g1 = list(GRADE1_INDICES)
        diff = (out_boosted[..., g1] - out_orig_boosted[..., g1]).abs().max().item()
        # Float32 accumulation over 5 tokens × 8 channels; tolerance O(1e-5)
        assert diff < 1e-4, f"Equivariance violation in grade-1: max diff = {diff:.2e}"


