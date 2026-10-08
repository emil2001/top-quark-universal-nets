"""Equivariant layers for GATr-lite (Cl(1,3) backbone).

All layers act on tensors of shape (B, T, C, 16), where the final axis indexes
the multivector basis. The implementation follows the optimized LGaTr study in
the related CMS/cms-nets implementation: cached algebra tensors, SDPA attention, equivariant RMS
normalization, grade dropout, and optional 10-basis SO+(1,3) linear maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import algebra as A

_CAYLEY_DENSE_CPU = torch.from_numpy(A.CAYLEY_DENSE)
_METRIC_GRADES_CPU = torch.from_numpy(A.METRIC_GRADES)
_INNER_PRODUCT_FACTORS_CPU = torch.from_numpy(A.INNER_PRODUCT_FACTORS)
_REVERSE_SIGNS_CPU = torch.from_numpy(A.REVERSE_SIGNS)
_DUAL_DENSE_CPU = torch.from_numpy(A.DUAL_DENSE)
_GRADE_PROJ_CPU = torch.zeros(5, 16)
_B2G_CPU = torch.empty(16, dtype=torch.long)
_SCALAR_MASK_CPU = torch.zeros(16)
_SCALAR_MASK_CPU[0] = 1.0
for _grade, _indices in A.GRADE_INDICES.items():
    _GRADE_PROJ_CPU[_grade, list(_indices)] = 1.0
    _B2G_CPU[list(_indices)] = _grade


@lru_cache(maxsize=8)
def _cached_cayley(device, dtype) -> torch.Tensor:
    return _CAYLEY_DENSE_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_metric_grades(device, dtype) -> torch.Tensor:
    return _METRIC_GRADES_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_inner_product_factors(device, dtype) -> torch.Tensor:
    return _INNER_PRODUCT_FACTORS_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_reverse_signs(device, dtype) -> torch.Tensor:
    return _REVERSE_SIGNS_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_grade_proj(device, dtype) -> torch.Tensor:
    return _GRADE_PROJ_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_b2g(device) -> torch.Tensor:
    return _B2G_CPU.to(device=device)


@lru_cache(maxsize=8)
def _cached_dual_dense(device, dtype) -> torch.Tensor:
    return _DUAL_DENSE_CPU.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _cached_scalar_mask(device, dtype) -> torch.Tensor:
    return _SCALAR_MASK_CPU.to(device=device, dtype=dtype)


# Default to full Lorentz equivariance. Models can request 10 basis maps per
# layer by passing n_basis=10 through EquivariantLinear construction.
LINEAR_BASIS_ELEMENTS = 5


def _is_any_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()


def minimum_autocast_precision(min_dtype=torch.float32):
    """Upcast floating inputs to min_dtype inside AMP regions for GA kernels."""

    def decorator(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _is_any_autocast_enabled():
                return fn(*args, **kwargs)

            def cast(t):
                if (
                    isinstance(t, Tensor)
                    and t.is_floating_point()
                    and torch.finfo(t.dtype).bits < torch.finfo(min_dtype).bits
                ):
                    return t.to(min_dtype)
                return t

            args_cast = tuple(cast(a) for a in args)
            kwargs_cast = {k: cast(v) for k, v in kwargs.items()}
            with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
                return fn(*args_cast, **kwargs_cast)

        return wrapper

    return decorator


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Drop entire Clifford grades independently."""
    if not training or p == 0.0:
        return x
    proj = _cached_grade_proj(x.device, x.dtype)
    projected = torch.einsum("gi,...i->...gi", proj, x)
    shape = projected.shape
    flat = projected.reshape(-1, 5, 16)
    flat = F.dropout1d(flat, p=p, training=True)
    return flat.reshape(shape).sum(dim=-2)


class GradeDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.register_buffer("grade_proj", _GRADE_PROJ_CPU.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        proj = self.grade_proj.to(dtype=x.dtype)
        projected = torch.einsum("gi,...i->...gi", proj, x)
        shape = projected.shape
        flat = projected.reshape(-1, 5, 16)
        flat = F.dropout1d(flat, p=self.p, training=True)
        return flat.reshape(shape).sum(dim=-2)


class EquivariantRMSNorm(nn.Module):
    """Stateless equivariant RMS normalization using per-grade GA norms."""

    def __init__(self, channels: int | None = None, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("metric_grades", _METRIC_GRADES_CPU.clone())

    @minimum_autocast_precision(torch.float32)
    def _abs_squared_norm(self, x: torch.Tensor) -> torch.Tensor:
        metric_grades = self.metric_grades.to(dtype=x.dtype)
        per_grade = torch.einsum("...i,...i,gi->...g", x, x, metric_grades)
        return per_grade.abs().sum(-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms = self._abs_squared_norm(x)
        norms = norms.mean(dim=-2, keepdim=True)
        norms = torch.clamp(norms, min=self.epsilon)
        return x * torch.rsqrt(norms)


class EquivariantLinear(nn.Module):
    """Channel mixing within equivariant grade/dual basis maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias_scalar: bool = True,
        initialization: str = "default",
        n_basis: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_basis = n_basis if n_basis is not None else LINEAR_BASIS_ELEMENTS
        if self.n_basis not in (5, 10):
            raise ValueError("EquivariantLinear n_basis must be 5 or 10")

        scale = 0.1 if initialization == "small" else 1.0
        self.weights = nn.ParameterList()
        for _ in range(self.n_basis):
            w = torch.empty(out_channels, in_channels)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if scale != 1.0:
                with torch.no_grad():
                    w.mul_(scale)
            self.weights.append(nn.Parameter(w))

        if bias_scalar:
            self.bias_scalar = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_scalar", None)
        self.register_buffer("basis_to_grade", _B2G_CPU.clone())
        self.register_buffer("dual_dense", _DUAL_DENSE_CPU.clone())
        self.register_buffer("scalar_mask", _SCALAR_MASK_CPU.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b2g = self.basis_to_grade
        weights = torch.stack(list(self.weights))  # (n_basis, out, in)
        w_proj = weights[:5][b2g]                  # (16, out, in)
        out = torch.einsum("...ib,boi->...ob", x, w_proj)

        if self.n_basis == 10:
            w_dual = weights[5:][b2g]
            mixed = torch.einsum("...ib,boi->...ob", x, w_dual)
            dual = self.dual_dense.to(dtype=x.dtype)
            out = out + torch.einsum("...ob,bk->...ok", mixed, dual)

        if self.bias_scalar is not None:
            bias = self.bias_scalar.view(*([1] * (out.ndim - 2)), self.out_channels, 1)
            mask = self.scalar_mask.to(dtype=out.dtype).view(*([1] * (out.ndim - 1)), 16)
            out = out + bias * mask
        return out


class GeometricProduct(nn.Module):
    """Equivariant bilinear layer based on the Cl(1,3) geometric product."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        n_basis: int | None = None,
    ):
        super().__init__()
        self.mid_channels = mid_channels or max(4, min(in_channels, 8))
        self.proj_x = EquivariantLinear(
            in_channels, self.mid_channels, bias_scalar=False, n_basis=n_basis
        )
        self.proj_y = EquivariantLinear(
            in_channels, self.mid_channels, bias_scalar=False, n_basis=n_basis
        )
        w = torch.empty(out_channels, self.mid_channels, self.mid_channels)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        with torch.no_grad():
            w.mul_(1.0 / math.sqrt(self.mid_channels))
        self.combine = nn.Parameter(w)
        self.register_buffer("cayley", _CAYLEY_DENSE_CPU.clone())

    @minimum_autocast_precision(torch.float32)
    def _gp_forward(self, xp: Tensor, yp: Tensor, combine: Tensor) -> Tensor:
        cayley = self.cayley.to(dtype=xp.dtype)
        gp_pair = torch.einsum("...ai,...bj,ijk->...abk", xp, yp, cayley)
        return torch.einsum("...abk,oab->...ok", gp_pair, combine)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        y = x if y is None else y
        xp = self.proj_x(x)
        yp = self.proj_y(y)
        return self._gp_forward(xp, yp, self.combine)


class FusedMultiHeadAttention(nn.Module):
    """Multi-head equivariant attention dispatched through torch SDPA."""

    def __init__(
        self,
        channels: int,
        n_heads: int = 1,
        attn_channels: int | None = None,
        dropout: float = 0.0,
        n_basis: int | None = None,
    ):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by n_heads ({n_heads})")
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.attn_channels = attn_channels or self.head_dim

        self.q_proj = EquivariantLinear(
            channels, self.attn_channels * n_heads, bias_scalar=False, n_basis=n_basis
        )
        self.k_proj = EquivariantLinear(
            channels, self.attn_channels * n_heads, bias_scalar=False, n_basis=n_basis
        )
        self.v_proj = EquivariantLinear(channels, channels, bias_scalar=False, n_basis=n_basis)
        self.out_proj = EquivariantLinear(
            channels, channels, initialization="small", n_basis=n_basis
        )
        self.head_scale = nn.Parameter(torch.ones(n_heads))
        self.dropout = dropout
        self.register_buffer("inner_product_factors", _INNER_PRODUCT_FACTORS_CPU.clone())
        self.register_buffer("reverse_signs", _REVERSE_SIGNS_CPU.clone())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, channels, basis = x.shape
        heads = self.n_heads
        attn_channels = self.attn_channels

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        ip = self.inner_product_factors.to(dtype=x.dtype)
        rev = self.reverse_signs.to(dtype=x.dtype)
        q = q * (rev * ip)

        q = q.reshape(batch, tokens, heads, attn_channels * basis).transpose(1, 2)
        k = k.reshape(batch, tokens, heads, attn_channels * basis).transpose(1, 2)
        v = v.reshape(batch, tokens, heads, self.head_dim * basis).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # SDPA bool mask uses True for positions that participate. The rest
            # of the code passes True for padding, so invert here.
            attn_mask = (~mask.bool()).unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out * self.head_scale.view(1, heads, 1, 1)
        out = out.transpose(1, 2).reshape(batch, tokens, channels, basis)
        return self.out_proj(out)


class GatedActivation(nn.Module):
    """Per-channel scalar gate driven by the scalar component."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar = x[..., 0]
        gate = torch.sigmoid(self.gate(scalar))
        return x * gate.unsqueeze(-1)


class JoinProduct(nn.Module):
    """Antisymmetrized geometric product (wedge)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        zero_init: bool = False,
        n_basis: int | None = None,
    ):
        super().__init__()
        self.mid_channels = mid_channels or max(4, min(in_channels, 8))
        self.proj_x = EquivariantLinear(
            in_channels, self.mid_channels, bias_scalar=False, n_basis=n_basis
        )
        self.proj_y = EquivariantLinear(
            in_channels, self.mid_channels, bias_scalar=False, n_basis=n_basis
        )
        w = torch.empty(out_channels, self.mid_channels, self.mid_channels)
        if zero_init:
            nn.init.normal_(w, std=1e-4)
        else:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            with torch.no_grad():
                w.mul_(1.0 / math.sqrt(self.mid_channels))
        self.combine = nn.Parameter(w)
        self.register_buffer("cayley", _CAYLEY_DENSE_CPU.clone())

    @minimum_autocast_precision(torch.float32)
    def _wedge_forward(self, xp: Tensor, yp: Tensor, combine: Tensor) -> Tensor:
        cayley = self.cayley.to(dtype=xp.dtype)
        gp_xy = torch.einsum("...ai,...bj,ijk->...abk", xp, yp, cayley)
        gp_yx = torch.einsum("...bi,...aj,ijk->...abk", yp, xp, cayley)
        wedge_pair = 0.5 * (gp_xy - gp_yx)
        return torch.einsum("...abk,oab->...ok", wedge_pair, combine)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        y = x if y is None else y
        xp = self.proj_x(x)
        yp = self.proj_y(y)
        return self._wedge_forward(xp, yp, self.combine)


@dataclass
class BlockConfig:
    channels: int = 16
    gp_mid_channels: int = 8
    attn_channels: int = 8
    n_heads: int = 1
    scalar_dropout: float = 0.0
    n_basis: int = 5


class GATrBlock(nn.Module):
    """Residual block: RMSNorm -> GP, RMSNorm -> SDPA, RMSNorm -> gate."""

    def __init__(self, cfg: BlockConfig | None = None):
        super().__init__()
        cfg = cfg or BlockConfig()
        self.cfg = cfg
        channels = cfg.channels

        self.norm1 = EquivariantRMSNorm(channels)
        self.gp_pre = EquivariantLinear(channels, channels, bias_scalar=False, n_basis=cfg.n_basis)
        self.gp = GeometricProduct(
            channels, channels, mid_channels=cfg.gp_mid_channels, n_basis=cfg.n_basis
        )
        self.gp_post = EquivariantLinear(channels, channels, n_basis=cfg.n_basis)

        self.norm2 = EquivariantRMSNorm(channels)
        self.attn = FusedMultiHeadAttention(
            channels,
            n_heads=cfg.n_heads,
            attn_channels=cfg.attn_channels,
            dropout=cfg.scalar_dropout,
            n_basis=cfg.n_basis,
        )

        self.norm3 = EquivariantRMSNorm(channels)
        self.gate_pre = EquivariantLinear(
            channels, channels, bias_scalar=False, n_basis=cfg.n_basis
        )
        self.gate = GatedActivation(channels)
        self.dropout = GradeDropout(cfg.scalar_dropout) if cfg.scalar_dropout > 0 else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.gp_pre(h)
        h = self.gp(h, h)
        h = self.gp_post(h)
        if self.dropout is not None:
            h = self.dropout(h)
        x = x + h

        h = self.norm2(x)
        h = self.attn(h, mask=mask)
        if self.dropout is not None:
            h = self.dropout(h)
        x = x + h

        h = self.norm3(x)
        h = self.gate_pre(h)
        x = x + self.gate(h)
        return x


class JoinBlock(nn.Module):
    """Compatibility block built around an antisymmetric wedge residual."""

    def __init__(self, cfg: BlockConfig | None = None):
        super().__init__()
        cfg = cfg or BlockConfig()
        channels = cfg.channels
        self.join_pre = EquivariantLinear(
            channels, channels, bias_scalar=False, n_basis=cfg.n_basis
        )
        self.join = JoinProduct(
            channels, channels, mid_channels=cfg.gp_mid_channels, n_basis=cfg.n_basis
        )
        self.join_post = EquivariantLinear(channels, channels, n_basis=cfg.n_basis)
        self.attn = FusedMultiHeadAttention(
            channels,
            n_heads=cfg.n_heads,
            attn_channels=cfg.attn_channels,
            dropout=cfg.scalar_dropout,
            n_basis=cfg.n_basis,
        )
        self.gate_pre = EquivariantLinear(
            channels, channels, bias_scalar=False, n_basis=cfg.n_basis
        )
        self.gate = GatedActivation(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.join_pre(x)
        h = self.join(h, h)
        h = self.join_post(h)
        x = x + h
        h = self.attn(x, mask=mask)
        x = x + h
        h = self.gate_pre(x)
        return x + self.gate(h)


# Backward-compatible names used by older tests/imports.
ScalarAttention = FusedMultiHeadAttention
MultiHeadScalarAttention = FusedMultiHeadAttention
