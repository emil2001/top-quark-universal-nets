"""Single-stack Lorentz geometric algebra event reconstruction (LGaTr)."""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .gatr import algebra as A
from .gatr.layers import BlockConfig, EquivariantLinear, GATrBlock
from .models import REG_DIM_V5, _make_head

# Energy scale for normalising 4-momenta to O(1) before entering GA layers.
E_SCALE = 100.0  # GeV

# Input multivector channel layout
MV_C_IN = 6  # 1 grade-1 (4-momentum) + 5 grade-0 (btag, lep_flavor, type×3)
PARTON_REG_HEAD_TYPES = ("grade1", "strict-equivariant", "hybrid-residual")


def _linear_basis_count(fully_connected_subgroup: bool) -> int:
    return 10 if fully_connected_subgroup else 5


def _validate_parton_reg_head_type(head_type: str) -> str:
    if head_type not in PARTON_REG_HEAD_TYPES:
        allowed = ", ".join(PARTON_REG_HEAD_TYPES)
        raise ValueError(f"parton_reg_head_type must be one of {allowed}; got {head_type!r}")
    return head_type


def _embed_vector(vec) -> np.ndarray:
    mv = np.zeros(16, dtype=np.float32)
    for mu, val in enumerate(vec):
        mv[A.GRADE1_INDICES[mu]] = float(val)
    return mv


def build_spurions(
    beam_spurion: str | None = "xyplane",
    add_time_spurion: bool = True,
    beam_mirror: bool = True,
) -> np.ndarray:
    """Build constant reference multivectors for optional symmetry breaking."""
    if beam_spurion == "none":
        beam_spurion = None
    if beam_spurion not in ("xyplane", "lightlike", "spacelike", "timelike", None):
        raise ValueError("beam_spurion must be xyplane/lightlike/spacelike/timelike/none")

    rows = []
    if beam_spurion in ("lightlike", "spacelike", "timelike"):
        beam = {
            "lightlike": [1.0, 0.0, 0.0, 1.0],
            "timelike": [2.0 ** 0.5, 0.0, 0.0, 1.0],
            "spacelike": [0.0, 0.0, 0.0, 1.0],
        }[beam_spurion]
        rows.append(_embed_vector(beam))
        if beam_mirror:
            rows.append(_embed_vector([beam[0], 0.0, 0.0, -beam[3]]))
    elif beam_spurion == "xyplane":
        mv = np.zeros(16, dtype=np.float32)
        mv[A.IDX_E12] = 1.0
        rows.append(mv)

    if add_time_spurion:
        rows.append(_embed_vector([1.0, 0.0, 0.0, 0.0]))

    if not rows:
        return np.zeros((0, 16), dtype=np.float32)
    return np.stack(rows, axis=0)


def _build_detector_multivectors(
    raw_4mom: torch.Tensor,
    scalar_feats: torch.Tensor,
) -> torch.Tensor:
    """Build input multivectors for the 17 detector tokens.

    Args:
        raw_4mom:    (B, 17, 4)   raw (E, px, py, pz) in GeV (not z-scored)
        scalar_feats:(B, 17, 5)   [btag, lep_flavor, type_MET, type_lep, type_jet]

    Returns:
        (B, 17, MV_C_IN=6, 16)  multivector tensor
    """
    B, T, _ = raw_4mom.shape
    # Channel 0: grade-1 basis indices 1..4 -> (E, px, py, pz) / E_SCALE.
    p4_mv = F.pad(raw_4mom / E_SCALE, (1, 11)).unsqueeze(2)
    # Channels 1-5: grade-0 scalar component from scalar_feats.
    scalar_tail = raw_4mom.new_zeros(B, T, scalar_feats.size(-1), 15)
    scalar_mv = torch.cat([scalar_feats.unsqueeze(-1), scalar_tail], dim=-1)
    return torch.cat([p4_mv, scalar_mv], dim=2)


# ---------------------------------------------------------------------------
# VX-full helpers and classes
# ---------------------------------------------------------------------------

def _cartesian_to_v5(p4: torch.Tensor) -> torch.Tensor:
    """Convert physical Cartesian 4-momenta to the V5 target contract.

    Args:
        p4: (*, 4)  (E, px, py, pz) in GeV

    Returns:
        (*, 5)  (log1p(pT), eta, sin_phi, cos_phi, sign(E) * log1p(abs(E)))

    The cached V5 targets store pT and E in log1p space.  Keep this transform
    adjacent to the physical readout so that ``_zscore_v5`` always receives the
    same representation as the targets.  The smooth transverse norm keeps the
    zero-vector case finite and differentiable. Energy uses a signed logarithm
    so negative readouts remain distinguishable and train toward positive targets.
    """
    # bf16 machine epsilon is far too large in GeV^2 for low-pT directions.
    # Perform this kinematic conversion in fp32 while retaining gradients.
    work = p4.float() if p4.dtype in (torch.float16, torch.bfloat16) else p4
    E = work[..., 0]
    px = work[..., 1]
    py = work[..., 2]
    pz = work[..., 3]
    eps = 1e-12
    pT = torch.sqrt(px.square() + py.square() + eps)
    eta = torch.asinh(pz / pT)
    sin_phi = py / pT
    cos_phi = px / pT
    signed_log_energy = torch.where(E < 0, -torch.log1p(-E), torch.log1p(E))
    return torch.stack(
        [torch.log1p(pT), eta, sin_phi, cos_phi, signed_log_energy], dim=-1
    )


class InvariantAssignHead(nn.Module):
    """Assignment logits via per-channel Lorentz-invariant inner products.

    For each (source, target) token pair the per-channel grade-0 inner product
    <~source_c target_c>_0 is computed; these C scalars are passed through a
    small MLP to produce one assignment score.  A learned no-match score is
    appended as the last column.

    Input:
        source_mv: (B, N_s, C, 16)
        target_mv: (B, N_t, C, 16)
    Output:
        (B, N_s, N_t + 1)  logits
    """

    def __init__(self, channels: int, n_keys: int):
        super().__init__()
        self.channels = channels

        self.register_buffer("_cay0", torch.from_numpy(A.CAYLEY_DENSE[:, :, 0].copy()))
        self.scorer = nn.Sequential(
            nn.Linear(channels, max(channels // 2, 4)),
            nn.GELU(),
            nn.Linear(max(channels // 2, 4), 1),
        )
        self.no_match_score = nn.Parameter(torch.zeros(1))

    def forward(self, source_mv: torch.Tensor, target_mv: torch.Tensor) -> torch.Tensor:
        B, Ns = source_mv.shape[:2]

        # grade-0 inner product <~source target>_0 per channel, per pair
        source_rev = A.clifford_reverse(source_mv)          # (B, Ns, C, 16)
        cay0 = self._cay0.to(dtype=source_mv.dtype)          # (16, 16)
        src_cay0 = torch.matmul(source_rev, cay0)           # (B, Ns, C, 16)
        # pairwise: (B, Ns, C, 16) × (B, Nt, C, 16) → (B, Ns, Nt, C)
        inv = torch.einsum("bick,bjck->bijc", src_cay0, target_mv)

        scores = self.scorer(inv).squeeze(-1)               # (B, Ns, Nt)
        no_match = self.no_match_score.expand(B, Ns, 1)
        return torch.cat([scores, no_match], dim=-1)        # (B, Ns, Nt+1)


class GradeDecomposedAssignHead(nn.Module):
    """Assignment logits via per-grade Lorentz-invariant inner products.

    For each (source, target) token pair, five per-grade invariants
    <~source_{c,k}  target_{c,k}>_0  (k = 0..4) are computed per channel,
    giving 5*C features.  A small MLP maps these to a single score.
    A learned no-match score is appended as the last column.

    Input:
        source_mv: (B, N_s, C, 16)
        target_mv: (B, N_t, C, 16)
    Output:
        (B, N_s, N_t + 1)  logits
    """

    def __init__(self, channels: int, n_keys: int):
        super().__init__()
        self.channels = channels

        cay0 = A.CAYLEY_DENSE[:, :, 0]  # (16, 16)
        self._grade_idx: list[list[int]] = []
        for k in range(5):
            idx = list(A.GRADE_INDICES[k])
            self._grade_idx.append(idx)
            sub = cay0[idx][:, idx].copy()  # (n_k, n_k) submatrix
            self.register_buffer(f"_cay0_g{k}", torch.from_numpy(sub))

        mid = max(channels // 2, 4)
        self.scorer = nn.Sequential(
            nn.Linear(5 * channels, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )
        self.no_match_bias = nn.Parameter(torch.tensor(1.0))
        self.no_match_proj = nn.Sequential(
            nn.Linear(5 * channels, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

    def forward(self, source_mv: torch.Tensor, target_mv: torch.Tensor) -> torch.Tensor:
        B, Ns = source_mv.shape[:2]

        source_rev = A.clifford_reverse(source_mv)          # (B, Ns, C, 16)

        per_grade = []
        self_per_grade = []
        for k in range(5):
            idx = self._grade_idx[k]
            s_k = source_rev[..., idx]                       # (B, Ns, C, n_k)
            t_k = target_mv[..., idx]                        # (B, Nt, C, n_k)
            cay0_kk = getattr(self, f"_cay0_g{k}").to(dtype=s_k.dtype)  # (n_k, n_k)
            s_cay = torch.matmul(s_k, cay0_kk)              # (B, Ns, C, n_k)
            inv_k = torch.einsum("bick,bjck->bijc", s_cay, t_k)
            per_grade.append(inv_k)
            self_per_grade.append(
                (s_cay * source_mv[..., idx]).sum(dim=-1)    # (B, Ns, C)
            )

        feats = torch.cat(per_grade, dim=-1)                 # (B, Ns, Nt, 5*C)
        scores = self.scorer(feats).squeeze(-1)              # (B, Ns, Nt)

        src_self = torch.cat(self_per_grade, dim=-1)         # (B, Ns, 5*C)
        no_match = self.no_match_bias + self.no_match_proj(src_self)  # (B, Ns, 1)
        return torch.cat([scores, no_match], dim=-1)         # (B, Ns, Nt+1)


class PatEmbedderModelVXFull(nn.Module):
    """VX-full: single-stack GATr encoder over all 25 tokens (no decoder).

    All tokens — MET, leptons, jets, top queries, W queries — participate in
    one GATr encoder as Cl(1,3) multivectors.  Regression reads grade-1
    components; classification reads grade-0 scalars plus the full Clifford
    self-norm; assignment reads per-grade pairwise invariants (5*C features);
    HLR pools per-grade self-norms with learned invariant attention.
    There is no DETR decoder, no DN-DETR, and no auxiliary decoder losses.
    Presence heads predict per-slot occupancy (BCE-supervised via masks).

    Inputs:
        x:               (B, 130) z-scored flat features (for scalar channels)
        x_4mom:          (B, 17, 4) raw (E, px, py, pz) in GeV
        x_raw:           (B, 130) un-normalised flat features
        enc_padding_mask: optional (B, 17) bool mask — True = padding token

    Outputs: 12-tuple identical to PatEmbedderModelV3 minus aux_head_outputs
    and dn_outputs (both absent since there is no decoder):
        (lep_reg, neu_reg, jet_reg, top_reg, w_reg,
         lep_cls, neu_cls, jet_cls, top_cls, w_cls,
         assign, hlr)
    """

    def __init__(
        self,
        max_leptons: int = 4,
        max_jets: int = 12,
        max_tops: int = 4,
        max_ws: int = 4,
        ga_channels: int = 32,
        ga_heads: int = 4,
        n_enc_layers: int = 8,
        dropout: float = 0.1,
        num_hlr_vars: int = 0,
        reg_dim: int = REG_DIM_V5,
        fully_connected_subgroup: bool = False,
        scalar_dropout: float | None = None,
        gp_mid_channels: int | None = None,
        attn_channels: int | None = None,
        checkpoint_ga_blocks: bool = False,
        spurions: bool = False,
        beam_spurion: str | None = "xyplane",
        add_time_spurion: bool = True,
        beam_mirror: bool = True,
        parton_reg_head_type: str = "grade1",
        parton_reg_channels: int = 4,
        parton_reg_residual_scale: float = 0.25,
    ):
        super().__init__()
        parton_reg_head_type = _validate_parton_reg_head_type(parton_reg_head_type)
        if parton_reg_channels < 1:
            raise ValueError("parton_reg_channels must be >= 1")
        self.max_leptons = max_leptons
        self.max_jets = max_jets
        self.max_tops = max_tops
        self.max_ws = max_ws
        self.ga_channels = ga_channels
        self.num_hlr_vars = num_hlr_vars
        self.reg_dim = reg_dim
        self.checkpoint_ga_blocks = checkpoint_ga_blocks
        self.parton_reg_head_type = parton_reg_head_type
        self.parton_reg_channels = parton_reg_channels
        self.parton_reg_residual_scale = parton_reg_residual_scale

        n_basis = _linear_basis_count(fully_connected_subgroup)
        scalar_dropout = dropout if scalar_dropout is None else scalar_dropout
        spurion_arr = build_spurions(
            beam_spurion if spurions else None,
            add_time_spurion if spurions else False,
            beam_mirror,
        )
        self.n_spurions = int(spurion_arr.shape[0])
        self.register_buffer("spurion_tokens", torch.from_numpy(spurion_arr))

        # Token counts
        self._n_det = 1 + max_leptons + max_jets  # 17
        self._n_all = self._n_det + max_tops + max_ws  # 25

        # Slices into the 25-token sequence
        self._met_sl  = slice(0, 1)
        self._lep_sl  = slice(1, 1 + max_leptons)
        self._jet_sl  = slice(1 + max_leptons, self._n_det)
        self._top_sl  = slice(self._n_det, self._n_det + max_tops)
        self._w_sl    = slice(self._n_det + max_tops, self._n_all)

        # ---- GA encoder ----
        self.input_proj = EquivariantLinear(MV_C_IN, ga_channels, n_basis=n_basis)
        ga_cfg = BlockConfig(
            channels=ga_channels,
            n_heads=ga_heads,
            gp_mid_channels=gp_mid_channels or max(4, ga_channels // 4),
            attn_channels=attn_channels or max(4, ga_channels // 2),
            scalar_dropout=scalar_dropout,
            n_basis=n_basis,
        )
        self.ga_encoder = nn.ModuleList([GATrBlock(ga_cfg) for _ in range(n_enc_layers)])

        # ---- Learnable scalar parton queries ----
        # The final singleton dimension is the grade-0 coefficient.  Padding it
        # into a multivector below prevents optimisation from creating fixed
        # lab-frame directions.  Its changed state-dict shape deliberately makes
        # pre-contract checkpoints fail to load rather than reinterpret them.
        self.top_mv_q = nn.Parameter(torch.zeros(max_tops, ga_channels, 1))
        self.w_mv_q = nn.Parameter(torch.zeros(max_ws, ga_channels, 1))
        nn.init.normal_(self.top_mv_q[..., 0], std=0.02)
        nn.init.normal_(self.w_mv_q[..., 0], std=0.02)
        self.register_buffer("_vx_full_contract_version", torch.tensor(3, dtype=torch.int64))

        # ---- Regression heads: EquivariantLinear(C→1) → grade-1 → convert ----
        hd = dropout * 0.5
        self.lep_reg_proj = EquivariantLinear(ga_channels, 1, bias_scalar=False)
        self.neu_reg_proj = EquivariantLinear(ga_channels, 1, bias_scalar=False)
        self.jet_reg_proj = EquivariantLinear(ga_channels, 1, bias_scalar=False)
        self.top_reg_proj = EquivariantLinear(ga_channels, 1, bias_scalar=False)
        self.w_reg_proj   = EquivariantLinear(ga_channels, 1, bias_scalar=False)
        if parton_reg_head_type == "strict-equivariant":
            self.top_parton_reg_proj = EquivariantLinear(
                ga_channels, parton_reg_channels, bias_scalar=False
            )
            self.w_parton_reg_proj = EquivariantLinear(
                ga_channels, parton_reg_channels, bias_scalar=False
            )
            self.top_parton_reg_gate = _make_head(2 * ga_channels, parton_reg_channels, hd)
            self.w_parton_reg_gate = _make_head(2 * ga_channels, parton_reg_channels, hd)
        else:
            self.top_parton_reg_proj = None
            self.w_parton_reg_proj = None
            self.top_parton_reg_gate = None
            self.w_parton_reg_gate = None
        if parton_reg_head_type == "hybrid-residual":
            self.top_parton_reg_residual = _make_head(2 * ga_channels, reg_dim, hd)
            self.w_parton_reg_residual = _make_head(2 * ga_channels, reg_dim, hd)
        else:
            self.top_parton_reg_residual = None
            self.w_parton_reg_residual = None

        # ---- Classification heads: grade-0 + self-norm (2*C) → MLP → logits ----
        cls_in = 2 * ga_channels
        self.lepton_cls_head   = _make_head(cls_in, 7, hd)
        self.neutrino_cls_head = _make_head(cls_in, 7, hd)
        self.jet_cls_head      = _make_head(cls_in, 3, hd)
        self.top_cls_head      = _make_head(cls_in, 3, hd)
        self.w_cls_head        = _make_head(cls_in, 3, hd)

        # ---- Assignment heads: per-grade invariant inner products ----
        self.assign_top_to_w      = GradeDecomposedAssignHead(ga_channels, max_ws)
        self.assign_top_to_bjet   = GradeDecomposedAssignHead(ga_channels, max_jets)
        self.assign_w_to_lepton   = GradeDecomposedAssignHead(ga_channels, max_leptons)
        self.assign_w_to_neutrino = GradeDecomposedAssignHead(ga_channels, max_leptons)
        self.assign_w_to_jet1     = GradeDecomposedAssignHead(ga_channels, max_jets)
        self.assign_w_to_jet2     = GradeDecomposedAssignHead(ga_channels, max_jets)

        # ---- Presence heads: grade-0 + self-norm (2*C) → MLP → 1 logit ----
        self.top_presence_head = _make_head(cls_in, 1, hd)
        self.w_presence_head   = _make_head(cls_in, 1, hd)

        # ---- HLR: per-grade invariant self-norms → attention pool → MLP ----
        if num_hlr_vars > 0:
            self.hlr_attn_proj = nn.Linear(5 * ga_channels, 1)
            self.hlr_head = _make_head(5 * ga_channels, num_hlr_vars, hd)
        else:
            self.hlr_attn_proj = None
            self.hlr_head = None
        cay0_diag_by_grade = torch.zeros(5, 16)
        cay0_diag = torch.from_numpy(A.CAYLEY_DENSE[:, :, 0].diagonal().copy())
        for k, idx in A.GRADE_INDICES.items():
            cay0_diag_by_grade[k, list(idx)] = cay0_diag[list(idx)]
        self.register_buffer("_cay0_diag_by_grade", cay0_diag_by_grade)
        self.register_buffer(
            "_inner_product_factors",
            torch.from_numpy(A.INNER_PRODUCT_FACTORS.copy()),
        )

        # ---- Regression z-scoring buffers ----
        # _grade1_to_v5 outputs (log1p(pT), eta, sin_phi, cos_phi, log1p(E)),
        # matching the cache target contract. These buffers hold per-slot
        # (mu_pT, mu_eta, mu_E) and (std_pT, std_eta, std_E) so forward()
        # can map the output into the same domain as the targets.
        for short, n in [("lep", max_leptons), ("neu", max_leptons),
                         ("jet", max_jets), ("top", max_tops), ("w", max_ws)]:
            self.register_buffer(f"_reg_mean_{short}", torch.zeros(n, 3))
            self.register_buffer(f"_reg_std_{short}", torch.ones(n, 3))

    def _append_spurions_postproj(self, mv: torch.Tensor) -> torch.Tensor:
        if self.n_spurions == 0:
            return mv
        B = mv.size(0)
        sp = self.spurion_tokens.to(device=mv.device, dtype=mv.dtype)
        sp_first = sp.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)
        sp_rest = mv.new_zeros(B, self.n_spurions, self.ga_channels - 1, 16)
        sp_tokens = torch.cat([sp_first, sp_rest], dim=2)
        return torch.cat([mv, sp_tokens], dim=1)

    def _augment_full_mask(self, full_mask: torch.Tensor | None) -> torch.Tensor | None:
        if full_mask is None or self.n_spurions == 0:
            return full_mask
        valid = torch.zeros(
            full_mask.size(0),
            self.n_spurions,
            dtype=torch.bool,
            device=full_mask.device,
        )
        return torch.cat([full_mask, valid], dim=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_scalar_feats(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Same as VXHybrid's method — 5 scalar channels for detector tokens."""
        B = x_raw.size(0)
        zero_met = x_raw.new_zeros(B, 1)
        one_met = x_raw.new_ones(B, 1)
        met_s = torch.stack([zero_met, zero_met, one_met, zero_met, zero_met], dim=-1)

        lep_r = x_raw[:, 2:2 + self.max_leptons * 8].view(B, self.max_leptons, 8)
        zero_lep = x_raw.new_zeros(B, self.max_leptons)
        one_lep = x_raw.new_ones(B, self.max_leptons)
        lep_s = torch.stack([zero_lep, lep_r[:, :, 7], zero_lep, one_lep, zero_lep], dim=-1)

        jet_r = x_raw[:, 2 + self.max_leptons * 8:].view(B, self.max_jets, 8)
        zero_jet = x_raw.new_zeros(B, self.max_jets)
        one_jet = x_raw.new_ones(B, self.max_jets)
        jet_s = torch.stack([jet_r[:, :, 7], zero_jet, zero_jet, zero_jet, one_jet], dim=-1)

        return torch.cat([met_s, lep_s, jet_s], dim=1)  # (B, 17, 5)

    def _grade1_to_v5(self, proj: EquivariantLinear, mv: torch.Tensor, N: int) -> torch.Tensor:
        """Apply EquivariantLinear, read grade-1, convert to V5 targets."""
        out1ch = proj(mv)                             # (B, N, 1, 16)
        p4 = out1ch[:, :, 0, 1:5].contiguous() * E_SCALE  # (B, N, 4) raw (E,px,py,pz)
        return _cartesian_to_v5(p4)                   # (B, N, 5)

    def _invariant_features(self, mv: torch.Tensor) -> torch.Tensor:
        """Return grade-0 scalars plus Clifford self-norms: (B, N, 2*C)."""
        g0 = mv[:, :, :, 0]
        factors = self._inner_product_factors.to(dtype=mv.dtype)
        norm = (mv * mv * factors).sum(dim=-1)
        return torch.cat([g0, norm], dim=-1)

    def _multi_grade1_to_v5(
        self,
        proj: EquivariantLinear,
        gate: nn.Module,
        mv: torch.Tensor,
    ) -> torch.Tensor:
        """Read multiple equivariant grade-1 candidates and mix them with invariant gates."""
        candidates = proj(mv)[:, :, :, 1:5].contiguous() * E_SCALE
        B, N, R, _ = candidates.shape
        gate_logits = gate(self._invariant_features(mv).reshape(B * N, -1)).view(B, N, R)
        weights = torch.softmax(gate_logits, dim=-1)
        p4 = (candidates * weights.unsqueeze(-1)).sum(dim=2)
        return _cartesian_to_v5(p4)

    def _parton_regression(
        self,
        base_proj: EquivariantLinear,
        mv: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        strict_proj: EquivariantLinear | None,
        strict_gate: nn.Module | None,
        residual_head: nn.Module | None,
    ) -> torch.Tensor:
        if self.parton_reg_head_type == "strict-equivariant":
            if strict_proj is None or strict_gate is None:
                raise RuntimeError("strict-equivariant regression head was not initialised")
            raw_v5 = self._multi_grade1_to_v5(strict_proj, strict_gate, mv)
            return self._zscore_v5(raw_v5, mean, std)

        base = self._zscore_v5(self._grade1_to_v5(base_proj, mv, mv.size(1)), mean, std)
        if self.parton_reg_head_type != "hybrid-residual":
            return base
        if residual_head is None:
            raise RuntimeError("hybrid-residual regression head was not initialised")
        B, N = mv.shape[:2]
        residual = residual_head(self._invariant_features(mv).reshape(B * N, -1)).view(
            B, N, self.reg_dim
        )
        residual = self.parton_reg_residual_scale * torch.tanh(residual)
        # An invariant correction may shift invariant coordinates, but cannot
        # add fixed lab-frame sin/cos components. Rotate the existing phi pair
        # by an invariant angle instead.
        corrected = base.clone()
        corrected[..., 0] = corrected[..., 0] + residual[..., 0]
        corrected[..., 1] = corrected[..., 1] + residual[..., 1]
        corrected[..., 4] = corrected[..., 4] + residual[..., 4]
        delta_phi = residual[..., 2]
        sin_phi, cos_phi = base[..., 2], base[..., 3]
        corrected[..., 2] = sin_phi * torch.cos(delta_phi) + cos_phi * torch.sin(delta_phi)
        corrected[..., 3] = cos_phi * torch.cos(delta_phi) - sin_phi * torch.sin(delta_phi)
        return corrected

    def _invariant_cls(self, head: nn.Module, mv: torch.Tensor, N: int) -> torch.Tensor:
        """Extract grade-0 scalars + Clifford self-norm and classify."""
        B = mv.size(0)
        feats = self._invariant_features(mv)
        return head(feats.reshape(B * N, -1)).view(B, N, -1)

    @staticmethod
    def _zscore_v5(v5: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Z-score pT (dim 0), eta (dim 1), E (dim 4) of V5 kinematic output.

        sin_phi (dim 2) and cos_phi (dim 3) are left unchanged because the
        targets already store sin/cos of the un-normalised angle.

        Args:
            v5:   (B, N, 5)  raw (pT, eta, sin_phi, cos_phi, E)
            mean: (N, 3)     per-slot (mu_pT, mu_eta, mu_E)
            std:  (N, 3)     per-slot (std_pT, std_eta, std_E)
        """
        return torch.stack(
            [
                (v5[..., 0] - mean[..., 0]) / std[..., 0],
                (v5[..., 1] - mean[..., 1]) / std[..., 1],
                v5[..., 2],
                v5[..., 3],
                (v5[..., 4] - mean[..., 2]) / std[..., 2],
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        x_4mom: torch.Tensor,
        x_raw: torch.Tensor,
        enc_padding_mask: torch.Tensor | None = None,
        dn_reg: torch.Tensor | None = None,
        dn_cls: torch.Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple:
        del dn_reg, dn_cls, compute_aux
        B = x.size(0)

        # ---- 1. Build detector multivectors ----
        scalar_feats = self._extract_scalar_feats(x_raw)         # (B, 17, 5)
        det_mv = _build_detector_multivectors(x_4mom, scalar_feats)  # (B, 17, 6, 16)
        det_mv = self.input_proj(det_mv)                          # (B, 17, C, 16)
        if enc_padding_mask is not None:
            det_mv = det_mv.masked_fill(enc_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        # ---- 2. Append learnable parton query tokens ----
        top_q = F.pad(self.top_mv_q, (0, 15)).unsqueeze(0).expand(B, -1, -1, -1)
        w_q = F.pad(self.w_mv_q, (0, 15)).unsqueeze(0).expand(B, -1, -1, -1)
        mv = torch.cat([det_mv, top_q, w_q], dim=1)               # (B, 25, C, 16)
        mv = self._append_spurions_postproj(mv)

        # ---- 3. Build full padding mask: detector mask + False for parton slots ----
        full_mask = None
        if enc_padding_mask is not None:
            full_mask = F.pad(enc_padding_mask, (0, self.max_tops + self.max_ws), value=False)
            full_mask = self._augment_full_mask(full_mask)

        # ---- 4. GATr encoder ----
        for block in self.ga_encoder:
            if self.checkpoint_ga_blocks and self.training:
                mv = checkpoint(block, mv, full_mask, use_reentrant=False)
            else:
                mv = block(mv, mask=full_mask)                     # (B, 25+n_sp, C, 16)

        # ---- 5. Slice token groups ----
        lep_mv = mv[:, self._lep_sl]   # (B, 4, C, 16)
        jet_mv = mv[:, self._jet_sl]   # (B, 12, C, 16)
        top_mv = mv[:, self._top_sl]   # (B, 4, C, 16)
        w_mv   = mv[:, self._w_sl]     # (B, 4, C, 16)

        # ---- 6. Regression via grade-1 readout + z-scoring ----
        lep_reg = self._zscore_v5(
            self._grade1_to_v5(self.lep_reg_proj, lep_mv, self.max_leptons),
            self._reg_mean_lep, self._reg_std_lep)
        neu_reg = self._zscore_v5(
            self._grade1_to_v5(self.neu_reg_proj, lep_mv, self.max_leptons),
            self._reg_mean_neu, self._reg_std_neu)
        jet_reg = self._zscore_v5(
            self._grade1_to_v5(self.jet_reg_proj, jet_mv, self.max_jets),
            self._reg_mean_jet, self._reg_std_jet)
        top_reg = self._parton_regression(
            self.top_reg_proj,
            top_mv,
            self._reg_mean_top,
            self._reg_std_top,
            self.top_parton_reg_proj,
            self.top_parton_reg_gate,
            self.top_parton_reg_residual,
        )
        w_reg = self._parton_regression(
            self.w_reg_proj,
            w_mv,
            self._reg_mean_w,
            self._reg_std_w,
            self.w_parton_reg_proj,
            self.w_parton_reg_gate,
            self.w_parton_reg_residual,
        )

        # ---- 7. Classification via grade-0 + self-norm invariants ----
        lep_cls = self._invariant_cls(self.lepton_cls_head,   lep_mv, self.max_leptons)
        neu_cls = self._invariant_cls(self.neutrino_cls_head, lep_mv, self.max_leptons)
        jet_cls = self._invariant_cls(self.jet_cls_head,      jet_mv, self.max_jets)
        top_cls = self._invariant_cls(self.top_cls_head,      top_mv, self.max_tops)
        w_cls   = self._invariant_cls(self.w_cls_head,        w_mv,   self.max_ws)

        # ---- 8. Assignment via invariant inner products ----
        assign = {
            "top_to_w":      self.assign_top_to_w(top_mv, w_mv),
            "top_to_bjet":   self.assign_top_to_bjet(top_mv, jet_mv),
            "w_to_lepton":   self.assign_w_to_lepton(w_mv, lep_mv),
            "w_to_neutrino": self.assign_w_to_neutrino(w_mv, lep_mv),
            "w_to_jet1":     self.assign_w_to_jet1(w_mv, jet_mv),
            "w_to_jet2":     self.assign_w_to_jet2(w_mv, jet_mv),
        }
        assign["top_presence"] = self._invariant_cls(
            self.top_presence_head, top_mv, self.max_tops,
        ).squeeze(-1)  # (B, max_tops)
        assign["w_presence"] = self._invariant_cls(
            self.w_presence_head, w_mv, self.max_ws,
        ).squeeze(-1)  # (B, max_ws)

        # ---- 9. HLR via per-grade invariant attention pooling ----
        hlr = None
        if self.hlr_head is not None:
            rev = A.clifford_reverse(mv)
            diag = self._cay0_diag_by_grade.to(dtype=mv.dtype)
            inv_by_grade = torch.einsum("btci,btci,gi->btcg", rev, mv, diag)
            inv_feats = inv_by_grade.permute(0, 1, 3, 2).reshape(
                mv.size(0), mv.size(1), 5 * mv.size(2)
            )
            attn_logits = self.hlr_attn_proj(inv_feats)        # (B, 25, 1)
            if full_mask is not None:
                attn_logits = attn_logits.masked_fill(full_mask.unsqueeze(-1), float("-inf"))
            attn = torch.softmax(attn_logits, dim=1)
            pooled = (attn * inv_feats).sum(dim=1)             # (B, 5*C)
            hlr = self.hlr_head(pooled)

        return (
            lep_reg, neu_reg, jet_reg, top_reg, w_reg,
            lep_cls, neu_cls, jet_cls, top_cls, w_cls,
            assign, hlr,
        )
