from itertools import permutations as _iter_perms

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TASKS = 4  # regression, classification, assignment, hlr


def focal_ce_loss(logits, targets, gamma=2.0, reduction="none", cls_weights=None):
    """Focal loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t).
    Keeps all classes in training but down-weights easy (dominant) ones.

    Args:
        cls_weights: optional (C,) tensor of per-class multipliers applied
            *after* the focal modulation, so class 0 (background) can be
            up-weighted without interfering with the focal p_t calculation.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce)
    focal = ((1.0 - p_t) ** gamma) * ce
    if cls_weights is not None:
        focal = focal * cls_weights.to(logits.device)[targets]
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal


def _focal_ce_per_slot(logits, targets, gamma=2.0):
    """Focal CE returning per-slot losses: (B, N) from (B, N, C) logits and (B, N) targets."""
    B, N, C = logits.shape
    return focal_ce_loss(logits.reshape(B * N, C), targets.reshape(B * N), gamma).view(B, N)


def _mse_per_slot(pred, tgt):
    """MSE per slot: (B, N) from (B, N, D) inputs."""
    return F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)


# ---------------------------------------------------------------------------
# Permutation matching utilities
# ---------------------------------------------------------------------------

def _find_best_permutation(cost_matrix, n_slots):
    """Find the permutation of target slots minimizing total pairwise cost.

    Args:
        cost_matrix: (B, N, N) -- cost[b, i, j] = cost of pred slot i matching target slot j
        n_slots: int

    Returns:
        best_perm: (B, N) LongTensor
    """
    device = cost_matrix.device
    all_perms = list(_iter_perms(range(n_slots)))
    perm_t = torch.tensor(all_perms, device=device, dtype=torch.long)
    P = len(all_perms)

    slot_idx = torch.arange(n_slots, device=device)
    perm_costs = torch.zeros(cost_matrix.size(0), P, device=device)
    for pi in range(P):
        perm_costs[:, pi] = cost_matrix[:, slot_idx, perm_t[pi]].sum(dim=-1)

    best_pi = perm_costs.argmin(dim=-1)
    return perm_t[best_pi]


def _apply_perm_1d(tgt, perm):
    """Reorder (B, N) targets by (B, N) permutation."""
    return tgt.gather(1, perm)


def _apply_perm_2d(tgt, perm):
    """Reorder (B, N, D) targets by (B, N) permutation."""
    return tgt.gather(1, perm.unsqueeze(-1).expand_as(tgt))


def _target_in_query_normalization(tgt, mean, std):
    """Express every target slot in every query slot's V5 normalization.

    ``tgt[b,j]`` was z-scored with slot ``j`` statistics, while prediction
    ``pred[b,i]`` uses query-slot ``i`` statistics.  Matching must compare them
    in one physical (pre-zscore) coordinate system.
    """
    raw = tgt.unsqueeze(1).expand(-1, mean.size(0), -1, -1).clone()
    source_mean, source_std = mean.unsqueeze(0).unsqueeze(1), std.unsqueeze(0).unsqueeze(1)
    query_mean, query_std = mean.unsqueeze(0).unsqueeze(2), std.unsqueeze(0).unsqueeze(2)
    coords = (0, 1, tgt.size(-1) - 1)
    for stat_idx, coord in enumerate(coords):
        raw[..., coord] = (raw[..., coord] * source_std[..., stat_idx] + source_mean[..., stat_idx] - query_mean[..., stat_idx]) / query_std[..., stat_idx]
    return raw


def _apply_perm_with_query_normalization(tgt, perm, mean, std):
    """Select matched targets then express each one in its receiving query slot."""
    selected = _apply_perm_2d(tgt, perm)
    source_mean = _apply_perm_2d(mean.unsqueeze(0).expand(tgt.size(0), -1, -1), perm)
    source_std = _apply_perm_2d(std.unsqueeze(0).expand(tgt.size(0), -1, -1), perm)
    coords = (0, 1, tgt.size(-1) - 1)
    for stat_idx, coord in enumerate(coords):
        selected[..., coord] = (
            selected[..., coord] * source_std[..., stat_idx] + source_mean[..., stat_idx] - mean[:, stat_idx]
        ) / std[:, stat_idx]
    return selected


def _invert_perm(perm):
    """Invert a permutation: (B, N) -> (B, N)."""
    inv = torch.empty_like(perm)
    src = torch.arange(perm.size(1), device=perm.device).unsqueeze(0).expand_as(perm)
    inv.scatter_(1, perm, src)
    return inv


def _remap_assignment_indices(assign_tgt, perm_inv, no_match_idx):
    """Remap assignment target indices through an inverse permutation.
    Indices equal to no_match_idx are left unchanged."""
    B, N = assign_tgt.shape
    result = assign_tgt.clone()
    valid = assign_tgt < no_match_idx
    if valid.any():
        flat_valid = valid.view(-1)
        flat_assign = assign_tgt.view(-1)
        flat_perm_inv = perm_inv.unsqueeze(1).expand(B, N, perm_inv.size(1)).reshape(B * N, -1)
        remapped = flat_perm_inv[
            torch.arange(B * N, device=assign_tgt.device),
            flat_assign.clamp(max=perm_inv.size(1) - 1),
        ]
        result.view(-1)[flat_valid] = remapped[flat_valid]
    return result


# ---------------------------------------------------------------------------
# Loss V1
# ---------------------------------------------------------------------------

class PatEmbedderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(N_TASKS))
        self.mse_loss = nn.MSELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def _masked_mse(self, pred, tgt, mask):
        """MSE only on valid (non-padding) particle slots."""
        mse = self.mse_loss(pred, tgt).mean(dim=-1)
        mask_f = mask.float()
        mse = mse * mask_f
        n_valid = mask_f.sum(dim=0).clamp(min=1.0)
        return (mse.sum(dim=0) / n_valid)

    def forward(
        self,
        regression_outputs,
        classification_outputs,
        regression_targets,
        classification_targets,
        masks,
        assignment_outputs=None,
        assignment_targets=None,
        hlr_outputs=None,
        hlr_targets=None,
    ):
        losses = {}

        reg_target_keys = {"leptons": "leptons", "neutrinos": "neutrinos",
                           "jets": "jets", "tops": "tops", "ws": "ws"}
        total_regression_loss = 0.0
        for name in reg_target_keys:
            pred = regression_outputs[name]
            tgt = regression_targets[name]
            mask = masks[name]
            per_pat_mse = self._masked_mse(pred, tgt, mask)
            for i in range(per_pat_mse.size(0)):
                losses[f"{name}_{i}_mse"] = per_pat_mse[i]
            total_regression_loss = total_regression_loss + per_pat_mse.sum()
        losses["regression"] = total_regression_loss

        cls_mapping = {
            "leptons": ("lepton_logits", "lepton_classes"),
            "neutrinos": ("neutrino_logits", "neutrino_classes"),
            "jets": ("jet_logits", "jet_classes"),
            "tops": ("top_logits", "top_classes"),
            "ws": ("w_logits", "w_classes"),
        }
        total_classification_loss = 0.0
        for name, (logit_key, tgt_key) in cls_mapping.items():
            logits = classification_outputs[logit_key]
            tgt = classification_targets[tgt_key]
            bs, ni, c_i = logits.shape
            flat_logits = logits.view(bs * ni, c_i)
            flat_tgt = tgt.view(bs * ni)
            ce_elem = self.ce_loss(flat_logits, flat_tgt)
            per_pat_ce = ce_elem.view(bs, ni).mean(dim=0)
            for i in range(ni):
                losses[f"{name}_{i}_ce"] = per_pat_ce[i]
            total_classification_loss = total_classification_loss + per_pat_ce.sum()
        losses["classification"] = total_classification_loss

        total_assignment_loss = torch.tensor(0.0, device=total_regression_loss.device)
        if assignment_outputs is not None and assignment_targets is not None:
            for key in assignment_outputs:
                logits = assignment_outputs[key]
                tgt = assignment_targets[key]
                bs, ni, c_i = logits.shape
                flat_logits = logits.view(bs * ni, c_i)
                flat_tgt = tgt.view(bs * ni)
                ce_elem = self.ce_loss(flat_logits, flat_tgt)
                per_slot_ce = ce_elem.view(bs, ni).mean(dim=0)
                for i in range(ni):
                    losses[f"assign_{key}_{i}_ce"] = per_slot_ce[i]
                total_assignment_loss = total_assignment_loss + per_slot_ce.sum()
        losses["assignment"] = total_assignment_loss

        total_hlr_loss = torch.tensor(0.0, device=total_regression_loss.device)
        if hlr_outputs is not None and hlr_targets is not None:
            hlr_mask = masks.get("hlr", None)
            if hlr_mask is not None:
                hlr_mse = self.mse_loss(hlr_outputs, hlr_targets)
                hlr_mask_f = hlr_mask.float()
                hlr_mse = (hlr_mse * hlr_mask_f).sum() / hlr_mask_f.sum().clamp(min=1.0)
            else:
                hlr_mse = self.mse_loss(hlr_outputs, hlr_targets).mean()
            total_hlr_loss = hlr_mse
        losses["hlr"] = total_hlr_loss

        task_losses = [total_regression_loss, total_classification_loss,
                       total_assignment_loss, total_hlr_loss]
        total_loss = 0.0
        for i, tl in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * tl + self.log_vars[i]

        losses["total"] = total_loss

        for i, name in enumerate(["regression", "classification", "assignment", "hlr"]):
            losses[f"task_weight/{name}"] = torch.exp(-self.log_vars[i]).detach()

        return losses


# ---------------------------------------------------------------------------
# Physics-informed mass loss helpers
# ---------------------------------------------------------------------------

# PDG 2022 masses in GeV
_M_W_GEV: float = 80.377
_M_TOP_GEV: float = 172.76


def _v5_to_cartesian(
    reg_v5: torch.Tensor,
    reg_mean: torch.Tensor,
    reg_std: torch.Tensor,
    signed_energy: bool = False,
) -> torch.Tensor:
    """Convert z-scored V5 regression output to Cartesian 4-momenta (E, px, py, pz) in GeV.

    Args:
        reg_v5:   (B, N, 5)  z-scored (log1p(pT)_n, eta_n, sin_phi, cos_phi,
                  log1p(E)_n or signed_log1p(E)_n)
        reg_mean: (N, 3)     per-slot means for (pT, eta, E) in this function's own
                              5-dim V5-representation space — columns [0,1,4] of
                              ``reg_v5``, NOT the raw 7-column cache layout used by
                              ``lightning.py``/``lightning_gatr.py`` (there the same
                              words "column 3"/"[0,1,3]" index a different space —
                              see the regression contract).
        reg_std:  (N, 3)     per-slot stds for (pT, eta, E), same column space as reg_mean

    pT is stored log1p-scaled. When ``signed_energy`` is set, E uses
    signed-log1p; otherwise it uses the legacy log1p parameterisation. Both are
    un-z-scored and inverted to recover physical GeV before use.

    Returns:
        (B, N, 4)  (E, px, py, pz) in GeV
    """
    # Un-z-score pT (dim 0), eta (dim 1), E (dim 4), then invert their transforms.
    pT  = torch.expm1(reg_v5[..., 0] * reg_std[..., 0] + reg_mean[..., 0])   # (B, N)
    eta = reg_v5[..., 1] * reg_std[..., 1] + reg_mean[..., 1]                # (B, N)
    # sin_phi / cos_phi are already actual values (not z-scored)
    sin_phi = reg_v5[..., 2]
    cos_phi = reg_v5[..., 3]
    e_signed_log = reg_v5[..., 4] * reg_std[..., 2] + reg_mean[..., 2]
    E = (
        torch.sign(e_signed_log) * torch.expm1(e_signed_log.abs())
        if signed_energy
        else torch.expm1(e_signed_log)
    )

    # Cartesian momenta
    px = pT * cos_phi
    py = pT * sin_phi
    # pz = pT * sinh(eta)
    pz = pT * torch.sinh(eta)

    return torch.stack([E, px, py, pz], dim=-1)  # (B, N, 4)


def reg_target_to_v5(reg_tgt_raw, phi_mean=None, phi_std=None):
    """Convert a raw z-scored regression target to the V5 (sin/cos) representation.

    Input layout (last dim == 7): ``(pT_n, eta_n, phi_n, px_n, py_n, pz_n, E_n)``
    — the raw cache column order; ``E_n`` is column 6, not column 3 (=px_n), see
    the regression contract.  Output:
    ``(pT_n, eta_n, sin_phi, cos_phi, E_n)`` — the exact representation the
    V5/VX models are trained against.  ``phi_mean``/``phi_std`` are the per-slot
    phi normalisation stats used to recover the physical angle from the z-scored
    ``phi_n`` before sin/cos (``None`` leaves the corresponding step out, matching
    the un-normalised default).

    Single source of truth shared by the training target prep
    (``PatEmbedderLightningV5._prepare_targets``) and the eval permutation-matching
    cost (``evaluation._get_decoder_perms``), so eval-matching is computed in the
    same space as the training cost.
    """
    pt_eta = reg_tgt_raw[..., :2]
    phi_actual = reg_tgt_raw[..., 2]
    e_n = reg_tgt_raw[..., 6:7]
    if phi_std is not None:
        phi_actual = phi_actual * phi_std
    if phi_mean is not None:
        phi_actual = phi_actual + phi_mean
    sin_phi = torch.sin(phi_actual).unsqueeze(-1)
    cos_phi = torch.cos(phi_actual).unsqueeze(-1)
    return torch.cat([pt_eta, sin_phi, cos_phi, e_n], dim=-1)


def _soft_select_p4(
    logits: torch.Tensor,
    p4_keys: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable soft selection of 4-momenta using assignment logits.

    Args:
        logits:   (B, N_q, N_k + 1)  assignment logits; last column = no-match
        p4_keys:  (B, N_k, 4)        4-momenta of candidate particles
        temperature: softmax temperature (default 1.0)

    Returns:
        (B, N_q, 4)  soft-selected 4-momenta (weighted sum over candidates)
    """
    # Exclude the no-match column before softmax so weights sum to 1 over real candidates
    weights = F.softmax(logits[..., :-1] / temperature, dim=-1)  # (B, N_q, N_k)
    # Weighted sum: (B, N_q, N_k) x (B, N_k, 4) -> (B, N_q, 4)
    return torch.bmm(weights, p4_keys)


def _lorentz_mass_sq(p4: torch.Tensor) -> torch.Tensor:
    """Compute M^2 = E^2 - px^2 - py^2 - pz^2 for a 4-momentum tensor.

    Args:
        p4: (..., 4)  (E, px, py, pz)

    Returns:
        (...,)  M^2 (can be negative for off-shell / unphysical predictions)
    """
    return p4[..., 0].pow(2) - p4[..., 1:].pow(2).sum(dim=-1)


def _mass_huber(m_sq: torch.Tensor, m_target_gev: float, delta: float = 20.0) -> torch.Tensor:
    """Huber loss on reconstructed invariant mass vs target PDG mass.

    Uses M (not M^2) to avoid gradient explosion for large deviations.
    Clamps M^2 >= 0 before sqrt to handle unphysical predictions gracefully.

    Args:
        m_sq:          (...,)  M^2 in GeV^2
        m_target_gev:  float   target mass in GeV
        delta:         float   Huber delta in GeV (default 20 GeV, about 25% of M_W)

    Returns:
        (...,)  per-element Huber loss
    """
    m_pred = torch.sqrt(m_sq.clamp(min=1e-4))  # (...)  GeV
    return F.huber_loss(m_pred, torch.full_like(m_pred, m_target_gev), delta=delta, reduction="none")


# ---------------------------------------------------------------------------
# Loss V5 (sin/cos phi + Huber pT + presence BCE + fixed task weights + aux)
# ---------------------------------------------------------------------------

class PatEmbedderLossV5(nn.Module):
    """V5 loss:
    - Regression: Huber on pT (dim 0) + MSE on remaining dims.
    - Classification: focal loss.
    - Assignment: focal loss with permutation matching; no masking for top-to-W and top-to-bjet.
    - Presence: BCEWithLogitsLoss on top_presence / w_presence logits vs masks.
    - Fixed task weights + deep supervision aux losses.
    - Physics-informed mass loss: soft-assignment invariant mass constraints at
      each decay vertex (W->jj, W->lnu, t->Wb).
    """

    def __init__(
        self,
        gamma=2.0,
        max_tops=4,
        max_ws=4,
        task_weights=None,
        aux_loss_weight=0.5,
        huber_delta=1.0,
        presence_loss_weight=1.0,
        dn_loss_weight=1.0,
        mass_loss_weight=0.0,
        mass_huber_delta=20.0,
        phi_norm_weight=0.0,
        phi_angle_weight=0.0,
        cls_empty_boost=1.0,
        signed_energy=False,
    ):
        super().__init__()
        self.gamma = gamma
        self.max_tops = max_tops
        self.max_ws = max_ws
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        if task_weights is None:
            task_weights = [1.0, 1.0, 2.0, 0.5]
        self.register_buffer("task_weights", torch.tensor(task_weights, dtype=torch.float32))
        self.aux_loss_weight = aux_loss_weight
        self.huber_delta = huber_delta
        self.presence_loss_weight = presence_loss_weight
        self.dn_loss_weight = dn_loss_weight
        self.mass_loss_weight = mass_loss_weight
        self.mass_huber_delta = mass_huber_delta
        self.phi_norm_weight = phi_norm_weight
        self.phi_angle_weight = phi_angle_weight
        self.cls_empty_boost = cls_empty_boost
        self.signed_energy = signed_energy

    def _masked_regression(self, pred, tgt, mask):
        pt_loss = F.huber_loss(
            pred[:, :, 0], tgt[:, :, 0], delta=self.huber_delta, reduction="none"
        )
        other_loss = F.mse_loss(pred[:, :, 1:], tgt[:, :, 1:], reduction="none").mean(dim=-1)
        mask_f = mask.float()
        loss = (pt_loss + other_loss) * mask_f

        if self.phi_angle_weight > 0.0 and pred.shape[-1] >= 4:
            pred_phi = torch.atan2(pred[:, :, 2], pred[:, :, 3])
            tgt_phi = torch.atan2(tgt[:, :, 2], tgt[:, :, 3])
            phi_angle_loss = 1.0 - torch.cos(pred_phi - tgt_phi)
            loss = loss + self.phi_angle_weight * phi_angle_loss * mask_f

        # sin^2(phi) + cos^2(phi) = 1 soft constraint for V5 reg_dim=5 outputs.
        if self.phi_norm_weight > 0.0 and pred.shape[-1] >= 4:
            norm_sq = pred[:, :, 2].pow(2) + pred[:, :, 3].pow(2)  # (B, N)
            phi_norm_loss = (norm_sq - 1.0).pow(2) * mask_f         # (B, N)
            loss = loss + self.phi_norm_weight * phi_norm_loss

        n_valid = mask_f.sum(dim=0).clamp(min=1.0)
        return loss.sum(dim=0) / n_valid

    def _reg_cost_per_slot(self, pred, tgt):
        pt_cost = F.huber_loss(
            pred[:, :, 0], tgt[:, :, 0], delta=self.huber_delta, reduction="none"
        )
        other_cost = F.mse_loss(pred[:, :, 1:], tgt[:, :, 1:], reduction="none").mean(dim=-1)
        return pt_cost + other_cost

    @torch.no_grad()
    def _w_pairwise_cost(
        self, cls_out, reg_out, assign_out, cls_tgt, reg_tgt, assign_tgt, w_mask, reg_mean=None, reg_std=None
    ):
        B, Nw = cls_out.size(0), self.max_ws
        device = cls_out.device
        cost = torch.zeros(B, Nw, Nw, device=device)
        pairwise_reg = _target_in_query_normalization(reg_tgt, reg_mean, reg_std) if reg_mean is not None else None
        for j in range(Nw):
            tgt_j_cls = cls_tgt[:, j:j + 1].expand(B, Nw)
            cost[:, :, j] += _focal_ce_per_slot(cls_out, tgt_j_cls, self.gamma)

            tgt_j_reg = pairwise_reg[:, :, j] if pairwise_reg is not None else reg_tgt[:, j:j + 1].expand(B, Nw, -1)
            cost[:, :, j] += self._reg_cost_per_slot(reg_out, tgt_j_reg) * w_mask[:, j:j + 1].float()

            for key in ("w_to_lepton", "w_to_neutrino"):
                t = assign_tgt[key][:, j:j + 1].expand(B, Nw)
                cost[:, :, j] += _focal_ce_per_slot(assign_out[key], t, self.gamma)

            t1 = assign_tgt["w_to_jet1"][:, j:j + 1].expand(B, Nw)
            t2 = assign_tgt["w_to_jet2"][:, j:j + 1].expand(B, Nw)
            p1 = (
                _focal_ce_per_slot(assign_out["w_to_jet1"], t1, self.gamma)
                + _focal_ce_per_slot(assign_out["w_to_jet2"], t2, self.gamma)
            )
            p2 = (
                _focal_ce_per_slot(assign_out["w_to_jet1"], t2, self.gamma)
                + _focal_ce_per_slot(assign_out["w_to_jet2"], t1, self.gamma)
            )
            cost[:, :, j] += torch.min(p1, p2)
        return cost

    @torch.no_grad()
    def _top_pairwise_cost(
        self, cls_out, reg_out, assign_out, cls_tgt, reg_tgt, assign_tgt, top_mask, reg_mean=None, reg_std=None
    ):
        B, Nt = cls_out.size(0), self.max_tops
        device = cls_out.device
        cost = torch.zeros(B, Nt, Nt, device=device)
        pairwise_reg = _target_in_query_normalization(reg_tgt, reg_mean, reg_std) if reg_mean is not None else None
        for j in range(Nt):
            tgt_j_cls = cls_tgt[:, j:j + 1].expand(B, Nt)
            cost[:, :, j] += _focal_ce_per_slot(cls_out, tgt_j_cls, self.gamma)

            tgt_j_reg = pairwise_reg[:, :, j] if pairwise_reg is not None else reg_tgt[:, j:j + 1].expand(B, Nt, -1)
            cost[:, :, j] += self._reg_cost_per_slot(reg_out, tgt_j_reg) * top_mask[:, j:j + 1].float()

            for key in ("top_to_bjet", "top_to_w"):
                t = assign_tgt[key][:, j:j + 1].expand(B, Nt)
                cost[:, :, j] += _focal_ce_per_slot(assign_out[key], t, self.gamma)
        return cost

    def forward(
        self,
        regression_outputs,
        classification_outputs,
        regression_targets,
        classification_targets,
        masks,
        assignment_outputs=None,
        assignment_targets=None,
        hlr_outputs=None,
        hlr_targets=None,
        aux_head_outputs=None,
        dn_outputs=None,
        dn_targets_reg=None,
        dn_targets_cls=None,
        dn_masks=None,
        reg_means=None,
        reg_stds=None,
        matching_reg_means=None,
        matching_reg_stds=None,
    ):
        if (
            (aux_head_outputs is not None and self.aux_loss_weight > 0.0)
            or dn_outputs is not None
            or reg_means is not None
            or reg_stds is not None
        ):
            return self.forward_with_aux(
                regression_outputs,
                classification_outputs,
                regression_targets,
                classification_targets,
                masks,
                assignment_outputs,
                assignment_targets,
                hlr_outputs,
                hlr_targets,
                aux_head_outputs=aux_head_outputs,
                dn_outputs=dn_outputs,
                dn_targets_reg=dn_targets_reg,
                dn_targets_cls=dn_targets_cls,
                dn_masks=dn_masks,
                reg_means=reg_means,
                reg_stds=reg_stds,
                matching_reg_means=matching_reg_means,
                matching_reg_stds=matching_reg_stds,
            )

        losses = {}
        device = next(iter(regression_outputs.values())).device
        if (matching_reg_means is None) != (matching_reg_stds is None):
            raise ValueError("matching_reg_means and matching_reg_stds must be supplied together")
        if matching_reg_means is not None:
            for key in ("tops", "ws"):
                mean, std = matching_reg_means[key], matching_reg_stds[key]
                if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                    raise ValueError(f"matching normalization for {key} must be finite with positive std")

        regression_targets = {**regression_targets}
        classification_targets = {**classification_targets}
        masks = {**masks}

        top_presence_logits = None
        w_presence_logits = None
        if assignment_outputs is not None:
            assignment_outputs = {**assignment_outputs}
            top_presence_logits = assignment_outputs.pop("top_presence", None)
            w_presence_logits = assignment_outputs.pop("w_presence", None)

        if assignment_outputs is not None and assignment_targets is not None:
            assignment_targets = {**assignment_targets}
            B = classification_outputs["w_logits"].size(0)

            w_cls_tgt = classification_targets["w_classes"].view(B, self.max_ws)
            w_cost = self._w_pairwise_cost(
                classification_outputs["w_logits"],
                regression_outputs["ws"],
                assignment_outputs,
                w_cls_tgt,
                regression_targets["ws"],
                assignment_targets,
                masks["ws"],
                *( (matching_reg_means["ws"], matching_reg_stds["ws"]) if matching_reg_means is not None else () ),
            )
            w_perm = _find_best_permutation(w_cost, self.max_ws)
            w_perm_inv = _invert_perm(w_perm)
            classification_targets["w_classes"] = _apply_perm_1d(w_cls_tgt, w_perm).view(-1)
            regression_targets["ws"] = _apply_perm_with_query_normalization(
                regression_targets["ws"], w_perm, matching_reg_means["ws"], matching_reg_stds["ws"]
            ) if matching_reg_means is not None else _apply_perm_2d(regression_targets["ws"], w_perm)
            masks["ws"] = _apply_perm_1d(masks["ws"], w_perm)
            for key in ("w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2"):
                assignment_targets[key] = _apply_perm_1d(assignment_targets[key], w_perm)
            assignment_targets["top_to_w"] = _remap_assignment_indices(
                assignment_targets["top_to_w"], w_perm_inv, no_match_idx=self.max_ws
            )

            top_cls_tgt = classification_targets["top_classes"].view(B, self.max_tops)
            top_cost = self._top_pairwise_cost(
                classification_outputs["top_logits"],
                regression_outputs["tops"],
                assignment_outputs,
                top_cls_tgt,
                regression_targets["tops"],
                assignment_targets,
                masks["tops"],
                *( (matching_reg_means["tops"], matching_reg_stds["tops"]) if matching_reg_means is not None else () ),
            )
            top_perm = _find_best_permutation(top_cost, self.max_tops)
            classification_targets["top_classes"] = _apply_perm_1d(top_cls_tgt, top_perm).view(-1)
            regression_targets["tops"] = _apply_perm_with_query_normalization(
                regression_targets["tops"], top_perm, matching_reg_means["tops"], matching_reg_stds["tops"]
            ) if matching_reg_means is not None else _apply_perm_2d(regression_targets["tops"], top_perm)
            masks["tops"] = _apply_perm_1d(masks["tops"], top_perm)
            for key in ("top_to_w", "top_to_bjet"):
                assignment_targets[key] = _apply_perm_1d(assignment_targets[key], top_perm)

        total_regression_loss = 0.0
        for name in ("leptons", "neutrinos", "jets", "tops", "ws"):
            pred, tgt, mask = regression_outputs[name], regression_targets[name], masks[name]
            per_pat = self._masked_regression(pred, tgt, mask)
            for i in range(per_pat.size(0)):
                losses[f"{name}_{i}_reg"] = per_pat[i]
            total_regression_loss = total_regression_loss + per_pat.sum()
        losses["regression"] = total_regression_loss

        cls_mapping = {
            "leptons": ("lepton_logits", "lepton_classes"),
            "neutrinos": ("neutrino_logits", "neutrino_classes"),
            "jets": ("jet_logits", "jet_classes"),
            "tops": ("top_logits", "top_classes"),
            "ws": ("w_logits", "w_classes"),
        }
        total_classification_loss = 0.0
        for name, (logit_key, tgt_key) in cls_mapping.items():
            logits = classification_outputs[logit_key]
            tgt = classification_targets[tgt_key]
            bs, ni, c_i = logits.shape
            cls_w = None
            if self.cls_empty_boost != 1.0:
                cls_w = torch.ones(c_i, device=device)
                cls_w[0] = self.cls_empty_boost
            fl = focal_ce_loss(
                logits.view(bs * ni, c_i), tgt.view(bs * ni),
                gamma=self.gamma, cls_weights=cls_w,
            )
            per_pat_fl = fl.view(bs, ni).mean(dim=0)
            for i in range(ni):
                losses[f"{name}_{i}_ce"] = per_pat_fl[i]
            total_classification_loss = total_classification_loss + per_pat_fl.sum()
        losses["classification"] = total_classification_loss

        total_assignment_loss = torch.tensor(0.0, device=device)
        if assignment_outputs is not None and assignment_targets is not None:
            slot_mask_map = {
                "top_to_w": None,
                "top_to_bjet": None,
                "w_to_lepton": masks["ws"],
                "w_to_neutrino": masks["ws"],
                "w_to_jet1": masks["ws"],
                "w_to_jet2": masks["ws"],
            }

            jet_done = False
            for key in assignment_outputs:
                logits = assignment_outputs[key]
                tgt = assignment_targets[key]
                bs, ni, c_i = logits.shape
                smask = slot_mask_map.get(key, None)

                if key in ("w_to_jet1", "w_to_jet2"):
                    if jet_done:
                        continue
                    jet_done = True
                    j1_l = assignment_outputs["w_to_jet1"]
                    j2_l = assignment_outputs["w_to_jet2"]
                    j1_t = assignment_targets["w_to_jet1"]
                    j2_t = assignment_targets["w_to_jet2"]
                    cost_p1 = _focal_ce_per_slot(j1_l, j1_t, self.gamma) + _focal_ce_per_slot(j2_l, j2_t, self.gamma)
                    cost_p2 = _focal_ce_per_slot(j1_l, j2_t, self.gamma) + _focal_ce_per_slot(j2_l, j1_t, self.gamma)
                    jet_loss = torch.min(cost_p1, cost_p2)
                    if smask is not None:
                        smask_f = smask.float()
                        jet_loss = jet_loss * smask_f
                        n_valid = smask_f.sum(dim=0).clamp(min=1.0)
                        per_slot_fl = jet_loss.sum(dim=0) / n_valid
                    else:
                        per_slot_fl = jet_loss.mean(dim=0)
                    for i in range(ni):
                        losses[f"assign_w_jets_{i}_ce"] = per_slot_fl[i]
                    total_assignment_loss = total_assignment_loss + per_slot_fl.sum()
                    continue

                fl = focal_ce_loss(
                    logits.view(bs * ni, c_i), tgt.view(bs * ni), gamma=self.gamma
                ).view(bs, ni)
                if smask is not None:
                    smask_f = smask.float()
                    fl = fl * smask_f
                    n_valid = smask_f.sum(dim=0).clamp(min=1.0)
                    per_slot_fl = fl.sum(dim=0) / n_valid
                else:
                    per_slot_fl = fl.mean(dim=0)
                for i in range(ni):
                    losses[f"assign_{key}_{i}_ce"] = per_slot_fl[i]
                total_assignment_loss = total_assignment_loss + per_slot_fl.sum()

        total_presence_loss = torch.tensor(0.0, device=device)
        if top_presence_logits is not None:
            top_pres_tgt = masks["tops"].float()
            top_pres_loss = self.bce_loss(top_presence_logits, top_pres_tgt).mean()
            losses["top_presence_bce"] = top_pres_loss
            total_presence_loss = total_presence_loss + top_pres_loss
        if w_presence_logits is not None:
            w_pres_tgt = masks["ws"].float()
            w_pres_loss = self.bce_loss(w_presence_logits, w_pres_tgt).mean()
            losses["w_presence_bce"] = w_pres_loss
            total_presence_loss = total_presence_loss + w_pres_loss
        total_assignment_loss = total_assignment_loss + self.presence_loss_weight * total_presence_loss
        losses["assignment"] = total_assignment_loss

        total_hlr_loss = torch.tensor(0.0, device=device)
        if hlr_outputs is not None and hlr_targets is not None:
            hlr_mask = masks.get("hlr", None)
            if hlr_mask is not None:
                hlr_mse = self.mse_loss(hlr_outputs, hlr_targets)
                hlr_mask_f = hlr_mask.float()
                hlr_mse = (hlr_mse * hlr_mask_f).sum() / hlr_mask_f.sum().clamp(min=1.0)
            else:
                hlr_mse = self.mse_loss(hlr_outputs, hlr_targets).mean()
            total_hlr_loss = hlr_mse
        losses["hlr"] = total_hlr_loss

        task_losses = [
            total_regression_loss,
            total_classification_loss,
            total_assignment_loss,
            total_hlr_loss,
        ]
        total_loss = torch.tensor(0.0, device=device)
        for i, tl in enumerate(task_losses):
            total_loss = total_loss + self.task_weights[i] * tl
        losses["total"] = total_loss
        for i, nm in enumerate(["regression", "classification", "assignment", "hlr"]):
            losses[f"task_weight/{nm}"] = self.task_weights[i].detach()
        return losses

    def _dn_loss(self, dn_outputs, dn_targets_reg, dn_targets_cls, dn_masks):
        """Direct supervision for DN-DETR: noised slot i -> target slot i.

        No permutation matching, no jet-pair symmetry. Regression slots use the
        same Huber+MSE recipe as the main path, masked by per-slot presence.
        Classification uses focal CE on every slot (so the model learns to
        predict the empty class at absent positions).
        """
        device = dn_outputs["top_reg"].device
        top_reg_loss = self._masked_regression(
            dn_outputs["top_reg"], dn_targets_reg["tops"], dn_masks["tops"]
        ).sum()
        w_reg_loss = self._masked_regression(
            dn_outputs["w_reg"], dn_targets_reg["ws"], dn_masks["ws"]
        ).sum()

        top_logits = dn_outputs["top_cls"]
        w_logits = dn_outputs["w_cls"]
        bs_t, nt, ct = top_logits.shape
        bs_w, nw, cw = w_logits.shape
        top_cls_loss = focal_ce_loss(
            top_logits.view(bs_t * nt, ct),
            dn_targets_cls["top_classes"].view(bs_t * nt),
            gamma=self.gamma,
        ).view(bs_t, nt).mean(dim=0).sum()
        w_cls_loss = focal_ce_loss(
            w_logits.view(bs_w * nw, cw),
            dn_targets_cls["w_classes"].view(bs_w * nw),
            gamma=self.gamma,
        ).view(bs_w, nw).mean(dim=0).sum()

        return torch.stack([top_reg_loss, w_reg_loss, top_cls_loss, w_cls_loss]).sum().to(device)

    def _physics_mass_loss(
        self,
        regression_outputs: dict,
        assignment_outputs: dict,
        masks: dict,
        reg_means: dict,
        reg_stds: dict,
    ) -> torch.Tensor:
        """Physics-informed invariant mass constraints at each decay vertex.

        Converts z-scored V5 regression outputs to physical 4-momenta, then
        uses soft assignment (differentiable softmax over logits) to reconstruct
        invariant masses at each decay vertex and penalises deviations from PDG
        values via Huber loss.

        Constraints applied:
          - W -> jj:   M(j1, j2)     ~= M_W  (hadronic W decay)
          - W -> lnu:  M(l, nu)      ~= M_W  (leptonic W decay)
          - t -> Wb:   M(W_reco, b)  ~= M_t  (top quark mass)

        Args:
            regression_outputs: dict with keys "leptons", "neutrinos", "jets",
                "tops", "ws" - each (B, N, 5) z-scored V5 predictions.
            assignment_outputs: dict with keys "w_to_jet1", "w_to_jet2",
                "w_to_lepton", "w_to_neutrino", "top_to_bjet", "top_to_w" -
                each (B, N_q, N_k+1) logits.
            masks: dict with keys "ws", "tops" - (B, N) bool tensors.
            reg_means: dict mapping particle type to (N, 3) tensor of per-slot
                means for (pT, eta, E).  Keys: "leptons", "neutrinos", "jets",
                "tops", "ws".
            reg_stds: same structure as reg_means, per-slot stds.

        Returns:
            Scalar mass loss (mean over batch and present slots).
        """
        device = regression_outputs["jets"].device
        delta = self.mass_huber_delta

        # Convert all particle types to physical Cartesian 4-momenta
        def _to_p4(key: str) -> torch.Tensor:
            return _v5_to_cartesian(
                regression_outputs[key],
                reg_means[key].to(device),
                reg_stds[key].to(device),
                signed_energy=self.signed_energy,
            )

        lep_p4 = _to_p4("leptons")    # (B, 4, 4)
        neu_p4 = _to_p4("neutrinos")  # (B, 4, 4)
        jet_p4 = _to_p4("jets")       # (B, 12, 4)

        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        # W -> jj mass constraint
        if "w_to_jet1" in assignment_outputs and "w_to_jet2" in assignment_outputs:
            j1_p4 = _soft_select_p4(assignment_outputs["w_to_jet1"], jet_p4)  # (B, Nw, 4)
            j2_p4 = _soft_select_p4(assignment_outputs["w_to_jet2"], jet_p4)  # (B, Nw, 4)
            w_jj_p4 = j1_p4 + j2_p4                                           # (B, Nw, 4)
            m_sq_w_jj = _lorentz_mass_sq(w_jj_p4)                             # (B, Nw)
            loss_w_jj = _mass_huber(m_sq_w_jj, _M_W_GEV, delta)               # (B, Nw)
            # Mask: only penalise W slots that are present
            w_mask_f = masks["ws"].float().to(device)                          # (B, Nw)
            n_valid_w = w_mask_f.sum().clamp(min=1.0)
            total_loss = total_loss + (loss_w_jj * w_mask_f).sum() / n_valid_w
            n_terms += 1

        # W -> lnu mass constraint
        if "w_to_lepton" in assignment_outputs and "w_to_neutrino" in assignment_outputs:
            l_p4  = _soft_select_p4(assignment_outputs["w_to_lepton"],   lep_p4)  # (B, Nw, 4)
            nu_p4 = _soft_select_p4(assignment_outputs["w_to_neutrino"], neu_p4)  # (B, Nw, 4)
            w_lnu_p4 = l_p4 + nu_p4                                               # (B, Nw, 4)
            m_sq_w_lnu = _lorentz_mass_sq(w_lnu_p4)                               # (B, Nw)
            loss_w_lnu = _mass_huber(m_sq_w_lnu, _M_W_GEV, delta)                 # (B, Nw)
            w_mask_f = masks["ws"].float().to(device)
            n_valid_w = w_mask_f.sum().clamp(min=1.0)
            total_loss = total_loss + (loss_w_lnu * w_mask_f).sum() / n_valid_w
            n_terms += 1

        # t -> Wb mass constraint
        # Reconstruct W 4-momentum from its daughters (hadronic or leptonic),
        # then combine with the b-jet to get the top mass.
        if (
            "top_to_w" in assignment_outputs
            and "top_to_bjet" in assignment_outputs
            and "w_to_jet1" in assignment_outputs
            and "w_to_jet2" in assignment_outputs
        ):
            # Soft-select which W each top is associated with
            # w_reg_p4: (B, Nw, 4) - W 4-momenta from hadronic decay
            j1_p4_h = _soft_select_p4(assignment_outputs["w_to_jet1"], jet_p4)  # (B, Nw, 4)
            j2_p4_h = _soft_select_p4(assignment_outputs["w_to_jet2"], jet_p4)  # (B, Nw, 4)
            w_p4_had = j1_p4_h + j2_p4_h                                        # (B, Nw, 4)

            # Soft-select which W each top points to -> weighted sum of W 4-momenta
            # top_to_w logits: (B, Nt, Nw+1); w_p4_had: (B, Nw, 4)
            top_w_p4 = _soft_select_p4(assignment_outputs["top_to_w"], w_p4_had)  # (B, Nt, 4)

            # Soft-select b-jet for each top
            top_b_p4 = _soft_select_p4(assignment_outputs["top_to_bjet"], jet_p4)  # (B, Nt, 4)

            top_reco_p4 = top_w_p4 + top_b_p4                                      # (B, Nt, 4)
            m_sq_top = _lorentz_mass_sq(top_reco_p4)                               # (B, Nt)
            loss_top = _mass_huber(m_sq_top, _M_TOP_GEV, delta)                    # (B, Nt)
            top_mask_f = masks["tops"].float().to(device)                           # (B, Nt)
            n_valid_t = top_mask_f.sum().clamp(min=1.0)
            total_loss = total_loss + (loss_top * top_mask_f).sum() / n_valid_t
            n_terms += 1

        return total_loss / max(n_terms, 1)

    def forward_with_aux(
        self,
        regression_outputs,
        classification_outputs,
        regression_targets,
        classification_targets,
        masks,
        assignment_outputs=None,
        assignment_targets=None,
        hlr_outputs=None,
        hlr_targets=None,
        aux_head_outputs=None,
        dn_outputs=None,
        dn_targets_reg=None,
        dn_targets_cls=None,
        dn_masks=None,
        reg_means=None,
        reg_stds=None,
        matching_reg_means=None,
        matching_reg_stds=None,
    ):
        losses = self.forward(
            regression_outputs,
            classification_outputs,
            regression_targets,
            classification_targets,
            masks,
            assignment_outputs,
            assignment_targets,
            hlr_outputs,
            hlr_targets,
            matching_reg_means=matching_reg_means,
            matching_reg_stds=matching_reg_stds,
        )
        if (
            self.aux_loss_weight > 0.0
            and aux_head_outputs is not None
            and len(aux_head_outputs) > 0
        ):
            n_aux = len(aux_head_outputs)
            aux_total = torch.tensor(0.0, device=losses["total"].device)
            for layer_i, aux in enumerate(aux_head_outputs):
                decay = self.aux_loss_weight ** (n_aux - layer_i)
                aux_reg_out = {**regression_outputs, "tops": aux["top_reg"], "ws": aux["w_reg"]}
                aux_cls_out = {**classification_outputs, "top_logits": aux["top_cls"], "w_logits": aux["w_cls"]}
                aux_losses = self.forward(
                    aux_reg_out,
                    aux_cls_out,
                    regression_targets,
                    classification_targets,
                    masks,
                    aux["assign"],
                    assignment_targets,
                    hlr_outputs,
                    hlr_targets,
                    matching_reg_means=matching_reg_means,
                    matching_reg_stds=matching_reg_stds,
                )
                aux_total = aux_total + decay * aux_losses["total"]
            losses["aux"] = aux_total
            losses["total"] = losses["total"] + aux_total

        if (
            dn_outputs is not None
            and dn_targets_reg is not None
            and dn_targets_cls is not None
            and dn_masks is not None
        ):
            dn_loss = self._dn_loss(dn_outputs, dn_targets_reg, dn_targets_cls, dn_masks)
            losses["dn"] = dn_loss
            losses["total"] = losses["total"] + self.dn_loss_weight * dn_loss

        if (
            self.mass_loss_weight > 0.0
            and assignment_outputs is not None
            and reg_means is not None
            and reg_stds is not None
        ):
            mass_loss = self._physics_mass_loss(
                regression_outputs, assignment_outputs, masks, reg_means, reg_stds
            )
            losses["mass"] = mass_loss
            losses["total"] = losses["total"] + self.mass_loss_weight * mass_loss

        return losses


