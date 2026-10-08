from itertools import permutations as _iperms

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .data import jet_class_labels
from .losses import _target_in_query_normalization, reg_target_to_v5

DEFAULT_LEPTON_LABEL_MAP = {0: 0, -11: 1, 11: 2, -13: 3, 13: 4, -15: 5, 15: 6}
DEFAULT_NEUTRINO_LABEL_MAP = {0: 0, 12: 1, -12: 2, 14: 3, -14: 4, 16: 5, -16: 6}
DEFAULT_TOP_LABEL_MAP = {0: 0, 6: 1, -6: 2}
DEFAULT_W_LABEL_MAP = {0: 0, 24: 1, -24: 2}

_REGRESSION_DEFAULT_CLS_MAP = {
    "leptons": DEFAULT_LEPTON_LABEL_MAP,
    "neutrinos": DEFAULT_NEUTRINO_LABEL_MAP,
    "tops": DEFAULT_TOP_LABEL_MAP,
    "ws": DEFAULT_W_LABEL_MAP,
}


def _call_model(model, inputs, batch=None):
    if batch is not None and getattr(model, "accepts_masks", False):
        return model(inputs, masks=batch.get("masks"))
    return model(inputs)


def _uses_signed_v5_energy(model) -> bool:
    """Whether this model's energy output follows the signed VX contract (v3+).

    The single place that knows how the inner ``nn.Module`` — the one carrying
    the contract buffer — may be wrapped. Callers reach the evaluation helpers
    holding one of three objects: the module itself, a LightningModule (inner
    module at ``.model``) or an eval wrapper from
    ``scripts/infer_and_plot.py`` (inner module at ``._model``). Looking up one
    name only silently decodes contract-v3 energy as unsigned on every path
    that uses the other — see the regression contract B1.

    The version itself lives in two buffers: ``_vx_full_contract_version`` on
    the inner module (read here) and ``_gatr_output_contract_version`` on the
    LightningModule. They are compared with each other in one place only —
    ``lightning_gatr.PatEmbedderLightningVXFull.on_load_checkpoint`` — which
    refuses anything but exactly 3, so a loaded checkpoint never reaches the
    ``> 3`` half of the comparison below. It stays ordered so that a future
    contract keeps the signed decode instead of silently falling back to the
    legacy one (the regression contract m6).
    """
    for candidate in (model, getattr(model, "_model", None), getattr(model, "model", None)):
        version = getattr(candidate, "_vx_full_contract_version", None)
        if version is not None:
            return int(version) >= 3
    return False


def _decode_v5_energy(encoded_energy: np.ndarray, signed_energy: bool) -> np.ndarray:
    if signed_energy:
        return np.sign(encoded_energy) * np.expm1(np.abs(encoded_energy))
    return np.expm1(encoded_energy)


def _jet_label_func(flags: np.ndarray) -> np.ndarray:
    """Convert jet classification targets to class labels (0=empty, 1=light, 2=b)."""
    return jet_class_labels(flags)


# ── Permutation matching utilities (for DETR-style unordered decoder outputs) ──

def _find_best_perm_np(cost_matrix, n_slots):
    """Brute-force optimal assignment for small n_slots (n! <= 24).

    cost_matrix: (B, N, N) — cost[b, i, j] = cost of pred i matching target j.
    Returns perm: (B, N) where perm[b, i] = target slot that pred i matches.
    """
    B = cost_matrix.shape[0]
    all_perms = list(_iperms(range(n_slots)))
    best = np.zeros((B, n_slots), dtype=np.int64)
    for b in range(B):
        best_c = float("inf")
        for p in all_perms:
            c = sum(cost_matrix[b, i, p[i]] for i in range(n_slots))
            if c < best_c:
                best_c = c
                best[b] = p
    return best


def _match_decoder_slots(pred_reg, tgt_reg, pred_cls, tgt_cls, mask, n_slots):
    """Compute optimal permutation matching decoder pred slots to target slots.

    ``tgt_reg[b, i, j]`` expresses target slot ``j`` in prediction query
    ``i``'s normalization.  Uses regression MSE (on valid slots) +
    classification mismatch as cost.
    Returns perm: (B, N) where new_tgt[b,i] = old_tgt[b, perm[b,i]] matches pred[b,i].
    """
    B, N, D = pred_reg.shape
    cost = np.zeros((B, N, N))
    for j in range(N):
        diff = pred_reg - tgt_reg[:, :, j]
        mse = (diff ** 2).mean(axis=-1)
        cls_mismatch = (pred_cls != tgt_cls[:, j:j+1]).astype(np.float32)
        cost[:, :, j] = (mse + cls_mismatch) * mask[:, j:j+1].astype(np.float32)
    return _find_best_perm_np(cost, n_slots)


def _apply_perm_np(arr, perm):
    """Reorder arr[b, :, ...] by perm[b, :].  arr: (B, N, ...), perm: (B, N)."""
    idx = np.arange(perm.shape[0])[:, None]
    return arr[idx, perm]


def _apply_perm_1d_np(arr, perm):
    """Reorder 1-D arr[b, :] by perm[b, :]."""
    idx = np.arange(perm.shape[0])[:, None]
    return arr[idx, perm]


def _matching_reg_stats(key, n_slots, reg_means, reg_stds, *, device, dtype):
    """Return the mandatory pT/eta/E statistics used for set matching."""
    if reg_means is None or reg_stds is None:
        raise ValueError(
            "reg_means and reg_stds are required for query-normalized evaluation matching"
        )
    if key not in reg_means or key not in reg_stds:
        raise ValueError(f"missing matching normalization statistics for {key}")

    mean = torch.as_tensor(reg_means[key], dtype=dtype, device=device)
    std = torch.as_tensor(reg_stds[key], dtype=dtype, device=device)
    if mean.ndim != 2 or mean.shape != std.shape or mean.shape != (n_slots, 7):
        raise ValueError(f"matching normalization for {key} must have shape ({n_slots}, 7)")
    required = (0, 1, 2, 6)
    if not torch.isfinite(mean[:, required]).all() or not torch.isfinite(std[:, required]).all():
        raise ValueError(f"matching normalization for {key} must be finite")
    if (std[:, required] <= 0).any():
        raise ValueError(f"matching normalization for {key} must have positive std")
    return mean[:, [0, 1, 6]], std[:, [0, 1, 6]], mean[:, 2], std[:, 2]


def _tgt_to_pred_space(tgt_raw, pred_dim, key, reg_means, reg_stds):
    """Bring the raw z-scored cache regression target into the same representation
    as the model prediction, so the permutation-matching cost is computed in the
    space the model was trained against.

    For V5/VX predictions (``pred_dim == 5``) the prediction is
    ``(pT_n, eta_n, sin_phi, cos_phi, E_n)``; the raw cache target
    ``(pT_n, eta_n, phi_n, px_n, py_n, pz_n, E_n)`` (E_n is column 6, not 3 —
    see the regression contract) is converted via
    the shared train-time helper :func:`pat_embedder.losses.reg_target_to_v5`
    using the per-slot phi stats (``reg_*[key][:, 2]``).  Four-component predictions
    (``pred_dim == 4``) use ``(pT_n, eta_n, phi, E_n)``.  V1/V2/V3 predictions
    remain in the raw cache space.
    """
    _, _, phi_mean, phi_std = _matching_reg_stats(
        key, tgt_raw.size(1), reg_means, reg_stds, device=tgt_raw.device, dtype=tgt_raw.dtype
    )
    if pred_dim == 5:
        return reg_target_to_v5(tgt_raw, phi_mean, phi_std).cpu().numpy()
    if pred_dim == 4:
        phi = tgt_raw[..., 2] * phi_std + phi_mean
        return torch.cat([tgt_raw[..., :2], phi.unsqueeze(-1), tgt_raw[..., 6:7]], dim=-1).cpu().numpy()
    return tgt_raw[:, :, :pred_dim].cpu().numpy()


def _get_decoder_perms(model, inputs, targets, masks_raw, device,
                       max_tops=4, max_ws=4, outputs=None,
                       reg_means=None, reg_stds=None):
    """Run model (if outputs is None) and compute optimal top/W permutations.

    ``reg_means``/``reg_stds`` are mandatory cache statistics. They convert raw
    targets into the decoder representation and each receiving query's
    normalization (see :func:`_tgt_to_pred_space`).

    Returns: outputs tuple, top_perm (B, Nt), w_perm (B, Nw)
    """
    if outputs is None:
        with torch.no_grad():
            outputs = _call_model(model, inputs.to(device), {"masks": masks_raw})

    top_pred_reg = outputs[3].cpu().numpy()
    w_pred_reg   = outputs[4].cpu().numpy()
    top_pred_cls = outputs[8].cpu().numpy().argmax(axis=-1)
    w_pred_cls   = outputs[9].cpu().numpy().argmax(axis=-1)

    bs = inputs.size(0)
    top_tgt_raw = targets["regression"]["tops"].reshape(bs, max_tops, -1)
    w_tgt_raw   = targets["regression"]["ws"].reshape(bs, max_ws, -1)
    top_mean, top_std, _, _ = _matching_reg_stats(
        "tops", max_tops, reg_means, reg_stds,
        device=top_tgt_raw.device, dtype=top_tgt_raw.dtype,
    )
    w_mean, w_std, _, _ = _matching_reg_stats(
        "ws", max_ws, reg_means, reg_stds,
        device=w_tgt_raw.device, dtype=w_tgt_raw.dtype,
    )
    top_tgt_reg = _tgt_to_pred_space(top_tgt_raw, top_pred_reg.shape[-1], "tops", reg_means, reg_stds)
    w_tgt_reg   = _tgt_to_pred_space(w_tgt_raw, w_pred_reg.shape[-1], "ws", reg_means, reg_stds)
    top_tgt_reg = _target_in_query_normalization(
        torch.as_tensor(top_tgt_reg, dtype=top_tgt_raw.dtype), top_mean.cpu(), top_std.cpu()
    ).numpy()
    w_tgt_reg = _target_in_query_normalization(
        torch.as_tensor(w_tgt_reg, dtype=w_tgt_raw.dtype), w_mean.cpu(), w_std.cpu()
    ).numpy()

    top_mask = (masks_raw["tops"].cpu().numpy()
                if "tops" in masks_raw
                else np.ones((bs, max_tops), dtype=bool))
    w_mask   = (masks_raw["ws"].cpu().numpy()
                if "ws" in masks_raw
                else np.ones((bs, max_ws), dtype=bool))

    _top_map = {0: 0, 6: 1, -6: 2}
    _w_map   = {0: 0, 24: 1, -24: 2}
    top_raw_cls = targets["classification"]["tops"].cpu().numpy().flatten()
    top_tgt_cls = np.array([_top_map.get(int(p), 0) for p in top_raw_cls]).reshape(bs, max_tops)
    w_raw_cls   = targets["classification"]["ws"].cpu().numpy().flatten()
    w_tgt_cls   = np.array([_w_map.get(int(p), 0) for p in w_raw_cls]).reshape(bs, max_ws)

    top_perm = _match_decoder_slots(
        top_pred_reg, top_tgt_reg, top_pred_cls, top_tgt_cls, top_mask, max_tops)
    w_perm = _match_decoder_slots(
        w_pred_reg, w_tgt_reg, w_pred_cls, w_tgt_cls, w_mask, max_ws)

    return outputs, top_perm, w_perm


# Default 7-element label set (unchanged) vs. rd-dependent selection:
# picking labels by positional slice of a single shared list silently mislabels
# columns whose data order differs from this list's order (e.g. rd==7 raw cache
# order is pt,eta,phi,px,py,pz,energy — NOT this list's order). See
_DEFAULT_DIMENSION_NAMES = ['pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz']
_DIMENSION_NAMES_BY_RD = {
    4: ['pt', 'eta', 'phi', 'energy'],
    5: ['pt', 'eta', 'phi', 'energy'],
    7: ['pt', 'eta', 'phi', 'px', 'py', 'pz', 'energy'],
}


def plot_regression_distribution_all_particles(
    model,
    loader,
    device,
    reg_means,
    reg_stds,
    class_name,
    max_samples=None,
    denormalize=True,
    dimension_names=None,
    variables_to_plot=None,
    hue_by_classification=None,
    classification_label_map=None,
    classification_class_names=None,
    use_perm_matching=False,
):
    cls_output_idx = {"leptons": 5, "neutrinos": 6, "jets": 7, "tops": 8, "ws": 9}
    class_mapping = {
        'leptons': {'output_idx': 0, 'max_particles': 'max_leptons', 'target_key': 'leptons'},
        'neutrinos': {'output_idx': 1, 'max_particles': 'max_leptons', 'target_key': 'neutrinos'},
        'jets': {'output_idx': 2, 'max_particles': 'max_jets', 'target_key': 'jets'},
        'tops': {'output_idx': 3, 'max_particles': 'max_tops', 'target_key': 'tops'},
        'ws': {'output_idx': 4, 'max_particles': 'max_tops', 'target_key': 'ws'},
    }

    if class_name not in class_mapping:
        raise ValueError(f"Unknown class: {class_name}. Available: {list(class_mapping.keys())}")

    mapping = class_mapping[class_name]
    output_idx = mapping['output_idx']
    max_particles = getattr(model, mapping['max_particles'])
    target_key = mapping['target_key']

    model.eval()
    signed_energy = _uses_signed_v5_energy(model)

    if dimension_names is None:
        dimension_names = list(_DEFAULT_DIMENSION_NAMES)

    # Separate arrays for histogram (data-order) and scatter (decoder-order),
    # matching the notebook's two-view design.
    all_pred_hist = []   # data-order: pred permuted by perm_inv
    all_tgt_hist  = []   # data-order: target NOT permuted
    all_mask_hist = []   # data-order mask NOT permuted

    all_pred_scat = []   # decoder-order: pred NOT permuted
    all_tgt_scat  = []   # decoder-order: target permuted by perm
    all_mask_scat = []   # decoder-order mask permuted by perm

    all_cls_hist  = []   # hue labels in data-order
    all_cls_scat  = []   # hue labels in decoder-order

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(loader):
            if max_samples is not None and sample_count >= max_samples:
                break

            inputs = batch["features"].to(device)
            targets = batch["targets"]
            masks_raw = batch.get("masks", {})
            bs = inputs.size(0)

            if use_perm_matching and class_name in ("tops", "ws"):
                max_tops = getattr(model, "max_tops", 2)
                max_ws_m = getattr(model, "max_ws", max_tops)
                outputs, top_perm, w_perm = _get_decoder_perms(
                    model, inputs, targets, masks_raw, device,
                    max_tops=max_tops, max_ws=max_ws_m,
                    reg_means=reg_means, reg_stds=reg_stds,
                )
                perm = top_perm if class_name == "tops" else w_perm
            else:
                outputs = _call_model(model, inputs, batch)
                perm = None

            pred = outputs[output_idx].cpu().numpy()
            target = targets["regression"][target_key].cpu().numpy()

            if target_key in masks_raw:
                mask = masks_raw[target_key].cpu().numpy()
            else:
                mask = (
                    np.any(target != 0, axis=-1) if target.ndim == 3
                    else np.any(target.reshape(bs, max_particles, 7) != 0, axis=-1)
                )

            if len(target.shape) == 2:
                target = target.reshape(bs, max_particles, -1)

            # Support different regression dims (e.g. V5 uses (pt,eta,sin_phi,cos_phi,E))
            rd = pred.shape[-1]

            # rd-dependent label selection — only overrides the untouched
            # caller default, never an explicitly-passed dimension_names.
            if dimension_names == _DEFAULT_DIMENSION_NAMES and rd in _DIMENSION_NAMES_BY_RD:
                dimension_names = _DIMENSION_NAMES_BY_RD[rd]

            # Keep the full (un-truncated) target/stats available for the energy
            # column (raw col 6) — the :rd truncation below drops it for rd==5
            # Preserve energy column 6 before denormalization.
            target_full = target
            class_mean_full = np.array(reg_means[target_key])
            class_std_full = np.array(reg_stds[target_key])

            # Truncate target to rd dims BEFORE denorm (notebook style).
            # For rd=5: target becomes (pt_n, eta_n, phi_n, px_n, py_n).
            target = target[:, :, :rd]

            # IMPORTANT: denormalize in original slot order BEFORE any reordering.
            # reg_means/reg_stds are indexed per-slot; permuting first then denorming
            # would apply the wrong slot's stats.
            target_energy = None
            if denormalize:
                class_mean = class_mean_full[:, :rd]
                class_std  = class_std_full[:, :rd]
                if rd == 5:
                    # V5: pred is (pt_n, eta_n, sin_phi_actual, cos_phi_actual, E_n).
                    # sin/cos phi are already actual values in [-1,1] — do NOT denorm them.
                    # pt (dim 0) and E (dim 4) are log1p-scaled — expm1() after
                    # un-z-score recovers physical GeV.
                    pred_denorm = pred.copy()
                    pred_denorm[:, :, 0] = np.expm1(pred[:, :, 0] * class_std[:, 0] + class_mean[:, 0])
                    pred_denorm[:, :, 1] = pred[:, :, 1] * class_std[:, 1] + class_mean[:, 1]
                    # raw cache layout: (pT_n,eta_n,phi_n,px_n,py_n,pz_n,E_n) — energy
                    # stats live at column 6, not 3 (=px).
                    pred_denorm[:, :, 4] = _decode_v5_energy(
                        pred[:, :, 4] * class_std_full[:, 6] + class_mean_full[:, 6],
                        signed_energy,
                    )
                    # dims 2,3 (sin_phi, cos_phi) left as-is (already actual values)
                    pred = pred_denorm
                    # target: (pt_n, eta_n, phi_n, px_n, py_n) — denorm with per-column
                    # stats, then expm1 the pt column; energy is read from the full
                    # (un-truncated) target/stats below, not this rd-truncated slice.
                    target = target * class_std + class_mean
                    target[:, :, 0] = np.expm1(target[:, :, 0])
                    target_energy = np.expm1(
                        target_full[:, :, 6] * class_std_full[:, 6] + class_mean_full[:, 6]
                    )
                elif rd == 7:
                    # V1 (rd==7): raw cache layout (pT_n,eta_n,phi_n,px_n,py_n,pz_n,E_n).
                    # pT (col0) and E (col6) are log1p-scaled — expm1() after
                    # un-z-score recovers physical GeV.
                    pred   = pred   * class_std + class_mean
                    target = target * class_std + class_mean
                    pred[:, :, 0]   = np.expm1(pred[:, :, 0])
                    pred[:, :, 6]   = np.expm1(pred[:, :, 6])
                    target[:, :, 0] = np.expm1(target[:, :, 0])
                    target[:, :, 6] = np.expm1(target[:, :, 6])
                else:
                    # Other output dimensions use the coordinate-wise normalization.
                    pred   = pred   * class_std + class_mean
                    target = target * class_std + class_mean
            elif rd == 5:
                target_energy = target_full[:, :, 6]

            if rd == 5:
                # pred: (pt, eta, sin_phi_actual, cos_phi_actual, E)
                # Reconstruct phi from actual sin/cos (no denorm applied to them above).
                phi_from_pred = np.arctan2(pred[:, :, 2], pred[:, :, 3])
                pred = np.stack([pred[:, :, 0], pred[:, :, 1], phi_from_pred, pred[:, :, 4]], axis=-1)
                # target: (pt, eta, phi_actual, E_actual) — energy from the raw
                # col-6 read above (target_full), not the rd-truncated slice.
                target = np.stack(
                    [target[:, :, 0], target[:, :, 1], target[:, :, 2], target_energy], axis=-1
                )

            # Save data-order copies (before permutation)
            tgt_data  = target.copy()
            mask_data = mask.copy()

            # Apply permutation AFTER denormalization.
            # Decoder-order: target permuted by perm (forward), pred NOT permuted.
            # Data-order:    pred permuted by perm_inv (inverse), target NOT permuted.
            perm_inv = None
            if perm is not None:
                perm_inv = np.argsort(perm, axis=1)

            tgt_dec   = _apply_perm_np(target, perm)    if perm is not None else target.copy()
            mask_dec  = _apply_perm_1d_np(mask, perm)   if perm is not None else mask.copy()
            pred_data = _apply_perm_np(pred, perm_inv)  if perm_inv is not None else pred.copy()

            all_pred_scat.append(pred)        # decoder order (not permuted)
            all_tgt_scat.append(tgt_dec)      # decoder order (target permuted by perm)
            all_mask_scat.append(mask_dec)    # decoder order mask

            all_pred_hist.append(pred_data)   # data order (pred permuted by perm_inv)
            all_tgt_hist.append(tgt_data)     # data order (target not permuted)
            all_mask_hist.append(mask_data)   # data order mask

            if hue_by_classification:
                cls_idx = cls_output_idx[class_name]
                logits = outputs[cls_idx].cpu().numpy()
                pred_cls = np.argmax(logits, axis=-1)
                if hue_by_classification == "target":
                    raw_cls = targets["classification"][target_key].cpu().numpy()
                    if raw_cls.ndim == 1:
                        raw_cls = raw_cls.reshape(bs, -1)[:, :max_particles]
                    elif raw_cls.shape[1] != max_particles:
                        raw_cls = raw_cls.reshape(bs, max_particles)
                    pid_map = classification_label_map or _REGRESSION_DEFAULT_CLS_MAP.get(class_name)
                    if pid_map is None:
                        tgt_cls = _jet_label_func(raw_cls)
                    else:
                        tgt_cls = np.array([[pid_map.get(int(p), 0) for p in row] for row in raw_cls])
                    # hist: data-order target cls (not permuted)
                    all_cls_hist.append(tgt_cls)
                    # scat: decoder-order target cls (permuted by perm)
                    all_cls_scat.append(
                        _apply_perm_1d_np(tgt_cls, perm) if perm is not None else tgt_cls.copy()
                    )
                else:
                    # pred cls: data-order (permuted by perm_inv), decoder-order (not permuted)
                    pred_cls_data = (
                        _apply_perm_1d_np(pred_cls, perm_inv) if perm_inv is not None else pred_cls.copy()
                    )
                    all_cls_hist.append(pred_cls_data)
                    all_cls_scat.append(pred_cls)

            sample_count += bs
            if max_samples is not None and sample_count >= max_samples:
                excess = sample_count - max_samples
                if excess > 0:
                    all_pred_scat[-1]  = all_pred_scat[-1][:-excess]
                    all_tgt_scat[-1]   = all_tgt_scat[-1][:-excess]
                    all_mask_scat[-1]  = all_mask_scat[-1][:-excess]
                    all_pred_hist[-1]  = all_pred_hist[-1][:-excess]
                    all_tgt_hist[-1]   = all_tgt_hist[-1][:-excess]
                    all_mask_hist[-1]  = all_mask_hist[-1][:-excess]
                    if hue_by_classification:
                        all_cls_hist[-1] = all_cls_hist[-1][:-excess]
                        all_cls_scat[-1] = all_cls_scat[-1][:-excess]
                break

    pred_scat = np.concatenate(all_pred_scat, axis=0)
    tgt_scat  = np.concatenate(all_tgt_scat,  axis=0)
    mask_scat = np.concatenate(all_mask_scat, axis=0)

    pred_hist = np.concatenate(all_pred_hist, axis=0)
    tgt_hist  = np.concatenate(all_tgt_hist,  axis=0)
    mask_hist = np.concatenate(all_mask_hist, axis=0)

    print(tgt_hist.shape)

    cls_hist = np.concatenate(all_cls_hist, axis=0) if hue_by_classification else None
    cls_scat = np.concatenate(all_cls_scat, axis=0) if hue_by_classification else None
    n_cls = int(cls_hist.max()) + 1 if cls_hist is not None else 0
    cmap = plt.get_cmap("tab10")
    cls_names = (
        classification_class_names
        if classification_class_names is not None
        else [str(i) for i in range(max(n_cls, 1))]
    )

    # Trim dimension_names to actual number of dims in the output
    dimension_names = dimension_names[:pred_hist.shape[2]]

    n_dims = pred_hist.shape[2]
    if variables_to_plot is None:
        dimensions_to_plot = list(range(n_dims))
    else:
        dimensions_to_plot = []
        for var_name in variables_to_plot:
            if var_name in dimension_names:
                dim_idx = dimension_names.index(var_name)
                if dim_idx < n_dims:
                    dimensions_to_plot.append(dim_idx)

    n_vars = len(dimensions_to_plot)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    def _hist_block(ax, tv, pv, hv, dim_name):
        """Draw target + prediction histograms on ax, optionally split by hue."""
        # Float32 cannot represent 50 distinct bin edges between neighboring
        # values. Promote plotting data, retaining the original model outputs.
        tv = np.asarray(tv, dtype=np.float64)
        pv = np.asarray(pv, dtype=np.float64)
        if len(pv) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dim_name)
            return
        if hv is not None and len(hv) > 0:
            for c in range(n_cls):
                mc = hv == c
                if np.any(mc):
                    lbl = cls_names[c] if c < len(cls_names) else str(c)
                    ax.hist(tv[mc], bins=50, alpha=0.4, density=True,
                            label=f'Target ({lbl})', color=cmap(c % 10),
                            histtype='step', linewidth=1.5)
            for c in range(n_cls):
                mc = hv == c
                if np.any(mc):
                    lbl = cls_names[c] if c < len(cls_names) else str(c)
                    ax.hist(pv[mc], bins=50, alpha=0.4, density=True,
                            label=f'Pred ({lbl})', color=cmap(c % 10),
                            linestyle='--', histtype='step', linewidth=1.5)
        else:
            ax.hist(tv, bins=50, alpha=0.6, label='Target', color='blue', density=True)
            ax.hist(pv, bins=50, alpha=0.6, label='Predicted', color='red', density=True)
        ax.set_xlabel(dim_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{dim_name} (n={len(pv)})')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── 1. Aggregate: all valid particles combined (decoder order) ──
    all_vmask = mask_scat.astype(bool)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(f'{class_name}: all particles (n={all_vmask.sum()})', fontsize=16)
    hue_flat = cls_scat[all_vmask] if cls_scat is not None else None
    for pi, di in enumerate(dimensions_to_plot):
        _hist_block(axes[pi],
                    tgt_scat[:, :, di][all_vmask], pred_scat[:, :, di][all_vmask],
                    hue_flat, dimension_names[di])
    for i in range(n_vars, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # ── 2. Per-slot: original data-order targets (unbiased physics distributions) ──
    for particle_idx in range(max_particles):
        n_valid = mask_hist[:, particle_idx].astype(bool).sum()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f'{class_name}, data slot {particle_idx} '
            f'({n_valid}/{len(mask_hist)} occupied) — original target order',
            fontsize=14
        )
        hue_vals = cls_hist[:, particle_idx] if cls_hist is not None else None
        for pi, di in enumerate(dimensions_to_plot):
            _hist_block(axes[pi],
                        tgt_hist[:, particle_idx, di], pred_hist[:, particle_idx, di],
                        hue_vals, dimension_names[di])
        for i in range(n_vars, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    # ── 3. Per-slot: decoder-order matched pairs + scatter ──
    for particle_idx in range(max_particles):
        vmask_d = mask_scat[:, particle_idx].astype(bool)
        n_valid = vmask_d.sum()
        n_total = len(mask_scat)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f'{class_name}, slot {particle_idx} '
            f'({n_valid}/{n_total} occupied)', fontsize=16
        )

        hue_vals = cls_scat[:, particle_idx] if cls_scat is not None else None
        for plot_idx, dim_idx in enumerate(dimensions_to_plot):
            ax = axes[plot_idx]
            pred_vals = pred_scat[vmask_d, particle_idx, dim_idx]
            tgt_vals  = tgt_scat[vmask_d,  particle_idx, dim_idx]
            hue_dim   = hue_vals[vmask_d] if hue_vals is not None else None
            _hist_block(ax, tgt_vals, pred_vals, hue_dim, dimension_names[dim_idx])

        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        plt.show()

        # Scatter plot: decoder-ordered pred vs permutation-matched targets
        if len(dimensions_to_plot) > 0:
            first_dim = dimensions_to_plot[0]
            pred_vals = pred_scat[vmask_d, particle_idx, first_dim]
            tgt_vals  = tgt_scat[vmask_d,  particle_idx, first_dim]
            hue_sc    = hue_vals[vmask_d] if hue_vals is not None else None

            if len(pred_vals) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                if hue_sc is not None and len(hue_sc) > 0:
                    sc = ax.scatter(tgt_vals, pred_vals, c=hue_sc, alpha=0.5, s=10,
                                    cmap="tab10", vmin=0, vmax=9)
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label("Class (classification)")
                    if classification_class_names is not None:
                        cbar.set_ticks(range(len(classification_class_names)))
                        cbar.set_ticklabels(classification_class_names)
                else:
                    ax.scatter(tgt_vals, pred_vals, alpha=0.5, s=10)

                min_val = min(np.min(tgt_vals), np.min(pred_vals))
                max_val = max(np.max(tgt_vals), np.max(pred_vals))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')

                ax.set_xlabel(f'Target {dimension_names[first_dim]}')
                ax.set_ylabel(f'Predicted {dimension_names[first_dim]}')
                ax.set_title(
                    f'Scatter: {dimension_names[first_dim]} for {class_name}, particle {particle_idx}'
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                mse = np.mean((pred_vals - tgt_vals) ** 2)
                mae = np.mean(np.abs(pred_vals - tgt_vals))
                corr = np.corrcoef(tgt_vals, pred_vals)[0, 1]

                textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nCorr: {corr:.4f}'
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                plt.show()

    return pred_scat, tgt_scat, mask_scat


def plot_classification_distribution(
    model,
    loader,
    device,
    class_name,
    label_map=None,
    max_samples=None,
    class_names=None,
    show_confusion_matrix=True,
    show_per_class_accuracy=True,
    use_perm_matching=False,
    reg_means=None,
    reg_stds=None,
):
    class_mapping = {
        "leptons": {"output_idx": 5, "max_particles": "max_leptons", "target_key": "leptons", "n_classes": 7},
        "neutrinos": {"output_idx": 6, "max_particles": "max_leptons", "target_key": "neutrinos", "n_classes": 7},
        "jets": {"output_idx": 7, "max_particles": "max_jets", "target_key": "jets", "n_classes": 3},
        "tops": {"output_idx": 8, "max_particles": "max_tops", "target_key": "tops", "n_classes": 3},
        "ws": {"output_idx": 9, "max_particles": "max_tops", "target_key": "ws", "n_classes": 3},
    }
    default_label_maps = {
        "leptons": DEFAULT_LEPTON_LABEL_MAP,
        "neutrinos": DEFAULT_NEUTRINO_LABEL_MAP,
        "jets": None,
        "tops": DEFAULT_TOP_LABEL_MAP,
        "ws": DEFAULT_W_LABEL_MAP,
    }
    if class_name not in class_mapping:
        raise ValueError(f"Unknown class: {class_name}. Available: {list(class_mapping.keys())}")

    mapping = class_mapping[class_name]
    output_idx = mapping["output_idx"]
    max_particles = getattr(model, mapping["max_particles"])
    target_key = mapping["target_key"]
    n_classes = mapping["n_classes"]
    pid_to_idx = label_map if label_map is not None else default_label_maps[class_name]

    model.eval()
    all_pred = []
    all_tgt = []

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(loader):
            if max_samples is not None and sample_count >= max_samples:
                break
            inputs = batch["features"].to(device)
            targets = batch["targets"]
            bs = inputs.size(0)

            if use_perm_matching and class_name in ("tops", "ws"):
                masks_raw = batch.get("masks", {})
                max_tops = getattr(model, "max_tops", 2)
                max_ws_m = getattr(model, "max_ws", max_tops)
                outputs, top_perm, w_perm = _get_decoder_perms(
                    model, inputs, targets, masks_raw, device,
                    max_tops=max_tops, max_ws=max_ws_m,
                    reg_means=reg_means, reg_stds=reg_stds,
                )
                perm = top_perm if class_name == "tops" else w_perm
            else:
                outputs = _call_model(model, inputs, batch)
                perm = None

            logits = outputs[output_idx].cpu().numpy()
            pred_idx = np.argmax(logits, axis=-1)

            raw_tgt = targets["classification"][target_key].cpu().numpy()
            if raw_tgt.ndim == 1:
                raw_tgt = raw_tgt.reshape(bs, -1)[:, :max_particles]
            elif raw_tgt.shape[1] != max_particles:
                raw_tgt = raw_tgt.reshape(bs, max_particles)

            if pid_to_idx is None:
                tgt_idx = _jet_label_func(raw_tgt)
            else:
                tgt_idx = np.array([[pid_to_idx.get(int(p), 0) for p in row] for row in raw_tgt])

            # Permute targets by perm (forward) so tgt slot i aligns with pred slot i.
            # Do NOT permute pred — predictions are already in decoder-slot order.
            if perm is not None:
                tgt_idx = _apply_perm_1d_np(tgt_idx, perm)

            all_pred.append(pred_idx)
            all_tgt.append(tgt_idx)
            sample_count += bs
            if max_samples is not None and sample_count >= max_samples:
                excess = sample_count - max_samples
                if excess > 0:
                    all_pred[-1] = all_pred[-1][:-excess]
                    all_tgt[-1] = all_tgt[-1][:-excess]
                break

    predictions = np.concatenate(all_pred, axis=0)
    targets = np.concatenate(all_tgt, axis=0)

    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    valid = tgt_flat != 0
    if np.any(valid):
        acc_overall = np.mean(pred_flat == tgt_flat)
        acc_non_bkg = np.mean(pred_flat[valid] == tgt_flat[valid])
    else:
        acc_overall = np.mean(pred_flat == tgt_flat)
        acc_non_bkg = float("nan")

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    n_plots = int(show_confusion_matrix) + int(show_per_class_accuracy)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if show_confusion_matrix:
        ax = axes[plot_idx]
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for p, t in zip(pred_flat, tgt_flat):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[int(t), int(p)] += 1
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names[:n_classes])
        ax.set_yticklabels(class_names[:n_classes])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"Confusion matrix: {class_name}")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.colorbar(im, ax=ax, label="Count")
        plot_idx += 1

    if show_per_class_accuracy:
        ax = axes[plot_idx]
        per_class_correct = np.zeros(n_classes)
        per_class_total = np.zeros(n_classes)
        for p, t in zip(pred_flat, tgt_flat):
            if 0 <= t < n_classes:
                per_class_total[int(t)] += 1
                if p == t:
                    per_class_correct[int(t)] += 1
        acc_per_class = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)
        ax.bar(range(n_classes), acc_per_class, color="steelblue", edgecolor="black")
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names[:n_classes])
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Class")
        ax.set_title(f"Per-class accuracy: {class_name}")
        ax.set_ylim(0, 1.05)
        ax.axhline(acc_overall, color="red", linestyle="--", label=f"Overall: {acc_overall:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print(f"{class_name} -- overall accuracy: {acc_overall:.4f}, non-background accuracy: {acc_non_bkg:.4f}")
    return predictions, targets, {"accuracy_overall": acc_overall, "accuracy_non_background": acc_non_bkg}


def plot_assignment_quality(
    model,
    loader,
    device,
    max_samples=None,
    show_confusion=True,
    use_perm_matching=False,
    reg_means=None,
    reg_stds=None,
):
    assignment_keys = [
        "top_to_w", "top_to_bjet",
        "w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2",
    ]

    model.eval()
    all_preds = {k: [] for k in assignment_keys}
    all_tgts = {k: [] for k in assignment_keys}

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(loader):
            if max_samples is not None and sample_count >= max_samples:
                break
            inputs = batch["features"].to(device)
            targets = batch["targets"]
            bs = inputs.size(0)

            if use_perm_matching:
                masks_raw = batch.get("masks", {})
                max_tops = getattr(model, "max_tops", 2)
                max_ws_m = getattr(model, "max_ws", max_tops)
                outputs, top_perm, w_perm = _get_decoder_perms(
                    model, inputs, targets, masks_raw, device,
                    max_tops=max_tops, max_ws=max_ws_m,
                    reg_means=reg_means, reg_stds=reg_stds,
                )
            else:
                outputs = _call_model(model, inputs, batch)

            assign_out = outputs[10]

            if "assignments" not in targets:
                print("No assignment targets in data")
                return

            assign_tgt = {
                k: v.long().view(bs, -1).cpu().numpy()
                for k, v in targets["assignments"].items()
            }

            if use_perm_matching:
                # Permute target slot axes to align with prediction slots.
                # top-keyed targets: permute along top slot axis by top_perm.
                # w-keyed targets:   permute along W slot axis by w_perm.
                rows = np.arange(bs)[:, None]
                for k in ("top_to_w", "top_to_bjet"):
                    if k in assign_tgt:
                        assign_tgt[k] = assign_tgt[k][rows, top_perm]
                for k in ("w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2"):
                    if k in assign_tgt:
                        assign_tgt[k] = assign_tgt[k][rows, w_perm]
                # top_to_w values are indices into W target slots; remap through
                # the inverse of w_perm so they point to the new (permuted) W slots.
                if "top_to_w" in assign_tgt:
                    n_ws = w_perm.shape[1]
                    w_inv = np.empty_like(w_perm)
                    for b in range(bs):
                        for i in range(n_ws):
                            w_inv[b, w_perm[b, i]] = i
                    ttw = assign_tgt["top_to_w"]
                    valid = ttw < n_ws
                    ttw_safe = np.where(valid, ttw, 0)
                    assign_tgt["top_to_w"] = np.where(valid, w_inv[rows, ttw_safe], ttw)

            for k in assignment_keys:
                if k in assign_out and k in assign_tgt:
                    pred_idx = assign_out[k].cpu().numpy().argmax(axis=-1)
                    tgt_idx = assign_tgt[k] if use_perm_matching else assign_tgt[k]
                    all_preds[k].append(pred_idx)
                    all_tgts[k].append(tgt_idx)

            sample_count += bs
            if max_samples is not None and sample_count >= max_samples:
                excess = sample_count - max_samples
                if excess > 0:
                    for k in assignment_keys:
                        if all_preds[k]:
                            all_preds[k][-1] = all_preds[k][-1][:-excess]
                            all_tgts[k][-1] = all_tgts[k][-1][:-excess]
                break

    active_keys = [k for k in assignment_keys if all_preds[k]]

    accuracies = {}
    accuracies_assigned = {}
    for k in active_keys:
        preds = np.concatenate(all_preds[k], axis=0).flatten()
        tgts = np.concatenate(all_tgts[k], axis=0).flatten()
        accuracies[k] = np.mean(preds == tgts)
        n_choices = int(max(preds.max(), tgts.max())) + 1
        assigned = tgts < (n_choices - 1)
        if np.any(assigned):
            accuracies_assigned[k] = np.mean(preds[assigned] == tgts[assigned])
        else:
            accuracies_assigned[k] = float("nan")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(active_keys))
    width = 0.35
    bars1 = ax.bar(x - width / 2, [accuracies[k] for k in active_keys],
                   width, label="Overall", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width / 2, [accuracies_assigned[k] for k in active_keys],
                   width, label="Assigned only", color="coral", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(active_keys, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Assignment Head Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()

    if show_confusion:
        for k in active_keys:
            preds_k = np.concatenate(all_preds[k], axis=0).flatten()
            tgts_k = np.concatenate(all_tgts[k], axis=0).flatten()
            n_choices = int(max(preds_k.max(), tgts_k.max())) + 1

            cm = np.zeros((n_choices, n_choices), dtype=np.int64)
            for pi, ti in zip(preds_k, tgts_k):
                if 0 <= ti < n_choices and 0 <= pi < n_choices:
                    cm[int(ti), int(pi)] += 1

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

            fig, ax = plt.subplots(1, 1, figsize=(max(5, n_choices * 0.6), max(4, n_choices * 0.5)))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Assignment: {k} (row-normalised)")
            ax.set_xticks(range(n_choices))
            ax.set_yticks(range(n_choices))
            last_label = "none"
            labels = [str(i) for i in range(n_choices - 1)] + [last_label]
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            for i in range(n_choices):
                for j in range(n_choices):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=6)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()

    for k in active_keys:
        print(f"  {k}: overall={accuracies[k]:.4f}, assigned={accuracies_assigned[k]:.4f}")

    return accuracies, accuracies_assigned


def quick_eval_v5_style_metrics(model, loader, device, max_samples=100_000):
    """Headless classification (tops, leptons) + assignment accuracies.

    ``model`` must be callable as ``model(inputs)`` and return the same 14-tuple
    as :class:`PatEmbedderModelV3` (V5), e.g. the inner ``.model`` of
    :class:`PatEmbedderLightningV5`.

    Returns
    -------
    dict
        Nested metrics suitable for comparing a GATr-wrapped model vs V5 baseline.
    """
    model.eval()
    cls_specs = [
        ("tops", 8, "tops", DEFAULT_TOP_LABEL_MAP),
        ("leptons", 5, "leptons", DEFAULT_LEPTON_LABEL_MAP),
    ]
    cls_out = {}
    with torch.no_grad():
        sample_count = 0
        all_pred = {name: [] for name, *_ in cls_specs}
        all_tgt = {name: [] for name, *_ in cls_specs}
        assign_keys = [
            "top_to_w", "top_to_bjet",
            "w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2",
        ]
        all_ap = {k: [] for k in assign_keys}
        all_at = {k: [] for k in assign_keys}

        for batch in tqdm(loader, desc="quick_eval"):
            if max_samples is not None and sample_count >= max_samples:
                break
            inputs = batch["features"].to(device)
            targets = batch["targets"]
            bs = inputs.size(0)
            outputs = _call_model(model, inputs, batch)

            for name, out_idx, tgt_key, pid_map in cls_specs:
                logits = outputs[out_idx].cpu().numpy()
                pred_idx = np.argmax(logits, axis=-1)
                raw_tgt = targets["classification"][tgt_key].cpu().numpy()
                if raw_tgt.ndim == 1:
                    raw_tgt = raw_tgt.reshape(bs, -1)[:, : pred_idx.shape[1]]
                elif raw_tgt.shape[1] != pred_idx.shape[1]:
                    raw_tgt = raw_tgt.reshape(bs, pred_idx.shape[1])
                tgt_idx = np.array([[pid_map.get(int(p), 0) for p in row] for row in raw_tgt])
                all_pred[name].append(pred_idx)
                all_tgt[name].append(tgt_idx)

            if "assignments" in targets:
                assign_logits = outputs[10]
                assign_tgt = {k: v.long().view(bs, -1) for k, v in targets["assignments"].items()}
                for k in assign_keys:
                    if k in assign_logits and k in assign_tgt:
                        all_ap[k].append(assign_logits[k].cpu().numpy().argmax(axis=-1))
                        all_at[k].append(assign_tgt[k].cpu().numpy())

            sample_count += bs
            if max_samples is not None and sample_count >= max_samples:
                excess = sample_count - max_samples
                if excess > 0:
                    for name, *_ in cls_specs:
                        if all_pred[name]:
                            all_pred[name][-1] = all_pred[name][-1][:-excess]
                            all_tgt[name][-1] = all_tgt[name][-1][:-excess]
                    for k in assign_keys:
                        if all_ap[k]:
                            all_ap[k][-1] = all_ap[k][-1][:-excess]
                            all_at[k][-1] = all_at[k][-1][:-excess]
                break

        for name, *_ in cls_specs:
            p = np.concatenate(all_pred[name], axis=0).flatten()
            t = np.concatenate(all_tgt[name], axis=0).flatten()
            valid = t != 0
            acc_o = float(np.mean(p == t))
            acc_nb = float(np.mean(p[valid] == t[valid])) if np.any(valid) else float("nan")
            cls_out[name] = {"accuracy_overall": acc_o, "accuracy_non_background": acc_nb}

        assign_out = {}
        for k in assign_keys:
            if not all_ap[k]:
                continue
            preds = np.concatenate(all_ap[k], axis=0).flatten()
            tgts = np.concatenate(all_at[k], axis=0).flatten()
            acc_o = float(np.mean(preds == tgts))
            n_choices = int(max(preds.max(), tgts.max())) + 1
            assigned = tgts < (n_choices - 1)
            acc_a = float(np.mean(preds[assigned] == tgts[assigned])) if np.any(assigned) else float("nan")
            assign_out[k] = {"overall": acc_o, "assigned_only": acc_a}

    return {"classification": cls_out, "assignment": assign_out}


def plot_hlr_regression(
    model,
    loader,
    device,
    hlr_mean,
    hlr_std,
    branch_names=None,
    variables_to_plot=None,
    max_samples=None,
    denormalize=True,
    model_branches=None,
):
    """Evaluate and visualize high-level regression head quality.

    Parameters
    ----------
    branch_names : list[str] or None
        Full ordered list of branch names in the cached targets/stats.
    model_branches : list[str] or None
        Branch names the model was trained on (subset of branch_names).
        When provided and branch_names is also given, the correct columns
        are selected from targets and stats by name, even if branches
        were removed from the middle.  When None, columns are assumed to
        align positionally.
    """
    if model_branches is not None and branch_names is not None:
        col_indices = np.array([branch_names.index(b) for b in model_branches])
    else:
        col_indices = None

    model.eval()
    all_preds = []
    all_tgts = []

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(loader):
            if max_samples is not None and sample_count >= max_samples:
                break
            inputs = batch["features"].to(device)
            targets = batch["targets"]
            bs = inputs.size(0)

            outputs = _call_model(model, inputs, batch)
            hlr_pred = outputs[11]
            if hlr_pred is None:
                print("Model has no HLR head")
                return

            hlr_pred = hlr_pred.cpu().numpy()
            hlr_tgt = targets["high_level_regression"].cpu().numpy()

            n_pred = hlr_pred.shape[1]
            if col_indices is not None:
                hlr_tgt = hlr_tgt[:, col_indices]
            else:
                hlr_tgt = hlr_tgt[:, :n_pred]

            if denormalize and hlr_mean is not None and hlr_std is not None:
                if col_indices is not None:
                    hlr_mean_s = hlr_mean[col_indices]
                    hlr_std_s = hlr_std[col_indices]
                else:
                    hlr_mean_s = hlr_mean[:n_pred]
                    hlr_std_s = hlr_std[:n_pred]
                hlr_pred = hlr_pred * hlr_std_s + hlr_mean_s
                hlr_tgt = hlr_tgt * hlr_std_s + hlr_mean_s

            all_preds.append(hlr_pred)
            all_tgts.append(hlr_tgt)
            sample_count += bs
            if max_samples is not None and sample_count >= max_samples:
                excess = sample_count - max_samples
                if excess > 0:
                    all_preds[-1] = all_preds[-1][:-excess]
                    all_tgts[-1] = all_tgts[-1][:-excess]
                break

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_tgts, axis=0)
    num_vars = preds.shape[1]

    if model_branches is not None:
        branch_names = list(model_branches)
    elif branch_names is None:
        branch_names = [f"var_{i}" for i in range(num_vars)]
    else:
        branch_names = list(branch_names[:num_vars])

    if variables_to_plot is not None:
        if isinstance(variables_to_plot[0], str):
            indices = [branch_names.index(v) for v in variables_to_plot if v in branch_names]
        else:
            indices = list(variables_to_plot)
    else:
        indices = list(range(min(num_vars, 20)))

    mse_per_var = np.mean((preds - tgts) ** 2, axis=0)
    corr_per_var = np.array([
        np.corrcoef(preds[:, i], tgts[:, i])[0, 1]
        if np.std(tgts[:, i]) > 1e-9 else 0.0
        for i in range(num_vars)
    ])

    n_vars = len(indices)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle("HLR: Predicted vs Target distributions", fontsize=16)

    for plot_idx, var_idx in enumerate(indices):
        ax = axes[plot_idx]
        pred_vals = preds[:, var_idx].astype(np.float64)
        tgt_vals = tgts[:, var_idx].astype(np.float64)
        ax.hist(tgt_vals, bins=50, alpha=0.6, label="Target", color="blue", density=True)
        ax.hist(pred_vals, bins=50, alpha=0.6, label="Predicted", color="red", density=True)
        ax.set_title(f"{branch_names[var_idx]}\nr={corr_per_var[var_idx]:.3f}", fontsize=8)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_vars, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

    n_scatter = min(len(indices), 8)
    if n_scatter > 0:
        sc_cols = min(n_scatter, 4)
        sc_rows = (n_scatter + sc_cols - 1) // sc_cols
        fig, axes_sc = plt.subplots(sc_rows, sc_cols, figsize=(5 * sc_cols, 5 * sc_rows))
        axes_sc = np.atleast_1d(axes_sc).flatten()
        fig.suptitle("HLR: Predicted vs Target scatter", fontsize=16)

        for plot_idx, var_idx in enumerate(indices[:n_scatter]):
            ax = axes_sc[plot_idx]
            pred_vals = preds[:, var_idx]
            tgt_vals = tgts[:, var_idx]
            ax.scatter(tgt_vals, pred_vals, alpha=0.3, s=5)
            mn = min(tgt_vals.min(), pred_vals.min())
            mx = max(tgt_vals.max(), pred_vals.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="y=x")
            mse = mse_per_var[var_idx]
            corr = corr_per_var[var_idx]
            ax.set_xlabel("Target")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{branch_names[var_idx]}\nMSE={mse:.4f}, r={corr:.3f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_scatter, len(axes_sc)):
            axes_sc[idx].axis("off")
        plt.tight_layout()
        plt.show()

    sorted_idx = np.argsort(corr_per_var)[::-1]
    sorted_names = [branch_names[i] for i in sorted_idx]
    sorted_corrs = corr_per_var[sorted_idx]
    colors = [
        "forestgreen" if c > 0.7 else "orange" if c > 0.3 else "firebrick"
        for c in sorted_corrs
    ]

    fig, ax = plt.subplots(1, 1, figsize=(max(14, num_vars * 0.12), 5))
    ax.bar(range(len(sorted_names)), sorted_corrs, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("HLR: Per-variable correlation (predicted vs target)")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.7, color="green", linestyle="--", alpha=0.5, label="r=0.7")
    ax.axhline(0.3, color="orange", linestyle="--", alpha=0.5, label="r=0.3")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    n_good = np.sum(corr_per_var > 0.7)
    n_ok = np.sum((corr_per_var > 0.3) & (corr_per_var <= 0.7))
    n_bad = np.sum(corr_per_var <= 0.3)
    print(f"HLR summary: {n_good} vars with r>0.7, {n_ok} with 0.3<r<=0.7, {n_bad} with r<=0.3")
    print(f"  Mean correlation: {np.nanmean(corr_per_var):.4f}")
    print(f"  Mean MSE: {np.mean(mse_per_var):.4f}")

    return preds, tgts, {"mse_per_var": mse_per_var, "corr_per_var": corr_per_var}
