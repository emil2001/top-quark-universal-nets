import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

CACHE_CONTRACT_VERSION = 2


class CacheContractError(RuntimeError):
    """A cache cannot safely be used by the current training contract."""


class LegacyCacheSidecarError(CacheContractError):
    """A legacy cache has unreconstructed targets and an obsolete stats sidecar."""


class CachedBatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


# Hard sentinel for compute_stats' outlier guard. Legitimate physical
# quantities in this pipeline (log1p-transformed pT/E, eta, sin/cos phi)
# never exceed the low hundreds (see reg_stds ranges in the cache); a value
# beyond this is a corrupted generator-level entry (e.g. the ws-slot-3 eta
# targets that came out raw~1e11, with an outlier guard),
# not real physics. Treated exactly like NaN: excluded from mean/var/count.
_OUTLIER_ABS_THRESHOLD = 1e6


def compute_stats(loader):
    batch0 = next(iter(loader))
    feats0 = batch0["features"]
    if isinstance(feats0, torch.Tensor):
        feats0 = feats0.cpu().numpy()
    B, F = feats0.shape

    regs0 = batch0["targets"]["regression"]
    reg_keys = list(regs0.keys())

    Pk: dict[str, int] = {}
    Dk: dict[str, int] = {}
    for k in reg_keys:
        arr0 = regs0[k]
        if isinstance(arr0, torch.Tensor):
            arr0 = arr0.cpu().numpy()
        Pk[k], Dk[k] = arr0.shape[1], arr0.shape[2]

    has_hlr = "high_level_regression" in batch0["targets"]
    if has_hlr:
        hlr0 = batch0["targets"]["high_level_regression"]
        if isinstance(hlr0, torch.Tensor):
            hlr0 = hlr0.cpu().numpy()
        H = hlr0.shape[1]

    feat_sum = np.zeros(F, dtype=np.float64)
    feat_sq = np.zeros(F, dtype=np.float64)
    feat_count = np.zeros(F, dtype=np.int64)

    reg_sums = {k: np.zeros((Pk[k], Dk[k]), dtype=np.float64) for k in reg_keys}
    reg_sqs = {k: np.zeros((Pk[k], Dk[k]), dtype=np.float64) for k in reg_keys}
    reg_count = {k: np.zeros((Pk[k], Dk[k]), dtype=np.int64) for k in reg_keys}

    if has_hlr:
        hlr_sum = np.zeros(H, dtype=np.float64)
        hlr_sq = np.zeros(H, dtype=np.float64)
        hlr_count = np.zeros(H, dtype=np.int64)

    for batch in tqdm(loader):
        f = batch["features"]
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy()

        valid = ~np.isnan(f) & (np.abs(f) <= _OUTLIER_ABS_THRESHOLD)
        f_masked = np.where(valid, f, np.nan)
        feat_sum += np.nansum(f_masked, axis=0)
        feat_sq += np.nansum(f_masked * f_masked, axis=0)
        feat_count += valid.sum(axis=0)

        regs = batch["targets"]["regression"]
        for k in reg_keys:
            a = regs[k]
            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()
            a = a.reshape(a.shape[0], Pk[k], Dk[k])
            valid_k = ~np.isnan(a) & (np.abs(a) <= _OUTLIER_ABS_THRESHOLD)
            a_masked = np.where(valid_k, a, np.nan)
            reg_sums[k] += np.nansum(a_masked, axis=0)
            reg_sqs[k] += np.nansum(a_masked * a_masked, axis=0)
            reg_count[k] += valid_k.sum(axis=0)

        if has_hlr:
            h = batch["targets"]["high_level_regression"]
            if isinstance(h, torch.Tensor):
                h = h.cpu().numpy()
            valid_h = ~np.isnan(h) & (np.abs(h) <= _OUTLIER_ABS_THRESHOLD)
            h_masked = np.where(valid_h, h, np.nan)
            hlr_sum += np.nansum(h_masked, axis=0)
            hlr_sq += np.nansum(h_masked * h_masked, axis=0)
            hlr_count += valid_h.sum(axis=0)

    feat_mean = feat_sum / feat_count
    feat_var = feat_sq / feat_count - feat_mean**2
    feat_std = np.sqrt(np.maximum(feat_var, 1e-6))

    reg_means, reg_stds = {}, {}
    for k in reg_keys:
        mu = reg_sums[k] / reg_count[k]
        var = reg_sqs[k] / reg_count[k] - mu**2
        std = np.sqrt(np.maximum(var, 1e-6))
        reg_means[k] = mu.astype(np.float32)
        reg_stds[k] = std.astype(np.float32)

    hlr_mean, hlr_std = None, None
    if has_hlr:
        hlr_mu = hlr_sum / hlr_count
        hlr_var = hlr_sq / hlr_count - hlr_mu**2
        hlr_mean = hlr_mu.astype(np.float32)
        hlr_std = np.sqrt(np.maximum(hlr_var, 1e-6)).astype(np.float32)

    return (
        feat_mean.astype(np.float32),
        feat_std.astype(np.float32),
        reg_means,
        reg_stds,
        hlr_mean,
        hlr_std,
    )


def collate_fn(batch: list[dict]):
    feats = np.stack([b["features"] for b in batch])
    regs0 = batch[0]["targets"]["regression"]
    reg_out = {k: np.stack([b["targets"]["regression"][k] for b in batch]) for k in regs0}
    clss0 = batch[0]["targets"]["classification"]
    cls_out = {k: np.stack([b["targets"]["classification"][k] for b in batch]) for k in clss0}
    result = {"features": feats, "targets": {"regression": reg_out, "classification": cls_out}}
    if "assignments" in batch[0]["targets"]:
        asn0 = batch[0]["targets"]["assignments"]
        result["targets"]["assignments"] = {
            k: np.stack([b["targets"]["assignments"][k] for b in batch]) for k in asn0
        }
    if "high_level_regression" in batch[0]["targets"]:
        result["targets"]["high_level_regression"] = np.stack(
            [b["targets"]["high_level_regression"] for b in batch]
        )
    return result


def make_collate_and_scale(feat_mean, feat_std, reg_means, reg_stds,
                           hlr_mean=None, hlr_std=None):
    mu_f = torch.from_numpy(feat_mean).unsqueeze(0)
    sd_f = torch.from_numpy(feat_std).unsqueeze(0)
    mu_reg = {k: torch.from_numpy(v).unsqueeze(0) for k, v in reg_means.items()}
    sd_reg = {k: torch.from_numpy(v).unsqueeze(0) for k, v in reg_stds.items()}
    if hlr_mean is not None:
        mu_h = torch.from_numpy(hlr_mean).unsqueeze(0)
        sd_h = torch.from_numpy(hlr_std).unsqueeze(0)

    def collate(batch: list[dict]):
        feats = np.stack([b["features"] for b in batch])
        tfeat = torch.from_numpy(feats).float()
        feat_valid = ~torch.isnan(tfeat)
        tfeat = (tfeat - mu_f) / sd_f
        tfeat[~feat_valid] = 0.0

        regs0 = batch[0]["targets"]["regression"]
        reg_out = {}
        masks = {}
        for k in regs0:
            arr = np.stack([b["targets"]["regression"][k] for b in batch])
            tarr = torch.from_numpy(arr).float()
            slot_valid = ~torch.isnan(tarr).all(dim=-1)
            masks[k] = slot_valid
            tarr_normed = (tarr - mu_reg[k]) / sd_reg[k]
            tarr_normed[torch.isnan(tarr_normed)] = 0.0
            reg_out[k] = tarr_normed

        clss0 = batch[0]["targets"]["classification"]
        cls_out = {
            k: torch.from_numpy(
                np.stack([b["targets"]["classification"][k] for b in batch])
            ).to(torch.int8)
            for k in clss0
        }

        result = {
            "features": tfeat,
            "masks": masks,
            "targets": {
                "regression": reg_out,
                "classification": cls_out,
            },
        }

        if "assignments" in batch[0]["targets"]:
            asn_keys = batch[0]["targets"]["assignments"].keys()
            result["targets"]["assignments"] = {
                k: torch.from_numpy(
                    np.stack([b["targets"]["assignments"][k] for b in batch])
                ).to(torch.int8)
                for k in asn_keys
            }

        if "high_level_regression" in batch[0]["targets"] and hlr_mean is not None:
            hlr_arr = np.stack([b["targets"]["high_level_regression"] for b in batch])
            hlr_t = torch.from_numpy(hlr_arr).float()
            hlr_valid = ~torch.isnan(hlr_t)
            hlr_normed = (hlr_t - mu_h) / sd_h
            hlr_normed[~hlr_valid] = 0.0
            result["targets"]["high_level_regression"] = hlr_normed
            result["masks"]["hlr"] = hlr_valid

        return result

    return collate


def jet_class_labels(flags):
    """Convert signed jet b-tag flags to class indices: empty=0, light=1, b=2.

    Cache contract v2 uses -1 for light jets, +1 for b jets and 0 for padding.
    Jet flags are not PDG IDs. Accepts a torch tensor or a NumPy array.
    """
    present = flags != 0
    is_b = flags > 0
    if isinstance(flags, torch.Tensor):
        return present.long() + is_b.long()
    return present.astype(np.int64) + is_b.astype(np.int64)


def detector_slot_presence(features, max_leptons, max_jets):
    """Per-slot detector presence inferred from model-input features only.

    A detector slot (lepton/jet) is *present* iff at least one of its seven
    kinematic features is non-zero. ``make_collate_and_scale`` forces padded
    kinematic features (NaN) to exactly ``0.0`` after scaling. The categorical
    feature is excluded: a padded category's raw ``0.0`` becomes non-zero after
    z-scoring. This depends on detector inputs alone — never on parton-level
    regression targets — so the same presence is reproducible at inference time
    on unlabelled data.

    Args:
        features: (B, input_dim) flat features, layout
            ``[MET(2) | leptons(max_leptons*8) | jets(max_jets*8)]``.

    Returns:
        (lep_present (B, max_leptons) bool, jet_present (B, max_jets) bool).
    """
    B = features.size(0)
    lep = features[:, 2:2 + max_leptons * 8].reshape(B, max_leptons, 8)
    jet = features[:, 2 + max_leptons * 8:].reshape(B, max_jets, 8)
    return (lep[..., :7] != 0.0).any(dim=-1), (jet[..., :7] != 0.0).any(dim=-1)


def detector_enc_padding_mask(features, max_leptons, max_jets, n_query=0):
    """Encoder padding mask (``True`` = ignored by attention) from detector inputs.

    Single inference-able source of the encoder padding mask, applied
    identically in training and evaluation across all architectures. Token
    layout is ``[MET(1) | leptons | jets | optional query slots]``. The MET
    token and any decoder query tokens (top/W) are always present (never
    masked); only detector tokens can be padding.

    Args:
        features: (B, input_dim) flat features.
        n_query: number of trailing always-present query tokens to append
            (e.g. ``max_tops + max_ws`` for the V2 single-stack mask). ``0`` for
            encoder-only masks (V3/DETR, GATr).
    """
    B = features.size(0)
    device = features.device
    lep_present, jet_present = detector_slot_presence(features, max_leptons, max_jets)
    met = torch.zeros(B, 1, dtype=torch.bool, device=device)
    cols = [met, ~lep_present, ~jet_present]
    if n_query > 0:
        cols.append(torch.zeros(B, n_query, dtype=torch.bool, device=device))
    return torch.cat(cols, dim=1)


def _cache_batch(batch):
    cached = {
        "features": batch["features"].cpu().clone(),
        "masks": {k: v.cpu().clone() for k, v in batch["masks"].items()},
        "targets": {
            "regression": {
                k: v.cpu().clone() for k, v in batch["targets"]["regression"].items()
            },
            "classification": {
                k: v.cpu().clone() for k, v in batch["targets"]["classification"].items()
            },
        },
    }
    if "assignments" in batch["targets"]:
        cached["targets"]["assignments"] = {
            k: v.cpu().clone() for k, v in batch["targets"]["assignments"].items()
        }
    if "high_level_regression" in batch["targets"]:
        cached["targets"]["high_level_regression"] = (
            batch["targets"]["high_level_regression"].cpu().clone()
        )
    return cached


def save_cache(train_batches, test_batches, feat_mean, feat_std,
               reg_means, reg_stds, hlr_mean, hlr_std,
               cache_dir="data", filename="level1_cached_full.pkl", build_metadata=None):
    os.makedirs(cache_dir, exist_ok=True)
    cache = {
        "train_batches": train_batches,
        "test_batches": test_batches,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "reg_means": reg_means,
        "reg_stds": reg_stds,
        "hlr_mean": hlr_mean,
        "hlr_std": hlr_std,
        "contract_version": CACHE_CONTRACT_VERSION,
    }
    if build_metadata is not None:
        cache["build_metadata"] = build_metadata
    cache_path = os.path.join(cache_dir, filename)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cache to {cache_path}  ({os.path.getsize(cache_path) / 1e6:.1f} MB)")
    return cache_path


def _reg_stats_fix_path(cache_dir, filename):
    return os.path.join(cache_dir, filename + ".reg_stats_fix.json")


def _cache_provenance_path(cache_path):
    return f"{cache_path}.provenance.json"


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache_provenance(cache_path, code_version, previous_cache_path=None, differences=(),
                           build_metadata=None):
    """Write immutable cache provenance next to a completed contract-v2 cache."""
    provenance = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "cache_path": os.path.abspath(cache_path),
        "cache_sha256": _sha256(cache_path),
        "cache_size_bytes": os.path.getsize(cache_path),
        "code_version": code_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "differences_from_previous": list(differences),
    }
    if previous_cache_path is not None:
        provenance["previous_cache"] = {
            "path": os.path.abspath(previous_cache_path),
            "sha256": _sha256(previous_cache_path),
            "size_bytes": os.path.getsize(previous_cache_path),
        }
    if build_metadata is not None:
        provenance["build_metadata"] = build_metadata

    path = _cache_provenance_path(cache_path)
    with open(path, "x") as output:
        json.dump(provenance, output, indent=2, sort_keys=True)
        output.write("\n")
    return path


def _denormalize_regression_targets(batch, reg_means, reg_stds):
    """Restore physical cache targets before recomputing their normalization."""
    raw_targets = batch["targets"]["regression"]
    for key, target in raw_targets.items():
        mean = torch.as_tensor(reg_means[key], dtype=target.dtype, device=target.device)
        std = torch.as_tensor(reg_stds[key], dtype=target.dtype, device=target.device)
        raw = target * std.unsqueeze(0) + mean.unsqueeze(0)
        raw[~batch["masks"][key].unsqueeze(-1).expand_as(raw)] = torch.nan
        raw_targets[key] = raw


def _valid_cache_events(batch):
    batch_size = batch["features"].shape[0]
    valid = torch.ones(batch_size, dtype=torch.bool)
    for target in batch["targets"]["regression"].values():
        present = ~torch.isnan(target)
        valid &= (~present | torch.isfinite(target)).all(dim=(1, 2))
        valid &= (~present | (target.abs() <= _OUTLIER_ABS_THRESHOLD)).all(dim=(1, 2))
    return valid


def _select_cache_events(value, keep, batch_size):
    if isinstance(value, dict):
        return {key: _select_cache_events(item, keep, batch_size) for key, item in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
        return value[keep].clone()
    return value


def _recompute_regression_stats(batches):
    """Reuse the canonical outlier-safe statistics implementation for raw labels."""
    _, _, reg_means, reg_stds, _, _ = compute_stats(batches)
    return reg_means, reg_stds


def _normalize_regression_targets(batches, reg_means, reg_stds):
    for batch in batches:
        for key, target in batch["targets"]["regression"].items():
            mean = torch.as_tensor(reg_means[key], dtype=target.dtype, device=target.device)
            std = torch.as_tensor(reg_stds[key], dtype=target.dtype, device=target.device)
            normalized = (target - mean.unsqueeze(0)) / std.unsqueeze(0)
            normalized[torch.isnan(normalized)] = 0.0
            batch["targets"]["regression"][key] = normalized


def rebuild_cache(source_cache_path, output_path):
    """Rebuild cache labels and regression statistics together under contract v2."""
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing cache: {output_path}")

    with open(source_cache_path, "rb") as source:
        cache = pickle.load(source)
    source_reg_means = cache["reg_means"]
    source_reg_stds = cache["reg_stds"]
    cleaned_batches = []
    dropped_corrupted_samples = 0
    for split_name in ("train_batches", "test_batches"):
        rebuilt_split = []
        for batch in cache[split_name]:
            _denormalize_regression_targets(batch, source_reg_means, source_reg_stds)
            batch_size = batch["features"].shape[0]
            keep = _valid_cache_events(batch)
            dropped_corrupted_samples += batch_size - int(keep.sum())
            if keep.any():
                rebuilt_split.append(
                    batch if keep.all() else _select_cache_events(batch, keep, batch_size)
                )
        cache[split_name] = rebuilt_split
        cleaned_batches.extend(rebuilt_split)

    reg_means, reg_stds = _recompute_regression_stats(cleaned_batches)
    _normalize_regression_targets(cleaned_batches, reg_means, reg_stds)
    build_metadata = {
        "dropped_corrupted_samples": dropped_corrupted_samples,
        "outlier_abs_threshold": _OUTLIER_ABS_THRESHOLD,
    }
    return save_cache(
        cache["train_batches"], cache["test_batches"], cache["feat_mean"], cache["feat_std"],
        reg_means, reg_stds, cache.get("hlr_mean"), cache.get("hlr_std"),
        cache_dir=os.path.dirname(output_path), filename=os.path.basename(output_path),
        build_metadata=build_metadata,
    ), build_metadata


def load_cache(cache_dir="data", filename="level1_cached_full.pkl"):
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(_reg_stats_fix_path(cache_dir, filename)):
        raise LegacyCacheSidecarError(
            f"Cache {cache_path} has legacy reg_stats_fix sidecar; rebuild targets and statistics."
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    if cache.get("contract_version") != CACHE_CONTRACT_VERSION:
        raise CacheContractError(
            f"Cache {cache_path} is not contract v{CACHE_CONTRACT_VERSION}; rebuild targets and statistics."
        )

    train_batches = cache["train_batches"]
    test_batches = cache["test_batches"]
    feat_mean = cache["feat_mean"]
    feat_std = cache["feat_std"]
    reg_means = cache["reg_means"]
    reg_stds = cache["reg_stds"]
    hlr_mean = cache.get("hlr_mean")
    hlr_std = cache.get("hlr_std")

    train_loader = DataLoader(
        CachedBatchDataset(train_batches),
        batch_size=None,
        num_workers=0,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CachedBatchDataset(test_batches),
        batch_size=None,
        num_workers=0,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loaded from {cache_path}: {len(train_batches)} train, {len(test_batches)} test batches")
    print(f"Feature dim: {train_batches[0]['features'].shape[-1]}")

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_batches": train_batches,
        "test_batches": test_batches,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "reg_means": reg_means,
        "reg_stds": reg_stds,
        "hlr_mean": hlr_mean,
        "hlr_std": hlr_std,
        "build_metadata": cache.get("build_metadata"),
    }
