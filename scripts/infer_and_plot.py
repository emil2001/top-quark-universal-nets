#!/usr/bin/env python
"""Run PatEmbedder inference plots from a checkpoint or MLflow run.

Examples
--------
Local checkpoint:
    python scripts/infer_and_plot.py --version vx-full --checkpoint path/to/best.ckpt --cache-path data/level1_cached_full.pkl --outdir plots/vx-full

MLflow run name:
    python scripts/infer_and_plot.py --version vx-full --run-name demo-lgatr --experiment-name event-reconstruction --outdir plots/vx-full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pat_embedder import (  # noqa: E402
    DEFAULT_HIGH_LEVEL_BRANCHES,
    DEFAULT_LEPTON_LABEL_MAP,
    DEFAULT_NEUTRINO_LABEL_MAP,
    DEFAULT_TOP_LABEL_MAP,
    DEFAULT_W_LABEL_MAP,
    PatEmbedderLightning,
    PatEmbedderLightningV5,
    PatEmbedderLightningVXFull,
    load_cache,
    plot_assignment_quality,
    plot_classification_distribution,
    plot_hlr_regression,
    plot_regression_distribution_all_particles,
    quick_eval_v5_style_metrics,
)
from pat_embedder.data import detector_enc_padding_mask  # noqa: E402
from pat_embedder.evaluation import (  # noqa: E402
    _apply_perm_1d_np,
    _apply_perm_np,
    _decode_v5_energy,
    _get_decoder_perms,
    _uses_signed_v5_energy,
)
from pat_embedder.losses import (  # noqa: E402
    _apply_perm_1d,
    _apply_perm_2d,
    _apply_perm_with_query_normalization,
    _find_best_permutation,
    _invert_perm,
    _remap_assignment_indices,
    focal_ce_loss,
)
from pat_embedder.mlflow_utils import pick_mlflow_checkpoint_path, plot_loss_curves  # noqa: E402

LIGHTNING_CLASSES = {
    "v1": PatEmbedderLightning,
    "v5": PatEmbedderLightningV5,
    "vx-full": PatEmbedderLightningVXFull,
}

GATR_VERSIONS = {"vx-full"}
# DETR uses an explicit detector-derived padding mask.
DETR_VERSIONS = {"v5"}
DEFAULT_REGRESSION_PARTICLES = ["tops", "ws"]
DEFAULT_CLASSIFICATION_PARTICLES = ["tops", "leptons"]
PARTICLES = ["leptons", "neutrinos", "jets", "tops", "ws"]
CLASS_LABELS = {
    "leptons": ["empty", "e+", "e-", "mu+", "mu-", "tau+", "tau-"],
    "neutrinos": ["empty", "nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"],
    "jets": ["empty", "light", "b"],
    "tops": ["empty", "t", "tbar"],
    "ws": ["empty", "W+", "W-"],
}
CLASS_MAPS = {
    "leptons": DEFAULT_LEPTON_LABEL_MAP,
    "neutrinos": DEFAULT_NEUTRINO_LABEL_MAP,
    "jets": None,
    "tops": DEFAULT_TOP_LABEL_MAP,
    "ws": DEFAULT_W_LABEL_MAP,
}


class SaveFigures:
    """Redirect plt.show() calls from evaluation helpers into PNG files."""

    def __init__(self, outdir: Path, dpi: int):
        self.outdir = outdir
        self.dpi = dpi
        self._stem: Path | None = None
        self._counts: dict[str, int] = {}
        self._original_show = plt.show

    def install(self) -> None:
        # Force backend initialisation before replacing ``plt.show``.  Some
        # Matplotlib versions mutate ``pyplot.show.__signature__`` during lazy
        # backend setup, which fails if ``show`` is already a bound method.
        fig = plt.figure()
        plt.close(fig)
        plt.show = self.show

    def restore(self) -> None:
        plt.show = self._original_show

    def set_stem(self, stem: str) -> None:
        path = self.outdir / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stem = path
        self._counts[str(path)] = 0

    def show(self, *args, **kwargs) -> None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        if not figs:
            return

        stem = self._stem or (self.outdir / "figure")
        stem_key = str(stem)
        call_idx = self._counts.get(stem_key, 0)
        self._counts[stem_key] = call_idx + 1

        for fig_idx, fig in enumerate(figs):
            suffix = ""
            if call_idx > 0:
                suffix += f"_c{call_idx}"
            if fig_idx > 0:
                suffix += f"_f{fig_idx}"
            path = Path(f"{stem}{suffix}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"saved {path}")
        plt.close("all")


class GATrEvalWrapper:
    """Adapt VX Lightning modules to the model(features) evaluation API."""

    def __init__(self, pl_model, device: torch.device):
        self.accepts_masks = True
        self._pl = pl_model.to(device)
        self._model = pl_model.model.to(device).eval()
        self.device = device
        for attr in ("max_leptons", "max_jets", "max_tops", "max_ws"):
            setattr(self, attr, getattr(self._model, attr))

    def eval(self):
        self._model.eval()
        return self

    def __call__(self, x: torch.Tensor, masks: dict[str, torch.Tensor] | None = None):
        x = x.to(self.device).float()
        x_4mom, x_raw = self._pl._prepare_4mom({"features": x})
        # Always build the encoder mask from the detector inputs — the same
        # single source used at training time. The `masks` argument (parton-truth
        # slot validity) is ignored on purpose: it is not constructible at
        # inference and previously made eval differ from train.
        enc_padding_mask = detector_enc_padding_mask(x, self.max_leptons, self.max_jets)
        with torch.no_grad():
            return self._model(
                x,
                x_4mom=x_4mom,
                x_raw=x_raw,
                enc_padding_mask=enc_padding_mask,
            )


class DetrEvalWrapper:
    """Adapt V3/V5 (DETR) Lightning modules to the model(features) eval API.

    Builds an explicit detector-derived encoder padding mask from the input
    features — the same single source used at training time — and passes it to
    the model, so eval and train use an identical mask. Exposes
    ``accepts_masks=True`` so the evaluation helpers route through this explicit
    path instead of the in-model None-fallback. The ``masks`` argument
    (parton-truth) is ignored on purpose.
    """

    def __init__(self, pl_model, device: torch.device):
        self.accepts_masks = True
        self._model = pl_model.model.to(device).eval()
        self.device = device
        for attr in ("max_leptons", "max_jets", "max_tops", "max_ws"):
            setattr(self, attr, getattr(self._model, attr))

    def eval(self):
        self._model.eval()
        return self

    def __call__(self, x: torch.Tensor, masks: dict[str, torch.Tensor] | None = None):
        x = x.to(self.device).float()
        enc_padding_mask = detector_enc_padding_mask(x, self.max_leptons, self.max_jets)
        with torch.no_grad():
            return self._model(x, enc_padding_mask=enc_padding_mask)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", choices=sorted(LIGHTNING_CLASSES), required=True)

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="Local Lightning checkpoint path.")
    source.add_argument("--run-name", type=str, help="MLflow run name tag.")
    source.add_argument("--run-id", type=str, help="MLflow run id.")

    p.add_argument("--experiment-name", type=str, default="event-reconstruction")
    p.add_argument("--tracking-uri", type=str, default="mlruns")
    p.add_argument("--checkpoint-preference", choices=["best", "last", "last_step"], default="best")
    p.add_argument("--weights-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require an exact checkpoint state_dict match. Defaults off for older run compatibility.",
    )

    p.add_argument("--cache-path", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--cache-filename", type=str, default="level1_cached_full.pkl")
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--outdir", type=Path, default=Path("plots/inference"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--dpi", type=int, default=150)

    p.add_argument(
        "--regression-particles",
        type=str,
        default=",".join(DEFAULT_REGRESSION_PARTICLES),
        help="Comma-separated particles, 'all', or 'none'.",
    )
    p.add_argument(
        "--classification-particles",
        type=str,
        default=",".join(DEFAULT_CLASSIFICATION_PARTICLES),
        help="Comma-separated particles, 'all', or 'none'.",
    )
    p.add_argument("--regression-variables", type=str, default="pt,eta,phi,energy")
    p.add_argument("--regression-quality", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--regression-quality-particles", type=str, default="tops,ws")
    p.add_argument("--regression-quality-variables", type=str, default="pt,phi,energy")
    p.add_argument("--hue-regression-by", choices=["none", "target", "prediction"], default="none")
    p.add_argument("--assignment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hlr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--hlr-variables",
        type=str,
        default="all",
        help="Comma-separated HLR branch names, 'all', or 'none'.",
    )
    p.add_argument("--quick-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write foreground/background loss diagnostics for the evaluated split.",
    )
    p.add_argument("--loss-curves", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-perm-matching", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _resolve_cache(args: argparse.Namespace) -> tuple[str, str]:
    if args.cache_path:
        path = Path(args.cache_path)
        return str(path.parent), path.name
    return args.cache_dir, args.cache_filename


def _parse_csv(value: str, allowed: list[str] | None = None) -> list[str]:
    value = value.strip()
    if value.lower() == "none":
        return []
    if value.lower() == "all":
        if allowed is None:
            raise ValueError("'all' needs an allowed value list")
        return list(allowed)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if allowed is not None:
        unknown = sorted(set(items) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown values {unknown}; allowed values are {allowed}")
    return items


def _checkpoint_kwargs(version: str, data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": args.device,
        "weights_only": args.weights_only,
        "strict": args.strict_load,
    }
    if version in {"v5", "vx-full"}:
        reg_means = data["reg_means"]
        reg_stds = data["reg_stds"]
        kwargs["phi_means"] = {k: np.asarray(reg_means[k])[:, 2] for k in reg_means}
        kwargs["phi_stds"] = {k: np.asarray(reg_stds[k])[:, 2] for k in reg_stds}
        kwargs["reg_means"] = reg_means
        kwargs["reg_stds"] = reg_stds
    if version in GATR_VERSIONS:
        kwargs["feat_mean"] = data["feat_mean"]
        kwargs["feat_std"] = data["feat_std"]
    return kwargs


def _normalize_compiled_checkpoint(path: str) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    prefix = "model._orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return path

    fixed_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = "model." + key[len(prefix):]
        fixed_state[key] = value
    checkpoint["state_dict"] = fixed_state

    tmp = tempfile.NamedTemporaryFile(prefix="pat_embedder_ckpt_", suffix=".ckpt", delete=False)
    tmp.close()
    torch.save(checkpoint, tmp.name)
    print(f"Normalised torch.compile checkpoint keys: {path} -> {tmp.name}")
    return tmp.name


def _iter_checkpoint_artifact_paths(client, run_id: str) -> list[str]:
    found: list[str] = []

    def walk(prefix: str = "") -> None:
        for ent in client.list_artifacts(run_id, path=prefix or None):
            if ent.is_dir:
                walk(ent.path)
            elif ent.path.endswith(".ckpt"):
                found.append(ent.path)

    walk()
    return found


def _download_mlflow_checkpoint(args: argparse.Namespace) -> str:
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_uri)
    if args.run_id:
        run_id = args.run_id
        run = client.get_run(run_id)
        print(f"Found run: {run_id} ({run.info.status})")
    else:
        exp = client.get_experiment_by_name(args.experiment_name)
        if exp is None:
            raise ValueError(f"No experiment found with name '{args.experiment_name}'")
        runs = client.search_runs(
            exp.experiment_id,
            filter_string=f"tags.mlflow.runName = '{args.run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"No run found with name '{args.run_name}' in experiment '{args.experiment_name}'"
            )
        run = runs[0]
        run_id = run.info.run_id
        print(f"Found run: {run_id} ({run.info.status})")

    ckpt_path = pick_mlflow_checkpoint_path(
        _iter_checkpoint_artifact_paths(client, run_id),
        preference=args.checkpoint_preference,
    )
    local_path = client.download_artifacts(run_id, ckpt_path)
    print(f"Downloaded: {local_path}")
    return local_path


def _load_lightning_model(args: argparse.Namespace, data: dict[str, Any]):
    cls = LIGHTNING_CLASSES[args.version]
    kwargs = _checkpoint_kwargs(args.version, data, args)
    if args.checkpoint:
        path = args.checkpoint
    else:
        path = _download_mlflow_checkpoint(args)
    path = _normalize_compiled_checkpoint(path)
    return cls.load_from_checkpoint(path, **kwargs)


def _eval_model(pl_model, version: str, device: torch.device):
    if version in GATR_VERSIONS:
        return GATrEvalWrapper(pl_model, device)
    if version in DETR_VERSIONS:
        return DetrEvalWrapper(pl_model, device)
    # MLP has no attention mask.
    model = pl_model.model.to(device).eval()
    return model


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "features": batch["features"].to(device, dtype=torch.float32, non_blocking=True),
        "targets": batch["targets"],
        "masks": batch["masks"],
    }


def _slice_batch(batch: dict[str, Any], size: int) -> dict[str, Any]:
    def take(value):
        if isinstance(value, torch.Tensor):
            return value[:size]
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value

    return take(batch)


def _split_outputs(outputs) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict, torch.Tensor]:
    reg_out = {
        "leptons": outputs[0],
        "neutrinos": outputs[1],
        "jets": outputs[2],
        "tops": outputs[3],
        "ws": outputs[4],
    }
    cls_out = {
        "lepton_logits": outputs[5],
        "neutrino_logits": outputs[6],
        "jet_logits": outputs[7],
        "top_logits": outputs[8],
        "w_logits": outputs[9],
    }
    return reg_out, cls_out, outputs[10], outputs[11]


def _forward_lightning_model(pl_model, version: str, inputs: torch.Tensor, masks: dict[str, torch.Tensor]):
    if version == "vx-full":
        x_4mom, x_raw = pl_model._prepare_4mom({"features": inputs})
        enc_padding_mask = detector_enc_padding_mask(
            inputs, pl_model.model.max_leptons, pl_model.model.max_jets
        )
        return pl_model.model(
            inputs,
            x_4mom=x_4mom,
            x_raw=x_raw,
            enc_padding_mask=enc_padding_mask,
        )
    return pl_model.model(inputs)


def _mass_stats(pl_model) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    # These persistent stats also define set-matching normalization, even when
    # the optional physics mass loss is disabled.
    keys = ("leptons", "neutrinos", "jets", "tops", "ws")
    if not all(hasattr(pl_model, f"mass_reg_mean_{k}") for k in keys):
        return None, None
    means = {k: getattr(pl_model, f"mass_reg_mean_{k}") for k in keys}
    stds = {k: getattr(pl_model, f"mass_reg_std_{k}") for k in keys}
    return means, stds


def _criterion_losses(
    pl_model,
    version: str,
    reg_out: dict[str, torch.Tensor],
    cls_out: dict[str, torch.Tensor],
    assign_out: dict[str, torch.Tensor],
    hlr_out: torch.Tensor,
    reg_tgt: dict[str, torch.Tensor],
    cls_tgt: dict[str, torch.Tensor],
    assign_tgt: dict[str, torch.Tensor] | None,
    hlr_tgt: torch.Tensor | None,
    masks: dict[str, torch.Tensor],
    outputs,
) -> dict[str, float]:
    mass_reg_means, mass_reg_stds = _mass_stats(pl_model)
    if version in {"v5"} and hasattr(pl_model.criterion, "forward_with_aux"):
        loss_dict = pl_model.criterion.forward_with_aux(
            reg_out,
            cls_out,
            reg_tgt,
            cls_tgt,
            masks,
            assign_out,
            assign_tgt,
            hlr_out,
            hlr_tgt,
            aux_head_outputs=outputs[12] if len(outputs) > 12 else None,
            dn_outputs=None,
            dn_targets_reg=None,
            dn_targets_cls=None,
            dn_masks=None,
            reg_means=mass_reg_means,
            reg_stds=mass_reg_stds,
            matching_reg_means=mass_reg_means,
            matching_reg_stds=mass_reg_stds,
        )
    else:
        loss_dict = pl_model.criterion(
            reg_out,
            cls_out,
            reg_tgt,
            cls_tgt,
            masks,
            assign_out,
            assign_tgt,
            hlr_out,
            hlr_tgt,
            aux_head_outputs=None,
            dn_outputs=None,
            reg_means=mass_reg_means,
            reg_stds=mass_reg_stds,
            matching_reg_means=mass_reg_means,
            matching_reg_stds=mass_reg_stds,
        )
    keys = ["total", "regression", "classification", "assignment", "hlr", "aux", "dn", "mass"]
    return {
        key: float(value.detach().cpu())
        for key, value in loss_dict.items()
        if key in keys and isinstance(value, torch.Tensor)
    }


def _aligned_targets_for_loss(
    criterion,
    reg_out: dict[str, torch.Tensor],
    cls_out: dict[str, torch.Tensor],
    assign_out: dict[str, torch.Tensor],
    reg_tgt: dict[str, torch.Tensor],
    cls_tgt: dict[str, torch.Tensor],
    assign_tgt: dict[str, torch.Tensor] | None,
    masks: dict[str, torch.Tensor],
    matching_reg_means: dict[str, torch.Tensor] | None = None,
    matching_reg_stds: dict[str, torch.Tensor] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor],
]:
    reg_tgt = {k: v.clone() for k, v in reg_tgt.items()}
    cls_tgt = {k: v.clone() for k, v in cls_tgt.items()}
    masks = {k: v.clone() for k, v in masks.items()}
    assign_tgt = None if assign_tgt is None else {k: v.clone() for k, v in assign_tgt.items()}
    assign_core = {k: v for k, v in assign_out.items() if k not in {"top_presence", "w_presence"}}

    if assign_tgt is None or not hasattr(criterion, "_w_pairwise_cost"):
        return reg_tgt, cls_tgt, assign_tgt, masks

    batch_size = cls_out["w_logits"].size(0)
    w_cls_tgt = cls_tgt["w_classes"].view(batch_size, criterion.max_ws)
    w_cost = criterion._w_pairwise_cost(
        cls_out["w_logits"],
        reg_out["ws"],
        assign_core,
        w_cls_tgt,
        reg_tgt["ws"],
        assign_tgt,
        masks["ws"],
        *(
            (matching_reg_means["ws"], matching_reg_stds["ws"])
            if matching_reg_means is not None
            else ()
        ),
    )
    w_perm = _find_best_permutation(w_cost, criterion.max_ws)
    w_perm_inv = _invert_perm(w_perm)
    cls_tgt["w_classes"] = _apply_perm_1d(w_cls_tgt, w_perm).view(-1)
    reg_tgt["ws"] = (
        _apply_perm_with_query_normalization(
            reg_tgt["ws"], w_perm, matching_reg_means["ws"], matching_reg_stds["ws"]
        )
        if matching_reg_means is not None
        else _apply_perm_2d(reg_tgt["ws"], w_perm)
    )
    masks["ws"] = _apply_perm_1d(masks["ws"], w_perm)
    for key in ("w_to_lepton", "w_to_neutrino", "w_to_jet1", "w_to_jet2"):
        if key in assign_tgt:
            assign_tgt[key] = _apply_perm_1d(assign_tgt[key], w_perm)
    if "top_to_w" in assign_tgt:
        assign_tgt["top_to_w"] = _remap_assignment_indices(
            assign_tgt["top_to_w"], w_perm_inv, no_match_idx=criterion.max_ws
        )

    top_cls_tgt = cls_tgt["top_classes"].view(batch_size, criterion.max_tops)
    top_cost = criterion._top_pairwise_cost(
        cls_out["top_logits"],
        reg_out["tops"],
        assign_core,
        top_cls_tgt,
        reg_tgt["tops"],
        assign_tgt,
        masks["tops"],
        *(
            (matching_reg_means["tops"], matching_reg_stds["tops"])
            if matching_reg_means is not None
            else ()
        ),
    )
    top_perm = _find_best_permutation(top_cost, criterion.max_tops)
    cls_tgt["top_classes"] = _apply_perm_1d(top_cls_tgt, top_perm).view(-1)
    reg_tgt["tops"] = (
        _apply_perm_with_query_normalization(
            reg_tgt["tops"], top_perm, matching_reg_means["tops"], matching_reg_stds["tops"]
        )
        if matching_reg_means is not None
        else _apply_perm_2d(reg_tgt["tops"], top_perm)
    )
    masks["tops"] = _apply_perm_1d(masks["tops"], top_perm)
    for key in ("top_to_w", "top_to_bjet"):
        if key in assign_tgt:
            assign_tgt[key] = _apply_perm_1d(assign_tgt[key], top_perm)
    return reg_tgt, cls_tgt, assign_tgt, masks


class RunningMeans:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add(self, key: str, value: torch.Tensor | float, count: torch.Tensor | float | int) -> None:
        count_f = float(count.detach().cpu()) if isinstance(count, torch.Tensor) else float(count)
        if count_f <= 0.0:
            return
        value_f = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        self.sums[key] = self.sums.get(key, 0.0) + value_f * count_f
        self.counts[key] = self.counts.get(key, 0.0) + count_f

    def result(self) -> dict[str, float]:
        return {key: self.sums[key] / self.counts[key] for key in sorted(self.sums)}


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((x * y).sum().cpu() / denom.cpu())


def _add_classification_diagnostics(
    stats: RunningMeans,
    cls_out: dict[str, torch.Tensor],
    cls_tgt: dict[str, torch.Tensor],
    gamma: float,
) -> None:
    specs = {
        "leptons": ("lepton_logits", "lepton_classes"),
        "neutrinos": ("neutrino_logits", "neutrino_classes"),
        "jets": ("jet_logits", "jet_classes"),
        "tops": ("top_logits", "top_classes"),
        "ws": ("w_logits", "w_classes"),
    }
    for name, (logit_key, tgt_key) in specs.items():
        logits = cls_out[logit_key]
        targets = cls_tgt[tgt_key].view(logits.shape[:2])
        losses = focal_ce_loss(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            gamma=gamma,
        ).view_as(targets)
        pred = logits.argmax(dim=-1)
        background = targets == 0
        foreground = ~background
        prefix = f"classification/{name}"
        stats.add(f"{prefix}/loss_all", losses.mean(), targets.numel())
        stats.add(f"{prefix}/loss_background", losses[background].mean(), background.sum())
        stats.add(f"{prefix}/loss_foreground", losses[foreground].mean(), foreground.sum())
        stats.add(f"{prefix}/accuracy_all", (pred == targets).float().mean(), targets.numel())
        stats.add(
            f"{prefix}/accuracy_foreground",
            (pred[foreground] == targets[foreground]).float().mean(),
            foreground.sum(),
        )
        stats.add(f"{prefix}/foreground_fraction", foreground.float().mean(), targets.numel())
        stats.add(f"{prefix}/pred_background_fraction", (pred == 0).float().mean(), targets.numel())
        stats.add(
            f"{prefix}/pred_background_on_foreground",
            (pred[foreground] == 0).float().mean(),
            foreground.sum(),
        )


def _add_assignment_diagnostics(
    stats: RunningMeans,
    assign_out: dict[str, torch.Tensor],
    assign_tgt: dict[str, torch.Tensor] | None,
    gamma: float,
) -> None:
    if assign_tgt is None:
        return
    for key, logits in assign_out.items():
        if key not in assign_tgt or key in {"top_presence", "w_presence"}:
            continue
        targets = assign_tgt[key].view(logits.shape[:2])
        losses = focal_ce_loss(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            gamma=gamma,
        ).view_as(targets)
        pred = logits.argmax(dim=-1)
        no_match_idx = logits.size(-1) - 1
        assigned = targets < no_match_idx
        no_match = ~assigned
        prefix = f"assignment/{key}"
        stats.add(f"{prefix}/loss_all", losses.mean(), targets.numel())
        stats.add(f"{prefix}/loss_assigned", losses[assigned].mean(), assigned.sum())
        stats.add(f"{prefix}/loss_no_match", losses[no_match].mean(), no_match.sum())
        stats.add(f"{prefix}/accuracy_all", (pred == targets).float().mean(), targets.numel())
        stats.add(
            f"{prefix}/accuracy_assigned",
            (pred[assigned] == targets[assigned]).float().mean(),
            assigned.sum(),
        )
        stats.add(f"{prefix}/assigned_fraction", assigned.float().mean(), targets.numel())
        stats.add(
            f"{prefix}/pred_no_match_fraction",
            (pred == no_match_idx).float().mean(),
            targets.numel(),
        )
        stats.add(
            f"{prefix}/pred_no_match_on_assigned",
            (pred[assigned] == no_match_idx).float().mean(),
            assigned.sum(),
        )


def _add_regression_diagnostics(
    stats: RunningMeans,
    criterion,
    reg_out: dict[str, torch.Tensor],
    reg_tgt: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> None:
    for name in ("leptons", "neutrinos", "jets", "tops", "ws"):
        mask = masks[name].bool()
        if not mask.any():
            continue
        pred = reg_out[name]
        tgt = reg_tgt[name]
        loss = criterion._reg_cost_per_slot(pred, tgt) if hasattr(criterion, "_reg_cost_per_slot") else (
            pred - tgt
        ).pow(2).mean(dim=-1)
        prefix = f"regression/{name}"
        stats.add(f"{prefix}/loss_valid", loss[mask].mean(), mask.sum())
        stats.add(f"{prefix}/pred_std_valid", pred[mask].std(), mask.sum())
        stats.add(f"{prefix}/target_std_valid", tgt[mask].std(), mask.sum())


def _add_hlr_diagnostics(
    stats: RunningMeans,
    hlr_out: torch.Tensor,
    hlr_tgt: torch.Tensor | None,
    masks: dict[str, torch.Tensor],
) -> None:
    if hlr_tgt is None:
        return
    loss = (hlr_out - hlr_tgt).pow(2)
    mask = masks.get("hlr")
    if mask is not None:
        mask = mask.bool()
        stats.add("hlr/loss_valid", loss[mask].mean(), mask.sum())
        pred = hlr_out[mask]
        tgt = hlr_tgt[mask]
    else:
        stats.add("hlr/loss_valid", loss.mean(), loss.numel())
        pred = hlr_out
        tgt = hlr_tgt
    stats.add("hlr/pred_std", pred.std(), pred.numel())
    stats.add("hlr/target_std", tgt.std(), tgt.numel())
    stats.add("hlr/correlation_flat", _safe_corr(pred, tgt), 1)


def run_diagnostics(
    pl_model,
    version: str,
    loader,
    device: torch.device,
    max_samples: int | None,
) -> dict[str, Any]:
    pl_model = pl_model.to(device).eval()
    pl_model.model.eval()
    stats = RunningMeans()
    criterion_stats = RunningMeans()
    gamma = float(getattr(pl_model.criterion, "gamma", 2.0))
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and seen >= max_samples:
                break
            remaining = None if max_samples is None else max_samples - seen
            if remaining is not None and batch["features"].size(0) > remaining:
                batch = _slice_batch(batch, remaining)
            moved = _move_batch_to_device(batch, device)
            inputs = moved["features"]
            targets = moved["targets"]
            masks_raw = moved["masks"]
            batch_size = inputs.size(0)

            reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks = pl_model._prepare_targets(
                targets, masks_raw, batch_size
            )
            outputs = _forward_lightning_model(pl_model, version, inputs, masks)
            reg_out, cls_out, assign_out, hlr_out = _split_outputs(outputs)

            losses = _criterion_losses(
                pl_model,
                version,
                reg_out,
                cls_out,
                assign_out,
                hlr_out,
                reg_tgt,
                cls_tgt,
                assign_tgt,
                hlr_tgt,
                masks,
                outputs,
            )
            for key, value in losses.items():
                criterion_stats.add(key, value, batch_size)

            reg_tgt_a, cls_tgt_a, assign_tgt_a, masks_a = _aligned_targets_for_loss(
                pl_model.criterion,
                reg_out,
                cls_out,
                assign_out,
                reg_tgt,
                cls_tgt,
                assign_tgt,
                masks,
                *_mass_stats(pl_model),
            )
            _add_classification_diagnostics(stats, cls_out, cls_tgt_a, gamma)
            _add_assignment_diagnostics(stats, assign_out, assign_tgt_a, gamma)
            _add_regression_diagnostics(stats, pl_model.criterion, reg_out, reg_tgt_a, masks_a)
            _add_hlr_diagnostics(stats, hlr_out, hlr_tgt, masks)

            seen += batch_size

    return {
        "num_samples": seen,
        "criterion": criterion_stats.result(),
        "foreground_background": stats.result(),
    }


def _safe_np_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _angle_delta(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred - tgt), np.cos(pred - tgt))


def _denorm_v5_regression(
    pred: np.ndarray,
    target: np.ndarray,
    reg_means: dict[str, Any],
    reg_stds: dict[str, Any],
    target_key: str,
    signed_energy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rd = pred.shape[-1]
    # Keep the full (un-truncated) target/stats available for the energy column
    # (raw col 6) — the :rd truncation below drops it for rd==5.
    target_full = target
    class_mean_full = np.asarray(reg_means[target_key])
    class_std_full = np.asarray(reg_stds[target_key])
    target = target[:, :, :rd]
    class_mean = class_mean_full[:, :rd]
    class_std = class_std_full[:, :rd]
    if rd == 5:
        # pt (dim 0) and E (dim 4) are log1p-scaled — expm1() after un-z-score
        # recovers physical GeV.
        pred_denorm = pred.copy()
        pred_denorm[:, :, 0] = np.expm1(pred[:, :, 0] * class_std[:, 0] + class_mean[:, 0])
        pred_denorm[:, :, 1] = pred[:, :, 1] * class_std[:, 1] + class_mean[:, 1]
        # raw cache layout: (pT_n,eta_n,phi_n,px_n,py_n,pz_n,E_n) — energy stats
        # live at column 6, not 3 (=px).
        pred_denorm[:, :, 4] = _decode_v5_energy(
            pred[:, :, 4] * class_std_full[:, 6] + class_mean_full[:, 6],
            signed_energy,
        )
        pred_phi = np.arctan2(pred_denorm[:, :, 2], pred_denorm[:, :, 3])
        pred_out = np.stack(
            [pred_denorm[:, :, 0], pred_denorm[:, :, 1], pred_phi, pred_denorm[:, :, 4]],
            axis=-1,
        )
        target_denorm = target * class_std + class_mean
        target_denorm[:, :, 0] = np.expm1(target_denorm[:, :, 0])
        target_energy = np.expm1(
            target_full[:, :, 6] * class_std_full[:, 6] + class_mean_full[:, 6]
        )
        target_out = np.stack(
            [target_denorm[:, :, 0], target_denorm[:, :, 1], target_denorm[:, :, 2], target_energy],
            axis=-1,
        )
        return pred_out, target_out

    if rd == 7:
        # V1 (rd==7): raw cache layout (pT_n,eta_n,phi_n,px_n,py_n,pz_n,E_n). pT
        # (col0) and E (col6) are log1p-scaled — expm1() after un-z-score recovers
        # physical GeV.
        pred_out = pred * class_std + class_mean
        target_out = target * class_std + class_mean
        pred_out[:, :, 0] = np.expm1(pred_out[:, :, 0])
        pred_out[:, :, 6] = np.expm1(pred_out[:, :, 6])
        target_out[:, :, 0] = np.expm1(target_out[:, :, 0])
        target_out[:, :, 6] = np.expm1(target_out[:, :, 6])
        # (pt, eta, phi, energy) — select columns explicitly, not a positional :4
        # slice, or energy (col 6) would be dropped and col 3 (px) mislabelled as
        # it.
        return pred_out[:, :, [0, 1, 2, 6]], target_out[:, :, [0, 1, 2, 6]]

    # Other output dimensions use the coordinate-wise normalization.
    pred_out = pred * class_std + class_mean
    target_out = target * class_std + class_mean
    return pred_out[:, :, :4], target_out[:, :, :4]


def compute_regression_quality_metrics(
    model,
    loader,
    device: torch.device,
    reg_means: dict[str, Any],
    reg_stds: dict[str, Any],
    particles: list[str],
    variables: list[str],
    max_samples: int | None,
) -> dict[str, Any]:
    specs = {
        "tops": {"output_idx": 3, "max_attr": "max_tops", "target_key": "tops"},
        "ws": {"output_idx": 4, "max_attr": "max_ws", "target_key": "ws"},
    }
    dim_idx = {"pt": 0, "eta": 1, "phi": 2, "energy": 3}
    unknown_particles = sorted(set(particles) - set(specs))
    if unknown_particles:
        raise ValueError(f"Regression quality only supports tops/ws, got {unknown_particles}")
    unknown_variables = sorted(set(variables) - set(dim_idx))
    if unknown_variables:
        raise ValueError(f"Unknown regression quality variables: {unknown_variables}")

    model.eval()
    accum = {p: {"pred": [], "target": [], "mask": []} for p in particles}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and seen >= max_samples:
                break
            inputs = batch["features"].to(device)
            targets = batch["targets"]
            masks_raw = batch.get("masks", {})
            batch_size = inputs.size(0)
            max_tops = getattr(model, "max_tops", 4)
            max_ws = getattr(model, "max_ws", max_tops)
            outputs, top_perm, w_perm = _get_decoder_perms(
                model, inputs, targets, masks_raw, device, max_tops=max_tops, max_ws=max_ws,
                reg_means=reg_means, reg_stds=reg_stds,
            )
            perms = {"tops": top_perm, "ws": w_perm}

            for particle in particles:
                spec = specs[particle]
                target_key = spec["target_key"]
                max_particles = getattr(model, spec["max_attr"])
                pred = outputs[spec["output_idx"]].detach().cpu().numpy()
                target = targets["regression"][target_key].cpu().numpy()
                if target.ndim == 2:
                    target = target.reshape(batch_size, max_particles, -1)
                pred, target = _denorm_v5_regression(
                    pred,
                    target,
                    reg_means,
                    reg_stds,
                    target_key,
                    signed_energy=_uses_signed_v5_energy(model),
                )
                perm = perms[particle]
                target = _apply_perm_np(target, perm)
                mask = masks_raw[target_key].cpu().numpy()
                mask = _apply_perm_1d_np(mask, perm).astype(bool)
                accum[particle]["pred"].append(pred)
                accum[particle]["target"].append(target)
                accum[particle]["mask"].append(mask)

            seen += batch_size
            if max_samples is not None and seen >= max_samples:
                excess = seen - max_samples
                if excess > 0:
                    for particle in particles:
                        for key in accum[particle]:
                            accum[particle][key][-1] = accum[particle][key][-1][:-excess]
                break

    metrics: dict[str, Any] = {}
    for particle in particles:
        pred = np.concatenate(accum[particle]["pred"], axis=0)
        target = np.concatenate(accum[particle]["target"], axis=0)
        mask = np.concatenate(accum[particle]["mask"], axis=0)
        metrics[particle] = {}
        for variable in variables:
            idx = dim_idx[variable]
            pvals = pred[:, :, idx][mask]
            tvals = target[:, :, idx][mask]
            if variable == "phi":
                diff = _angle_delta(pvals, tvals)
                metrics[particle][variable] = {
                    "corr_sin_phi": _safe_np_corr(np.sin(pvals), np.sin(tvals)),
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff ** 2))),
                    "n": int(pvals.size),
                }
            else:
                diff = pvals - tvals
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                target_std = float(np.std(tvals))
                metrics[particle][variable] = {
                    "corr": _safe_np_corr(pvals, tvals),
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": rmse,
                    "nrmse": float(rmse / target_std) if target_std > 0.0 else float("nan"),
                    "target_std": target_std,
                    "n": int(pvals.size),
                }
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device cuda, but torch.cuda.is_available() is False. "
            "Run this script on a node/session with a visible NVIDIA driver or use --device cpu."
        )
    cache_dir, cache_filename = _resolve_cache(args)
    data = load_cache(cache_dir=cache_dir, filename=cache_filename)
    loader = data[f"{args.split}_loader"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    pl_model = _load_lightning_model(args, data)
    model = _eval_model(pl_model, args.version, device)
    print(f"Loaded {type(pl_model).__name__}; evaluating {type(getattr(pl_model, 'model', model)).__name__}")

    saver = SaveFigures(args.outdir, args.dpi)
    saver.install()
    metrics: dict[str, Any] = {
        "version": args.version,
        "split": args.split,
        "max_samples": args.max_samples,
    }

    try:
        if args.quick_metrics:
            print("Running quick metrics")
            metrics["quick"] = quick_eval_v5_style_metrics(
                model, loader, device, max_samples=args.max_samples
            )

        if args.diagnostics:
            print("Running foreground/background diagnostics")
            diagnostics = run_diagnostics(pl_model, args.version, loader, device, args.max_samples)
            metrics["diagnostics"] = diagnostics
            diagnostics_path = args.outdir / "diagnostics.json"
            diagnostics_path.write_text(json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True))
            print(f"wrote {diagnostics_path}")

        if args.regression_quality:
            print("Running regression quality metrics")
            metrics["regression_quality"] = compute_regression_quality_metrics(
                model=model,
                loader=loader,
                device=device,
                reg_means=data["reg_means"],
                reg_stds=data["reg_stds"],
                particles=_parse_csv(args.regression_quality_particles, ["tops", "ws"]),
                variables=_parse_csv(args.regression_quality_variables, ["pt", "eta", "phi", "energy"]),
                max_samples=args.max_samples,
            )

        regression_particles = _parse_csv(args.regression_particles, PARTICLES)
        regression_variables = _parse_csv(args.regression_variables)
        hue_by = None if args.hue_regression_by == "none" else args.hue_regression_by
        for particle in regression_particles:
            print(f"Plotting regression: {particle}")
            saver.set_stem(f"regression/{particle}")
            plot_regression_distribution_all_particles(
                model=model,
                loader=loader,
                device=device,
                reg_means=data["reg_means"],
                reg_stds=data["reg_stds"],
                class_name=particle,
                variables_to_plot=regression_variables or None,
                hue_by_classification=hue_by,
                classification_label_map=CLASS_MAPS[particle],
                classification_class_names=CLASS_LABELS[particle],
                max_samples=args.max_samples,
                use_perm_matching=args.use_perm_matching,
            )

        classification_particles = _parse_csv(args.classification_particles, PARTICLES)
        metrics["classification"] = {}
        for particle in classification_particles:
            print(f"Plotting classification: {particle}")
            saver.set_stem(f"classification/{particle}")
            _, _, cls_metrics = plot_classification_distribution(
                model=model,
                loader=loader,
                device=device,
                class_name=particle,
                label_map=CLASS_MAPS[particle],
                class_names=CLASS_LABELS[particle],
                max_samples=args.max_samples,
                use_perm_matching=args.use_perm_matching,
                reg_means=data["reg_means"],
                reg_stds=data["reg_stds"],
            )
            metrics["classification"][particle] = cls_metrics

        if args.assignment:
            print("Plotting assignment quality")
            saver.set_stem("assignment/quality")
            result = plot_assignment_quality(
                model=model,
                loader=loader,
                device=device,
                max_samples=args.max_samples,
                show_confusion=True,
                use_perm_matching=args.use_perm_matching,
                reg_means=data["reg_means"],
                reg_stds=data["reg_stds"],
            )
            if result is not None:
                accuracies, accuracies_assigned = result
                metrics["assignment"] = {
                    "overall": accuracies,
                    "assigned_only": accuracies_assigned,
                }

        hlr_variables = _parse_csv(args.hlr_variables, DEFAULT_HIGH_LEVEL_BRANCHES)
        if args.hlr and hlr_variables and data.get("hlr_mean") is not None:
            print("Plotting HLR regression")
            saver.set_stem("hlr/regression")
            result = plot_hlr_regression(
                model=model,
                loader=loader,
                device=device,
                hlr_mean=data["hlr_mean"],
                hlr_std=data["hlr_std"],
                branch_names=DEFAULT_HIGH_LEVEL_BRANCHES,
                model_branches=DEFAULT_HIGH_LEVEL_BRANCHES,
                variables_to_plot=hlr_variables,
                max_samples=args.max_samples,
            )
            if result is not None:
                _, _, hlr_metrics = result
                metrics["hlr"] = hlr_metrics

        if args.loss_curves:
            run_id = args.run_id
            if run_id is None and hasattr(pl_model, "logger") and pl_model.logger is not None:
                run_id = getattr(pl_model.logger, "run_id", None)
            if run_id is None:
                print("Skipping loss curves: pass --run-id for MLflow metric lookup.")
            else:
                print("Plotting loss curves")
                saver.set_stem("training/loss_curves")
                plot_loss_curves(run_id=run_id, tracking_uri=args.tracking_uri)
    finally:
        saver.restore()

    metrics_path = args.outdir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
