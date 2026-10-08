import math
import shlex
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch

from .data import detector_enc_padding_mask, jet_class_labels
from .losses import PatEmbedderLoss, PatEmbedderLossV5, reg_target_to_v5
from .models import REG_DIM_V5, PatEmbedderModel, PatEmbedderModelV3


def _matching_stats_from_cache(reg_means, reg_stds, name, n_slots):
    """Extract mandatory pT/eta/E statistics for query-normalized matching."""
    if reg_means is None or reg_stds is None:
        raise ValueError("reg_means and reg_stds are required for query-normalized matching")
    if name not in reg_means or name not in reg_stds:
        raise ValueError(f"missing matching normalization statistics for {name}")
    mean = np.asarray(reg_means[name])
    std = np.asarray(reg_stds[name])
    if mean.shape != (n_slots, 7) or std.shape != (n_slots, 7):
        raise ValueError(f"matching normalization for {name} must have shape ({n_slots}, 7)")
    required = [0, 1, 2, 6]
    if not np.isfinite(mean[:, required]).all() or not np.isfinite(std[:, required]).all():
        raise ValueError(f"matching normalization for {name} must be finite")
    if (std[:, required] <= 0).any():
        raise ValueError(f"matching normalization for {name} must have positive std")
    return mean[:, [0, 1, 6]].astype(np.float32), std[:, [0, 1, 6]].astype(np.float32)


class PatEmbedderLightning(L.LightningModule):
    def __init__(
        self,
        lr=3e-4,
        weight_decay=1e-4,
        max_leptons=4,
        max_jets=12,
        max_tops=4,
        max_ws=4,
        hidden_dim=512,
        num_layers=4,
        dropout=0.15,
        num_hlr_vars=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = PatEmbedderModel(
            max_leptons=max_leptons,
            max_jets=max_jets,
            max_tops=max_tops,
            max_ws=max_ws,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_hlr_vars=num_hlr_vars,
        )

        self.criterion = PatEmbedderLoss()

        self.lr = lr
        self.weight_decay = weight_decay

        self.lepton_label_map = {0: 0, -11: 1, 11: 2, -13: 3, 13: 4, -15: 5, 15: 6}
        self.neutrino_label_map = {0: 0, 12: 1, -12: 2, 14: 3, -14: 4, 16: 5, -16: 6}
        self.top_label_map = {0: 0, 6: 1, -6: 2}
        self.w_label_map = {0: 0, 24: 1, -24: 2}

        # Vectorised lookup tables: index = pid + _PID_OFFSET (handles PIDs in [-24, 24])
        _PID_OFFSET = 24
        _LUT_SIZE = 49  # covers -24 … +24
        self._pid_offset = _PID_OFFSET

        def _build_lut(pid_map: dict) -> torch.Tensor:
            lut = torch.zeros(_LUT_SIZE, dtype=torch.long)
            for pid, label in pid_map.items():
                lut[int(pid) + _PID_OFFSET] = label
            return lut

        self.register_buffer("_lep_lut", _build_lut(self.lepton_label_map), persistent=False)
        self.register_buffer("_neut_lut", _build_lut(self.neutrino_label_map), persistent=False)
        self.register_buffer("_top_lut", _build_lut(self.top_label_map), persistent=False)
        self.register_buffer("_w_lut", _build_lut(self.w_label_map), persistent=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


    def _prepare_targets(self, targets, masks_raw, batch_size):
        def move(t, *, dtype=None):
            return t.to(self.device, dtype=dtype, non_blocking=True)

        reg_tgt = {
            "leptons": targets["regression"]["leptons"]
                .to(self.device, non_blocking=True).view(batch_size, self.model.max_leptons, 7),
            "neutrinos": targets["regression"]["neutrinos"]
                .to(self.device, non_blocking=True).view(batch_size, self.model.max_leptons, 7),
            "jets": targets["regression"]["jets"]
                .to(self.device, non_blocking=True).view(batch_size, self.model.max_jets, 7),
            "tops": targets["regression"]["tops"]
                .to(self.device, non_blocking=True).view(batch_size, self.model.max_tops, 7),
            "ws": targets["regression"]["ws"]
                .to(self.device, non_blocking=True).view(batch_size, self.model.max_tops, 7),
        }

        masks = {k: v.to(self.device, non_blocking=True) for k, v in masks_raw.items()}

        _off = self._pid_offset
        lep_pids = move(targets["classification"]["leptons"], dtype=torch.long).flatten()
        neut_pids = move(targets["classification"]["neutrinos"], dtype=torch.long).flatten()
        jet_pids = move(targets["classification"]["jets"], dtype=torch.long).flatten()
        top_pids = move(targets["classification"]["tops"], dtype=torch.long).flatten()
        w_pids = move(targets["classification"]["ws"], dtype=torch.long).flatten()

        cls_tgt = {
            "lepton_classes": self._lep_lut[lep_pids + _off],
            "neutrino_classes": self._neut_lut[neut_pids + _off],
            "jet_classes": jet_class_labels(jet_pids),
            "top_classes": self._top_lut[top_pids + _off],
            "w_classes": self._w_lut[w_pids + _off],
        }

        assign_tgt = None
        if "assignments" in targets:
            assign_tgt = {
                k: v.to(self.device, dtype=torch.long, non_blocking=True).view(batch_size, -1)
                for k, v in targets["assignments"].items()
            }

        hlr_tgt = None
        if "high_level_regression" in targets:
            hlr_tgt = targets["high_level_regression"].to(
                self.device, dtype=torch.float32, non_blocking=True
            )

        return reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks

    def _step(self, batch, prefix):
        inputs = batch["features"].to(self.device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"]
        masks_raw = batch["masks"]
        bs = inputs.size(0)

        reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks = self._prepare_targets(
            targets, masks_raw, bs
        )

        outputs = self.model(inputs)
        reg_out = {
            "leptons": outputs[0], "neutrinos": outputs[1], "jets": outputs[2],
            "tops": outputs[3], "ws": outputs[4],
        }
        cls_out = {
            "lepton_logits": outputs[5], "neutrino_logits": outputs[6],
            "jet_logits": outputs[7], "top_logits": outputs[8], "w_logits": outputs[9],
        }
        assign_out = outputs[10]
        hlr_out = outputs[11]

        loss_dict = self.criterion(
            reg_out, cls_out, reg_tgt, cls_tgt, masks,
            assign_out, assign_tgt, hlr_out, hlr_tgt,
        )
        loss = loss_dict["total"]

        on_step = (prefix == "train")
        self.log(f"loss/{prefix}", loss.detach(), prog_bar=True, on_step=on_step, on_epoch=True)
        for sub in ["regression", "classification", "assignment", "hlr"]:
            val = loss_dict[sub]
            if isinstance(val, torch.Tensor):
                self.log(f"loss/{prefix}_{sub}", val.detach(), on_step=on_step, on_epoch=True)
        if prefix == "train":
            for sub in ["regression", "classification", "assignment", "hlr"]:
                w_key = f"task_weight/{sub}"
                if w_key in loss_dict:
                    self.log(w_key, loss_dict[w_key].detach(), on_step=False, on_epoch=True)
            self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")


class PatEmbedderLightningV5(PatEmbedderLightning):
    """V5: encoder-decoder model + sin/cos phi targets + Huber pT + presence loss + aux loss."""

    # Top class indices live at 0..N_TOP_CLASSES-1 in dn_label_enc; W class
    # indices are offset by N_TOP_CLASSES so the two heads have disjoint rows.
    N_TOP_CLASSES = 3

    def __init__(
        self,
        gamma=2.0,
        embed_dim=256,
        n_heads=4,
        n_enc_layers=6,
        n_dec_layers=4,
        phi_means=None,
        phi_stds=None,
        task_weights=None,
        aux_loss_weight=0.5,
        huber_delta=1.0,
        presence_loss_weight=1.0,
        dn_noise_scale=0.3,
        dn_loss_weight=1.0,
        mass_loss_weight=0.0,
        mass_huber_delta=20.0,
        phi_norm_weight=0.0,
        phi_angle_weight=0.0,
        reg_means=None,
        reg_stds=None,
        cls_empty_boost=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["phi_means", "phi_stds", "reg_means", "reg_stds"])

        self.model = PatEmbedderModelV3(
            max_leptons=kwargs.get("max_leptons", 4),
            max_jets=kwargs.get("max_jets", 12),
            max_tops=kwargs.get("max_tops", 4),
            max_ws=kwargs.get("max_ws", 4),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            dropout=kwargs.get("dropout", 0.1),
            num_hlr_vars=kwargs.get("num_hlr_vars", 0),
            reg_dim=REG_DIM_V5,
        )

        self.criterion = PatEmbedderLossV5(
            gamma=gamma,
            max_tops=kwargs.get("max_tops", 4),
            max_ws=kwargs.get("max_ws", 4),
            task_weights=task_weights,
            aux_loss_weight=aux_loss_weight,
            huber_delta=huber_delta,
            presence_loss_weight=presence_loss_weight,
            dn_loss_weight=dn_loss_weight,
            mass_loss_weight=mass_loss_weight,
            mass_huber_delta=mass_huber_delta,
            phi_norm_weight=phi_norm_weight,
            phi_angle_weight=phi_angle_weight,
            cls_empty_boost=cls_empty_boost,
        )

        self.dn_noise_scale = dn_noise_scale
        self.dn_loss_weight = dn_loss_weight
        self.mass_loss_weight = mass_loss_weight

        _phi_slot_sizes = {
            "leptons":   kwargs.get("max_leptons", 4),
            "neutrinos": kwargs.get("max_leptons", 4),
            "jets":      kwargs.get("max_jets", 12),
            "tops":      kwargs.get("max_tops", 4),
            "ws":        kwargs.get("max_ws", 4),
        }
        if phi_means is not None:
            for name, arr in phi_means.items():
                self.register_buffer(f"phi_mean_{name}", torch.tensor(arr, dtype=torch.float32))
        else:
            for name, n in _phi_slot_sizes.items():
                self.register_buffer(f"phi_mean_{name}", torch.zeros(n, dtype=torch.float32))
        if phi_stds is not None:
            for name, arr in phi_stds.items():
                self.register_buffer(f"phi_std_{name}", torch.tensor(arr, dtype=torch.float32))
        else:
            for name, n in _phi_slot_sizes.items():
                self.register_buffer(f"phi_std_{name}", torch.ones(n, dtype=torch.float32))

        # Per-slot regression stats for physics mass loss (pT, η, E columns only).
        # reg_means/reg_stds are (N, 7) arrays from the data cache; we store
        # columns [0, 1, 6] = (pT, η, E) as (N, 3) buffers.
        # Note: raw 7-dim cache layout is (pT, η, φ, px, py, pz, E) — E_n is
        # index 6, not 3 (index 3 is px_n).
        _reg_slot_sizes = {
            "leptons":   kwargs.get("max_leptons", 4),
            "neutrinos": kwargs.get("max_leptons", 4),
            "jets":      kwargs.get("max_jets", 12),
            "tops":      kwargs.get("max_tops", 4),
            "ws":        kwargs.get("max_ws", 4),
        }
        for name, n in _reg_slot_sizes.items():
            mean, std = _matching_stats_from_cache(reg_means, reg_stds, name, n)
            self.register_buffer(f"mass_reg_mean_{name}", torch.from_numpy(mean))
            self.register_buffer(f"mass_reg_std_{name}", torch.from_numpy(std))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-2)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _prepare_targets(self, targets, masks_raw, batch_size):
        reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks = PatEmbedderLightning._prepare_targets(
            self, targets, masks_raw, batch_size
        )

        new_reg_tgt = {}
        for k, v in reg_tgt.items():
            phi_std = getattr(self, f"phi_std_{k}", None)
            phi_mean = getattr(self, f"phi_mean_{k}", None)
            new_reg_tgt[k] = reg_target_to_v5(v, phi_mean, phi_std)

        return new_reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks

    def _build_dn_inputs(self, reg_tgt, cls_tgt):
        """Construct DN-DETR query inputs from clean GT.

        Concatenates max_tops top slots followed by max_ws W slots so the
        decoder's downstream slicing (`dn_dec[:, :max_tops]` / `[max_tops:]`)
        is correct. Adds Gaussian noise of stdev `self.dn_noise_scale` to the
        regression GT. Class indices are mapped to disjoint rows of
        `dn_label_enc`: top labels live at 0..N_TOP_CLASSES-1, W labels at
        N_TOP_CLASSES..2*N_TOP_CLASSES-1.
        """
        top_reg = reg_tgt["tops"]
        w_reg = reg_tgt["ws"]
        B = top_reg.size(0)
        max_tops = self.model.max_tops
        max_ws = self.model.max_ws

        dn_reg = torch.cat([top_reg, w_reg], dim=1)
        if self.dn_noise_scale > 0.0:
            dn_reg = dn_reg + torch.randn_like(dn_reg) * self.dn_noise_scale

        top_cls = cls_tgt["top_classes"].view(B, max_tops)
        w_cls = cls_tgt["w_classes"].view(B, max_ws) + self.N_TOP_CLASSES
        dn_cls = torch.cat([top_cls, w_cls], dim=1).long()
        return dn_reg, dn_cls

    def _step(self, batch, prefix):
        inputs = batch["features"].to(self.device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"]
        masks_raw = batch["masks"]
        bs = inputs.size(0)

        reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks = self._prepare_targets(targets, masks_raw, bs)

        enc_padding_mask = detector_enc_padding_mask(
            inputs, self.model.max_leptons, self.model.max_jets
        )

        dn_reg = None
        dn_cls = None
        if prefix == "train" and self.dn_loss_weight > 0.0:
            dn_reg, dn_cls = self._build_dn_inputs(reg_tgt, cls_tgt)

        outputs = self.model(
            inputs,
            enc_padding_mask=enc_padding_mask,
            dn_reg=dn_reg,
            dn_cls=dn_cls,
            compute_aux=self.criterion.aux_loss_weight > 0.0,
        )
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
        assign_out = outputs[10]
        hlr_out = outputs[11]
        aux_head_outputs = outputs[12]
        dn_outputs = outputs[13]

        # Build per-slot reg stats dicts for physics mass loss (only when enabled)
        mass_reg_means = None
        mass_reg_stds = None
        matching_reg_means = {
            k: getattr(self, f"mass_reg_mean_{k}")
            for k in ("leptons", "neutrinos", "jets", "tops", "ws")
        }
        matching_reg_stds = {
            k: getattr(self, f"mass_reg_std_{k}")
            for k in ("leptons", "neutrinos", "jets", "tops", "ws")
        }
        if self.mass_loss_weight > 0.0:
            mass_reg_means = {
                k: getattr(self, f"mass_reg_mean_{k}") for k in ("leptons", "neutrinos", "jets", "tops", "ws")
            }
            mass_reg_stds = {
                k: getattr(self, f"mass_reg_std_{k}") for k in ("leptons", "neutrinos", "jets", "tops", "ws")
            }

        loss_dict = self.criterion.forward_with_aux(
            reg_out,
            cls_out,
            reg_tgt,
            cls_tgt,
            masks,
            assign_out,
            assign_tgt,
            hlr_out,
            hlr_tgt,
            aux_head_outputs=aux_head_outputs,
            dn_outputs=dn_outputs,
            dn_targets_reg=reg_tgt if dn_outputs is not None else None,
            dn_targets_cls=cls_tgt if dn_outputs is not None else None,
            dn_masks=masks if dn_outputs is not None else None,
            reg_means=mass_reg_means,
            reg_stds=mass_reg_stds,
            matching_reg_means=matching_reg_means,
            matching_reg_stds=matching_reg_stds,
        )
        loss = loss_dict["total"]

        on_step = (prefix == "train")
        self.log(f"loss/{prefix}", loss.detach(), prog_bar=True, on_step=on_step, on_epoch=True)
        for sub in ["regression", "classification", "assignment", "hlr"]:
            val = loss_dict.get(sub)
            if isinstance(val, torch.Tensor):
                self.log(f"loss/{prefix}_{sub}", val.detach(), on_step=on_step, on_epoch=True)
        if "aux" in loss_dict and isinstance(loss_dict["aux"], torch.Tensor):
            self.log(f"loss/{prefix}_aux", loss_dict["aux"].detach(), on_step=on_step, on_epoch=True)
        if "dn" in loss_dict and isinstance(loss_dict["dn"], torch.Tensor):
            self.log(f"loss/{prefix}_dn", loss_dict["dn"].detach(), on_step=on_step, on_epoch=True)
        if "mass" in loss_dict and isinstance(loss_dict["mass"], torch.Tensor):
            self.log(f"loss/{prefix}_mass", loss_dict["mass"].detach(), on_step=on_step, on_epoch=True)

        if prefix == "train":
            for sub in ["regression", "classification", "assignment", "hlr"]:
                w_key = f"task_weight/{sub}"
                if w_key in loss_dict:
                    self.log(w_key, loss_dict[w_key].detach(), on_step=False, on_epoch=True)
            self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss


class MLflowBestModelUploader(L.Callback):
    """Upload checkpoints to MLflow under **fixed** artifact names so they overwrite.

    Lightning saves ``best-epoch=..-step=..ckpt`` locally; logging that path
    verbatim creates a new MLflow file each time.  This callback copies to
    ``best-model/best.ckpt`` (and ``last-model/last.ckpt`` at end) so the
    remote path stays constant.

    Parameters
    ----------
    upload_every_n_epochs
        How often (in validation-epoch intervals) to upload the best checkpoint.
        ``1`` = every epoch, ``5`` = every fifth, ``0`` = never during training
        (still uploads once at ``on_train_end``).
    """

    _ART_BEST = "best-model"
    _ART_LAST = "last-model"
    _NAME_BEST = "best.ckpt"
    _NAME_LAST = "last.ckpt"

    def __init__(self, upload_every_n_epochs: int = 1):
        super().__init__()
        self.upload_every_n_epochs = max(0, upload_every_n_epochs)
        self._last_uploaded_best_path: str | None = None

    @staticmethod
    def _log_ckpt_copy(client, run_id: str, src_path: str, artifact_dir: str, dest_name: str) -> None:
        if not src_path or not Path(src_path).is_file():
            return
        with tempfile.TemporaryDirectory() as tmp:
            fixed = Path(tmp) / dest_name
            shutil.copy2(src_path, fixed)
            client.log_artifact(run_id, str(fixed), artifact_dir)

    def _mlflow_client_and_run_id(self, trainer: L.Trainer):
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return None, None
        return logger.experiment, logger.run_id

    def _upload_best_if_changed(self, trainer: L.Trainer) -> None:
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None or not hasattr(ckpt_cb, "best_model_path"):
            return
        best_path = ckpt_cb.best_model_path
        if not best_path or best_path == self._last_uploaded_best_path:
            return
        client, run_id = self._mlflow_client_and_run_id(trainer)
        if client is None or run_id is None:
            return
        try:
            self._log_ckpt_copy(client, run_id, best_path, self._ART_BEST, self._NAME_BEST)
            self._last_uploaded_best_path = best_path
        except Exception as exc:
            print(f"[MLflowBestModelUploader] best upload failed: {exc}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if self.upload_every_n_epochs <= 0:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.upload_every_n_epochs != 0:
            return
        self._upload_best_if_changed(trainer)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Always push final best + last under fixed names (covers ``upload_every_n_epochs=0``)."""
        client, run_id = self._mlflow_client_and_run_id(trainer)
        if client is None or run_id is None:
            return
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None:
            return
        try:
            best_path = getattr(ckpt_cb, "best_model_path", "") or ""
            last_path = getattr(ckpt_cb, "last_model_path", "") or ""
            self._log_ckpt_copy(client, run_id, best_path, self._ART_BEST, self._NAME_BEST)
            self._log_ckpt_copy(client, run_id, last_path, self._ART_LAST, self._NAME_LAST)
        except Exception as exc:
            print(f"[MLflowBestModelUploader] final upload failed: {exc}")


def load_checkpoint_weights(pl_model, path):
    """Initialise ``pl_model`` from a training checkpoint, weights only.

    Unlike ``trainer.fit(ckpt_path=...)`` this leaves the optimiser moments and the LR
    schedule behind: the schedule of a finished run sits at its cosine floor (1% of the
    peak), so a restored one would fine-tune at 3e-6. Checkpoints written by a
    ``torch.compile``d run carry a ``model._orig_mod.`` prefix, which is mapped back onto
    the eager module so either kind loads.

    Returns the training state the weights come from, for the run log.
    """
    # weights_only=True rejects the VXFull checkpoints: `save_hyperparameters` stores the
    # numpy feature statistics of that model. The path is a CLI argument of our own training run.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "model._orig_mod."
    state_dict = {
        ("model." + key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in checkpoint["state_dict"].items()
    }
    pl_model.load_state_dict(state_dict, strict=True)
    return {"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}


def log_run_provenance(logger, argv, checkpoint_path=None, checkpoint_state=None):
    """Record on the MLflow run what produced it: the command line, and — for a fine-tune —
    the checkpoint the weights came from.

    MLflow tags the run with the commit of the working tree at the moment it starts, so a run
    launched before its own code was committed carries the previous commit, and
    ``save_hyperparameters`` stores module arguments only. Without this, a fine-tune is
    indistinguishable from a run from scratch. The command line goes through the tracking client
    rather than ``logger.log_hyperparams``, which truncates values at 250 characters.
    """
    client, run_id = logger.experiment, logger.run_id
    client.set_tag(run_id, "cli", shlex.join(argv))
    if checkpoint_path is not None:
        client.log_param(run_id, "init_from_checkpoint", checkpoint_path)
        for key, value in (checkpoint_state or {}).items():
            client.log_param(run_id, f"init_from_checkpoint_{key}", value)


def _detect_accelerator():
    if torch.backends.mps.is_available():
        return "mps", 1
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        return "cuda", "auto"
    return "cpu", "auto"


def create_trainer(name, max_epochs=10, enable_progress_bar=True,
                   enable_model_summary=True, experiment_name="event-reconstruction",
                   run_name=None, swa_lrs=None, swa_epoch_start=0.8,
                   tracking_uri="mlruns",
                   upload_best_every_n_epochs=1,
                   precision="32-true", accelerator="auto", checkpoint_dir=None):
    """Create a PyTorch Lightning trainer with callbacks and MLflow logging.

    Parameters
    ----------
    upload_best_every_n_epochs
        Upload the current best-val-loss checkpoint to MLflow every N epochs as
        ``best-model/best.ckpt`` (same path each time, overwrites).  Set to ``0``
        to upload only once at the end of training (best + ``last-model/last.ckpt``).
    """
    detected_accelerator, detected_devices = _detect_accelerator()
    devices = detected_devices if accelerator == "auto" else 1
    if accelerator == "auto":
        accelerator = detected_accelerator
    callbacks = []
    callbacks.append(
        L.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-epoch={epoch:02d}-step={step}",
            auto_insert_metric_name=False,
        )
    )
    callbacks.append(
        MLflowBestModelUploader(upload_every_n_epochs=upload_best_every_n_epochs),
    )
    if enable_progress_bar:
        callbacks.append(L.callbacks.TQDMProgressBar(refresh_rate=50))
    if swa_lrs is not None:
        callbacks.append(L.callbacks.StochasticWeightAveraging(
            swa_lrs=swa_lrs,
            swa_epoch_start=swa_epoch_start,
        ))
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=L.loggers.MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            # Avoid one artifact dir per local checkpoint name; use MLflowBestModelUploader.
            log_model=False,
            run_name=run_name,
        ),
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
    )
    return trainer
