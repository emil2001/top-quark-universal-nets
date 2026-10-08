"""PyTorch Lightning wrappers for GATr PatEmbedder models."""

import torch

from .data import detector_enc_padding_mask
from .lightning import PatEmbedderLightningV5
from .models import REG_DIM_V5
from .models_gatr import PatEmbedderModelVXFull


class _GATrLightningBase(PatEmbedderLightningV5):
    """Internal feature reconstruction and checkpoint contract shared by LGaTr.

    Additional constructor args vs V5:
        feat_mean:   np.ndarray (input_dim,) — feature means from cache (optional when
                     loading a checkpoint that already stores feat_mean_buf)
        feat_std:    same shape as feat_mean — feature stds from cache
        ga_channels: int — GATr working channels (default 32)
        ga_heads:    int — GATr attention heads (default 4)
        n_enc_layers:int — number of GATrBlocks (default 6)
    """

    def __init__(
        self,
        feat_mean=None,
        feat_std=None,
        ga_channels: int = 32,
        ga_heads: int = 4,
        n_enc_layers: int = 6,
        ga_dropout: float | None = None,
        ga_gp_mid_channels: int | None = None,
        ga_attn_channels: int | None = None,
        fully_connected_subgroup: bool = False,
        checkpoint_ga_blocks: bool = False,
        spurions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # feat_mean / feat_std: omit only when loading from a checkpoint that already
        # contains feat_mean_buf / feat_std_buf (older runs used save_hyperparameters
        # with ignore=[...], so Lightning could not pass them back into __init__).
        self.save_hyperparameters(ignore=["phi_means", "phi_stds"])
        feat_dim = self.model.input_dim  # V5 PatEmbedderModelV3 before swap

        if feat_mean is not None and feat_std is not None:
            self.register_buffer(
                "feat_mean_buf",
                torch.tensor(feat_mean, dtype=torch.float32).unsqueeze(0),
            )
            self.register_buffer(
                "feat_std_buf",
                torch.tensor(feat_std, dtype=torch.float32).unsqueeze(0),
            )
        else:
            # Placeholders overwritten by load_state_dict when loading a checkpoint.
            self.register_buffer(
                "feat_mean_buf",
                torch.zeros(1, feat_dim, dtype=torch.float32),
            )
            self.register_buffer(
                "feat_std_buf",
                torch.ones(1, feat_dim, dtype=torch.float32),
            )
        # Feature reconstruction now has a physical-p4 contract.  Old weights
        # learned a mixed log/physical input and must not be reinterpreted.
        self.register_buffer("_gatr_input_contract_version", torch.tensor(2, dtype=torch.int64))

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        version = checkpoint.get("state_dict", {}).get("_gatr_input_contract_version")
        if version is None or int(version) != 2:
            raise RuntimeError(
                "This checkpoint predates GATr physical-p4 contract v2 and cannot be loaded "
                "safely. Evaluate it with the historical code or retrain under contract v2."
            )
        super().on_load_checkpoint(checkpoint)

    def _prepare_4mom(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-normalise features and extract raw 4-momenta + raw feature tensor.

        Returns:
            x_4mom: (B, 17, 4)  raw (E, px, py, pz) in GeV
            x_raw:  (B, 130)    un-normalised flat features
        """
        x = batch["features"].to(self.device, dtype=torch.float32, non_blocking=True)
        x_raw = x * self.feat_std_buf + self.feat_mean_buf    # (B, 130) un-normalised

        B = x_raw.size(0)

        # ---- MET (indices 0, 1 = log1p(pt), phi) ----
        met_pt = torch.expm1(x_raw[:, 0]).clamp_min(0)
        met_phi = x_raw[:, 1]
        met_E   = met_pt
        met_px  = met_pt * torch.cos(met_phi)
        met_py  = met_pt * torch.sin(met_phi)
        met_pz  = torch.zeros_like(met_pt)
        met_4mom = torch.stack([met_E, met_px, met_py, met_pz], dim=-1).unsqueeze(1)  # (B,1,4)

        # ---- Leptons: each slot = [log1p(pt),eta,phi,px,py,pz,log1p(E),flavor] ----
        lep_raw = x_raw[:, 2:2 + self.model.max_leptons * 8].view(B, self.model.max_leptons, 8)
        # indices: E=6, px=3, py=4, pz=5
        lep_4mom = torch.cat([lep_raw[:, :, 6:7].expm1().clamp_min(0), lep_raw[:, :, 3:6]], dim=-1)

        # ---- Jets: each slot = [pt,eta,phi,px,py,pz,E,btag] ----
        jet_raw = x_raw[:, 2 + self.model.max_leptons * 8:].view(B, self.model.max_jets, 8)
        jet_4mom = torch.cat([jet_raw[:, :, 6:7].expm1().clamp_min(0), jet_raw[:, :, 3:6]], dim=-1)

        x_4mom = torch.cat([met_4mom, lep_4mom, jet_4mom], dim=1)  # (B, 17, 4)
        # Padded detector slots have no physical momentum.  Apply the mask after
        # denormalisation so cache means cannot turn padding into fake particles.
        padding_mask = detector_enc_padding_mask(x, self.model.max_leptons, self.model.max_jets)
        x_4mom = x_4mom.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x_4mom, x_raw


class PatEmbedderLightningVXFull(_GATrLightningBase):
    """VX-full: single-stack GATr encoder over all 25 tokens, no DETR decoder.

    Inherits feat_mean/feat_std buffers, _prepare_4mom, configure_optimizers,
    and _prepare_targets from the internal GATr base and V5.  Only the model is swapped and
    _step is simplified to drop DN-DETR / aux losses.

    Additional constructor arguments:
        n_enc_layers: default 8 when constructing the wrapper directly
        reg_means:    dict[str, ndarray(N,7)] — per-slot regression means from
                      data cache (raw 7-dim layout: pT,eta,phi,px,py,pz,E).
                      Columns [0,1,6] (pT, eta, E) are stored as model buffers
                      so the equivariant readout can z-score its output to
                      match the z-scored loss targets.
        reg_stds:     same layout as reg_means — per-slot stds.
    """

    def __init__(
        self,
        n_enc_layers: int = 8,
        reg_means=None,
        reg_stds=None,
        parton_reg_head_type: str = "grade1",
        parton_reg_channels: int = 4,
        parton_reg_residual_scale: float = 0.25,
        **kwargs,
    ):

        super().__init__(n_enc_layers=n_enc_layers, reg_means=reg_means, reg_stds=reg_stds, **kwargs)

        # Install the single-stack LGaTr model
        ga_channels = kwargs.get("ga_channels", 32)
        ga_heads    = kwargs.get("ga_heads", 4)
        self.model  = PatEmbedderModelVXFull(
            max_leptons=kwargs.get("max_leptons", 4),
            max_jets=kwargs.get("max_jets", 12),
            max_tops=kwargs.get("max_tops", 4),
            max_ws=kwargs.get("max_ws", 4),
            ga_channels=ga_channels,
            ga_heads=ga_heads,
            n_enc_layers=n_enc_layers,
            dropout=kwargs.get("dropout", 0.1),
            scalar_dropout=kwargs.get("ga_dropout", None),
            gp_mid_channels=kwargs.get("ga_gp_mid_channels", None),
            attn_channels=kwargs.get("ga_attn_channels", None),
            fully_connected_subgroup=kwargs.get("fully_connected_subgroup", False),
            checkpoint_ga_blocks=kwargs.get("checkpoint_ga_blocks", False),
            spurions=kwargs.get("spurions", False),
            num_hlr_vars=kwargs.get("num_hlr_vars", 0),
            reg_dim=REG_DIM_V5,
            parton_reg_head_type=parton_reg_head_type,
            parton_reg_channels=parton_reg_channels,
            parton_reg_residual_scale=parton_reg_residual_scale,
        )

        self._set_model_reg_stats(reg_means, reg_stds)

        self.hparams.parton_reg_head_type = parton_reg_head_type
        self.hparams.parton_reg_channels = parton_reg_channels
        self.hparams.parton_reg_residual_scale = parton_reg_residual_scale
        self.criterion.signed_energy = True
        self.register_buffer("_gatr_output_contract_version", torch.tensor(3, dtype=torch.int64))

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        output_version = state_dict.get("_gatr_output_contract_version")
        model_version = state_dict.get("model._vx_full_contract_version")
        if (
            output_version is None
            or model_version is None
            or int(output_version) != 3
            or int(model_version) != 3
        ):
            raise RuntimeError(
                "This checkpoint predates GATr signed-energy output contract v3 and cannot be "
                "loaded safely. Evaluate it with the historical code or retrain under contract v3."
            )
        super().on_load_checkpoint(checkpoint)

    def _set_model_reg_stats(self, reg_means, reg_stds):
        """Populate the model's z-scoring buffers from data-cache statistics.

        Columns [0, 1, 6] = (pT, η, E) in the raw 7-dim cache layout
        (pT, η, φ, px, py, pz, E) — E_n is index 6, not 3 (=px_n). These
        buffers z-score the model's own raw grade-1 readout
        (``models_gatr.py::_zscore_v5``) symmetrically with the training
        target; expm1 applies only in the decoding/reconstruction direction.
        """
        import numpy as np
        mapping = {
            "leptons": "lep", "neutrinos": "neu", "jets": "jet",
            "tops": "top", "ws": "w",
        }
        for full_name, short_name in mapping.items():
            mu = torch.tensor(
                np.asarray(reg_means[full_name])[:, [0, 1, 6]],
                dtype=torch.float32,
            )
            std = torch.tensor(
                np.asarray(reg_stds[full_name])[:, [0, 1, 6]],
                dtype=torch.float32,
            )
            getattr(self.model, f"_reg_mean_{short_name}").copy_(mu)
            getattr(self.model, f"_reg_std_{short_name}").copy_(std)

    def _set_lightning_mass_reg_stats(self, reg_means, reg_stds):
        """Overwrite the parent's mass_reg_mean_* / mass_reg_std_* buffers."""
        import numpy as np
        for name in ("leptons", "neutrinos", "jets", "tops", "ws"):
            mu3 = torch.tensor(
                np.asarray(reg_means[name])[:, [0, 1, 6]].astype(np.float32)
            )
            sd3 = torch.tensor(
                np.asarray(reg_stds[name])[:, [0, 1, 6]].astype(np.float32)
            )
            getattr(self, f"mass_reg_mean_{name}").copy_(mu3)
            getattr(self, f"mass_reg_std_{name}").copy_(sd3)

    def _step(self, batch: dict, prefix: str) -> torch.Tensor:
        inputs    = batch["features"].to(self.device).float()
        targets   = batch["targets"]
        masks_raw = batch["masks"]
        bs        = inputs.size(0)

        reg_tgt, cls_tgt, assign_tgt, hlr_tgt, masks = self._prepare_targets(
            targets, masks_raw, bs
        )
        enc_padding_mask = detector_enc_padding_mask(
            inputs, self.model.max_leptons, self.model.max_jets
        )

        x_4mom, x_raw = self._prepare_4mom(batch)

        outputs = self.model(
            inputs,
            x_4mom=x_4mom,
            x_raw=x_raw,
            enc_padding_mask=enc_padding_mask,
        )

        reg_out = {
            "leptons":   outputs[0], "neutrinos": outputs[1],
            "jets":      outputs[2], "tops":      outputs[3], "ws": outputs[4],
        }
        cls_out = {
            "lepton_logits":   outputs[5], "neutrino_logits": outputs[6],
            "jet_logits":      outputs[7], "top_logits":      outputs[8],
            "w_logits":        outputs[9],
        }
        assign_out = outputs[10]
        hlr_out    = outputs[11]

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
                k: getattr(self, f"mass_reg_mean_{k}")
                for k in ("leptons", "neutrinos", "jets", "tops", "ws")
            }
            mass_reg_stds = {
                k: getattr(self, f"mass_reg_std_{k}")
                for k in ("leptons", "neutrinos", "jets", "tops", "ws")
            }

        # Base forward(): presence keys in assign_out activate BCE loss.
        # No aux, no DN for VX-full.
        loss_dict = self.criterion(
            reg_out, cls_out, reg_tgt, cls_tgt, masks,
            assign_out, assign_tgt, hlr_out, hlr_tgt,
            aux_head_outputs=None,
            dn_outputs=None,
            reg_means=mass_reg_means,
            reg_stds=mass_reg_stds,
            matching_reg_means=matching_reg_means,
            matching_reg_stds=matching_reg_stds,
        )
        loss = loss_dict["total"]

        on_step = (prefix == "train")
        self.log(f"loss/{prefix}", loss.detach(), prog_bar=True,
                 on_step=on_step, on_epoch=True)
        for sub in ["regression", "classification", "assignment", "hlr"]:
            val = loss_dict.get(sub)
            if isinstance(val, torch.Tensor):
                self.log(f"loss/{prefix}_{sub}", val.detach(),
                         on_step=on_step, on_epoch=True)
        if "mass" in loss_dict and isinstance(loss_dict["mass"], torch.Tensor):
            self.log(f"loss/{prefix}_mass", loss_dict["mass"].detach(),
                     on_step=on_step, on_epoch=True)
        if prefix == "train":
            for sub in ["regression", "classification", "assignment", "hlr"]:
                w_key = f"task_weight/{sub}"
                if w_key in loss_dict:
                    self.log(w_key, loss_dict[w_key].detach(), on_step=False, on_epoch=True)
            self.log("lr", self.optimizers().param_groups[0]["lr"],
                     on_step=True, on_epoch=False)

        return loss
