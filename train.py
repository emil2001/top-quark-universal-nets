import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pat_embedder import (
    DEFAULT_HIGH_LEVEL_BRANCHES,
    SEED,
    PatEmbedderLightning,
    PatEmbedderLightningV5,
    PatEmbedderLightningVXFull,
    create_trainer,
    load_cache,
    load_checkpoint_weights,
    log_run_provenance,
    seed_everything,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Train PatEmbedder from a cached pickle.")

    p.add_argument("--cache-path", type=str, default=None, help="Path to cache pickle. If set, overrides --cache-dir/--cache-filename.")
    p.add_argument("--cache-dir", type=str, default="data", help="Cache directory (used when --cache-path not set).")
    p.add_argument("--cache-filename", type=str, default="level1_cached_full.pkl", help="Cache filename (used when --cache-path not set).")

    p.add_argument("--version", type=str,
                   choices=["v1", "v5", "vx-full"],
                   default="v5", help="Model/Loss version.")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for best/last checkpoints.")
    p.add_argument("--init-from-checkpoint", type=str, default=None,
                   help="Fine-tune: start from the weights of this checkpoint. The optimiser and "
                        "the LR schedule start fresh, so the schedule spans --max-epochs.")
    p.add_argument("--run-name", type=str, default=None, help="MLflow run name.")
    p.add_argument("--experiment-name", type=str, default="event-reconstruction")
    p.add_argument("--tracking-uri", type=str, default="mlruns")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Lightning precision mode. Try bf16-mixed on Ampere/Hopper, 16-mixed on V100.",
    )
    p.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap the inner torch module with torch.compile.")
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode used with --compile-model.")

    # Common geometry
    p.add_argument("--max-leptons", type=int, default=4)
    p.add_argument("--max-jets", type=int, default=12)
    p.add_argument("--max-tops", type=int, default=4)
    p.add_argument("--max-ws", type=int, default=4)

    # V1 (MLP backbone)
    p.add_argument("--v1-hidden-dim", type=int, default=512, help="Hidden dim for V1 MLP backbone.")
    p.add_argument("--v1-num-layers", type=int, default=4, help="Number of MLP blocks for V1.")

    p.add_argument("--gamma", type=float, default=2.0, help="Focal gamma (DETR/LGaTr).")

    # V5 / LGaTr shared (encoder-decoder)
    p.add_argument("--v5-embed-dim", type=int, default=256)
    p.add_argument("--v5-n-heads", type=int, default=4)
    p.add_argument("--v5-n-enc-layers", type=int, default=6)
    p.add_argument("--v5-n-dec-layers", type=int, default=4)
    p.add_argument("--task-weights", type=str, default="1,1,2,5", help="Comma-separated weights: reg,cls,assign,hlr.")
    p.add_argument("--aux-loss-weight", type=float, default=0.5)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--presence-loss-weight", type=float, default=1.0)
    p.add_argument("--mass-loss-weight", type=float, default=0.0,
                   help="Weight for physics-informed decay chain mass loss (0 = disabled).")
    p.add_argument("--mass-huber-delta", type=float, default=20.0,
                   help="Huber delta (GeV) for invariant mass constraints.")
    p.add_argument("--phi-norm-weight", type=float, default=0.0,
                   help="Weight for sin²φ + cos²φ = 1 soft constraint (0 = disabled).")
    p.add_argument("--phi-angle-weight", type=float, default=0.0,
                   help="Weight for explicit atan2(sinφ,cosφ) angular regression loss.")

    # LGaTr specific (GA encoder)
    p.add_argument("--ga-channels", type=int, default=32, help="GATr working channels (vx-full).")
    p.add_argument("--ga-heads", type=int, default=4, help="GATr attention heads (vx-full).")
    p.add_argument("--ga-dropout", type=float, default=None,
                   help="Grade dropout inside GATr blocks. Defaults to --dropout.")
    p.add_argument("--ga-gp-mid-channels", type=int, default=None,
                   help="Intermediate channels for GATr geometric-product layers.")
    p.add_argument("--ga-attn-channels", type=int, default=None,
                   help="Per-head attention channels for GATr attention.")
    p.add_argument("--fully-connected-subgroup", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Use 10 SO+(1,3) equivariant linear basis maps instead of 5.")
    p.add_argument("--spurions", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Append beam/time reference multivector tokens to VX encoders.")
    p.add_argument("--checkpoint-ga-blocks", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Use activation checkpointing inside VX GATr blocks.")
    p.add_argument("--parton-reg-head-type", type=str, default="grade1",
                   choices=["grade1", "strict-equivariant", "hybrid-residual"],
                   help="VX-full top/W regression head variant.")
    p.add_argument("--parton-reg-channels", type=int, default=4,
                   help="Number of equivariant grade-1 candidates for strict parton regression.")
    p.add_argument("--parton-reg-residual-scale", type=float, default=0.25,
                   help="Tanh residual scale for hybrid parton regression.")

    # HLR
    p.add_argument("--num-hlr-vars", type=int, default=None, help="Number of HLR variables. Default: len(DEFAULT_HIGH_LEVEL_BRANCHES). Use 0 to disable.")
    p.add_argument("--upload-best-every-n-epochs", type=int, default=1, help="Upload best to MLflow as best-model/best.ckpt every N val epochs; 0 = only at end (+ last-model/last.ckpt).")

    return p.parse_args()


def _resolve_cache(args):
    if args.cache_path:
        cache_path = Path(args.cache_path)
        return str(cache_path.parent), cache_path.name
    return args.cache_dir, args.cache_filename


def _parse_task_weights(s):
    parts = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(parts) != 4:
        raise ValueError("--task-weights must have 4 comma-separated floats: reg,cls,assign,hlr")
    return parts


def main():
    args = _parse_args()
    seed_everything(args.seed)

    cache_dir, cache_filename = _resolve_cache(args)
    data = load_cache(cache_dir=cache_dir, filename=cache_filename)
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    reg_means = data["reg_means"]
    reg_stds = data["reg_stds"]

    num_hlr_vars = args.num_hlr_vars
    if num_hlr_vars is None:
        num_hlr_vars = len(DEFAULT_HIGH_LEVEL_BRANCHES)

    if args.version == "v1":
        pl_model = PatEmbedderLightning(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_leptons=args.max_leptons,
            max_jets=args.max_jets,
            max_tops=args.max_tops,
            max_ws=args.max_ws,
            hidden_dim=args.v1_hidden_dim,
            num_layers=args.v1_num_layers,
            dropout=args.dropout,
            num_hlr_vars=num_hlr_vars,
        )
    elif args.version == "vx-full":
        phi_means = {k: np.asarray(reg_means[k])[:, 2] for k in reg_means}
        phi_stds = {k: np.asarray(reg_stds[k])[:, 2] for k in reg_stds}
        task_weights = _parse_task_weights(args.task_weights)

        pl_model = PatEmbedderLightningVXFull(
            feat_mean=data["feat_mean"],
            feat_std=data["feat_std"],
            ga_channels=args.ga_channels,
            ga_heads=args.ga_heads,
            n_enc_layers=args.v5_n_enc_layers,
            ga_dropout=args.ga_dropout,
            ga_gp_mid_channels=args.ga_gp_mid_channels,
            ga_attn_channels=args.ga_attn_channels,
            fully_connected_subgroup=args.fully_connected_subgroup,
            checkpoint_ga_blocks=args.checkpoint_ga_blocks,
            spurions=args.spurions,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_leptons=args.max_leptons,
            max_jets=args.max_jets,
            max_tops=args.max_tops,
            max_ws=args.max_ws,
            dropout=args.dropout,
            num_hlr_vars=num_hlr_vars,
            gamma=args.gamma,
            phi_means=phi_means,
            phi_stds=phi_stds,
            task_weights=task_weights,
            huber_delta=args.huber_delta,
            mass_loss_weight=args.mass_loss_weight,
            mass_huber_delta=args.mass_huber_delta,
            phi_norm_weight=args.phi_norm_weight,
            phi_angle_weight=args.phi_angle_weight,
            reg_means=reg_means,
            reg_stds=reg_stds,
            parton_reg_head_type=args.parton_reg_head_type,
            parton_reg_channels=args.parton_reg_channels,
            parton_reg_residual_scale=args.parton_reg_residual_scale,
        )
    else:
        # v5
        phi_means = {k: np.asarray(reg_means[k])[:, 2] for k in reg_means}
        phi_stds = {k: np.asarray(reg_stds[k])[:, 2] for k in reg_stds}
        task_weights = _parse_task_weights(args.task_weights)

        pl_model = PatEmbedderLightningV5(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_leptons=args.max_leptons,
            max_jets=args.max_jets,
            max_tops=args.max_tops,
            max_ws=args.max_ws,
            dropout=args.dropout,
            num_hlr_vars=num_hlr_vars,
            gamma=args.gamma,
            embed_dim=args.v5_embed_dim,
            n_heads=args.v5_n_heads,
            n_enc_layers=args.v5_n_enc_layers,
            n_dec_layers=args.v5_n_dec_layers,
            phi_means=phi_means,
            phi_stds=phi_stds,
            task_weights=task_weights,
            aux_loss_weight=args.aux_loss_weight,
            huber_delta=args.huber_delta,
            presence_loss_weight=args.presence_loss_weight,
            mass_loss_weight=args.mass_loss_weight,
            mass_huber_delta=args.mass_huber_delta,
            phi_norm_weight=args.phi_norm_weight,
            phi_angle_weight=args.phi_angle_weight,
            reg_means=reg_means,
            reg_stds=reg_stds,
        )

    init_source = None
    if args.init_from_checkpoint:
        init_source = load_checkpoint_weights(pl_model, args.init_from_checkpoint)
        print(f"Initialised weights from {args.init_from_checkpoint} ({init_source})")

    if args.compile_model:
        pl_model.model = torch.compile(pl_model.model, mode=args.compile_mode)

    trainer = create_trainer(
        "pat_embedder",
        max_epochs=args.max_epochs,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enable_progress_bar=True,
        enable_model_summary=True,
        tracking_uri=args.tracking_uri,
        upload_best_every_n_epochs=args.upload_best_every_n_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        checkpoint_dir=args.checkpoint_dir,
    )

    log_run_provenance(trainer.logger, sys.argv, args.init_from_checkpoint, init_source)

    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
