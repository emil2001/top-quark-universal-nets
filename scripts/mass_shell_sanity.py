#!/usr/bin/env python
"""Check the fraction of space-like regression predictions using the checkpoint energy contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import infer_and_plot as ip  # noqa: E402

from pat_embedder import load_cache  # noqa: E402
from pat_embedder.evaluation import _uses_signed_v5_energy  # noqa: E402
from pat_embedder.losses import _lorentz_mass_sq, _v5_to_cartesian  # noqa: E402

PARTICLES = ("tops", "ws")


def _build_args(version: str, checkpoint: str, device: str, max_samples: int,
                 cache_path: str, split: str) -> SimpleNamespace:
    """Minimal namespace with the fields infer_and_plot helpers read."""
    return SimpleNamespace(
        version=version,
        checkpoint=checkpoint,
        run_name=None,
        run_id=None,
        experiment_name="event-reconstruction",
        tracking_uri="mlruns",
        checkpoint_preference="best",
        weights_only=False,
        strict_load=False,
        cache_path=cache_path,
        cache_dir="data",
        cache_filename="level1_cached_full.pkl",
        split=split,
        device=device,
        max_samples=max_samples,
    )


def compute_mass_shell(pl_model, version: str, loader, device: torch.device,
                        max_samples: int | None) -> dict:
    pl_model = pl_model.to(device).eval()
    pl_model.model.eval()
    # Energy parameterisation is a property of the checkpoint, not of this
    # script: VX contract v3 emits signed-log1p(E), everything
    # older emits log1p(E).
    signed_energy = _uses_signed_v5_energy(pl_model)
    m_sq_by_particle = {p: [] for p in PARTICLES}
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and seen >= max_samples:
                break
            remaining = None if max_samples is None else max_samples - seen
            if remaining is not None and batch["features"].size(0) > remaining:
                batch = ip._slice_batch(batch, remaining)
            moved = ip._move_batch_to_device(batch, device)
            inputs = moved["features"]
            targets = moved["targets"]
            masks_raw = moved["masks"]
            batch_size = inputs.size(0)

            _, _, _, _, masks = pl_model._prepare_targets(targets, masks_raw, batch_size)
            outputs = ip._forward_lightning_model(pl_model, version, inputs, masks)
            reg_out, _, _, _ = ip._split_outputs(outputs)

            for particle in PARTICLES:
                mean = getattr(pl_model, f"mass_reg_mean_{particle}").to(device)
                std = getattr(pl_model, f"mass_reg_std_{particle}").to(device)
                p4 = _v5_to_cartesian(reg_out[particle], mean, std, signed_energy=signed_energy)
                m_sq = _lorentz_mass_sq(p4)
                mask = masks[particle].bool()
                m_sq_by_particle[particle].append(m_sq[mask].detach().cpu().numpy())

            seen += batch_size

    result = {}
    for particle, chunks in m_sq_by_particle.items():
        m_sq = np.concatenate(chunks) if chunks else np.array([])
        n = int(m_sq.size)
        n_negative = int(np.sum(m_sq < 0))
        frac_physical = float(1.0 - n_negative / n) if n else float("nan")
        m_physical = np.sqrt(m_sq[m_sq >= 0]) if n else np.array([])
        neg_vals = m_sq[m_sq < 0]
        n_nonfinite = int(np.sum(~np.isfinite(m_sq)))
        neg_finite = neg_vals[np.isfinite(neg_vals)]
        result[particle] = {
            "n_slots": n,
            "n_negative_m_sq": n_negative,
            "n_nonfinite_m_sq": n_nonfinite,
            "frac_physical": frac_physical,
            "mass_mean_gev": float(np.mean(m_physical)) if m_physical.size else float("nan"),
            "mass_std_gev": float(np.std(m_physical)) if m_physical.size else float("nan"),
            "mass_median_gev": float(np.median(m_physical)) if m_physical.size else float("nan"),
            "mass_p05_gev": float(np.percentile(m_physical, 5)) if m_physical.size else float("nan"),
            "mass_p95_gev": float(np.percentile(m_physical, 95)) if m_physical.size else float("nan"),
            "neg_m_sq_median_gev2": float(np.median(neg_finite)) if neg_finite.size else 0.0,
            "neg_m_sq_p05_gev2": float(np.percentile(neg_finite, 5)) if neg_finite.size else 0.0,
        }
    result["signed_energy"] = signed_energy
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=["v5", "vx-full"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache-path", type=str, default=None)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--outdir", type=Path, default=Path("plots/mass-shell-sanity"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ns = _build_args(args.version, args.checkpoint, args.device, args.max_samples,
                      args.cache_path, args.split)
    cache_dir, cache_filename = ip._resolve_cache(ns)
    data = load_cache(cache_dir=cache_dir, filename=cache_filename)
    device = torch.device(args.device)
    pl_model = ip._load_lightning_model(ns, data)

    result = compute_mass_shell(pl_model, args.version, data[f"{args.split}_loader"],
                                 device, args.max_samples)
    result["version"] = args.version
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["max_samples"] = args.max_samples

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "mass_shell.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for particle in PARTICLES:
        r = result[particle]
        print(f"{particle}: frac_physical={r['frac_physical']:.4f} "
              f"mass_median={r['mass_median_gev']:.2f} GeV "
              f"(n={r['n_slots']}, n_negative={r['n_negative_m_sq']})")


if __name__ == "__main__":
    main()
