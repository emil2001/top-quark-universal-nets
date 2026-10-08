"""All supported models train, log locally and reload for inference on CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pat_embedder import (
    PatEmbedderLightning,
    PatEmbedderLightningV5,
    PatEmbedderLightningVXFull,
    create_trainer,
)

from .test_detector_mask import _load_infer_module


@pytest.mark.parametrize("version", ["v1", "v5", "vx-full"])
def test_training_checkpoint_and_inference(version, fixture_data, tmp_path):
    torch.manual_seed(2025)
    data = fixture_data
    common = {"num_hlr_vars": 51, "dropout": 0.0}
    if version == "v1":
        module = PatEmbedderLightning(hidden_dim=32, num_layers=1, **common)
    else:
        kwargs = {
            **common, "embed_dim": 32, "n_heads": 2, "n_enc_layers": 1, "n_dec_layers": 1,
            "reg_means": data["reg_means"], "reg_stds": data["reg_stds"],
            "phi_means": {key: np.asarray(value)[:, 2] for key, value in data["reg_means"].items()},
            "phi_stds": {key: np.asarray(value)[:, 2] for key, value in data["reg_stds"].items()},
        }
        if version == "v5":
            module = PatEmbedderLightningV5(**kwargs)
        else:
            module = PatEmbedderLightningVXFull(
                feat_mean=data["feat_mean"], feat_std=data["feat_std"],
                ga_channels=4, ga_heads=1, **kwargs,
            )
    trainer = create_trainer(
        "synthetic", max_epochs=1, accelerator="cpu", precision="32-true",
        tracking_uri=str(tmp_path / "mlruns"), run_name=version,
        checkpoint_dir=tmp_path / "checkpoints",
        enable_progress_bar=False, enable_model_summary=False,
    )
    from torch.utils.data import DataLoader

    trainer.fit(
        module,
        train_dataloaders=DataLoader(data["train_batches"], batch_size=None),
        val_dataloaders=DataLoader(data["test_batches"], batch_size=None),
    )
    assert trainer.global_step == 4
    assert torch.isfinite(trainer.callback_metrics["loss/val"])
    assert trainer.checkpoint_callback.best_model_path
    checkpoint = trainer.checkpoint_callback.last_model_path
    inference = _load_infer_module()
    args = SimpleNamespace(device="cpu", weights_only=False, strict_load=True)
    restored = type(module).load_from_checkpoint(
        checkpoint, **inference._checkpoint_kwargs(version, data, args)
    )
    restored.eval()
    module.eval()
    batch = data["test_batches"][0]
    with torch.no_grad():
        original = inference._forward_lightning_model(module, version, batch["features"], batch["masks"])
        reloaded = inference._forward_lightning_model(restored, version, batch["features"], batch["masks"])
    for expected, observed in zip(original[:10], reloaded[:10]):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, expected)
    client = trainer.logger.experiment
    artifacts = client.list_artifacts(trainer.logger.run_id, "last-model")
    assert any(artifact.path.endswith("last.ckpt") for artifact in artifacts)
