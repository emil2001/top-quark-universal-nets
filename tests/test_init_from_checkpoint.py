"""Fine-tuning starts from trained weights and from a fresh optimiser.

``trainer.fit(ckpt_path=...)`` would also restore the LR schedule, and a finished run
leaves it at the cosine floor (1% of the peak), so continuing a run that way trains at
3e-6. ``--init-from-checkpoint`` takes the weights only and lets the schedule start over.
"""

from types import SimpleNamespace

import pytest
import torch

from pat_embedder.lightning import PatEmbedderLightningV5, load_checkpoint_weights

MAX_LEPTONS = 4
MAX_JETS = 12
MAX_TOPS = 4
MAX_WS = 4
NUM_HLR_VARS = 51


def _module(reg_means, reg_stds, seed):
    torch.manual_seed(seed)
    return PatEmbedderLightningV5(
        embed_dim=32, n_heads=2, n_enc_layers=1, n_dec_layers=1,
        max_leptons=MAX_LEPTONS, max_jets=MAX_JETS, max_tops=MAX_TOPS, max_ws=MAX_WS,
        num_hlr_vars=NUM_HLR_VARS, reg_means=reg_means, reg_stds=reg_stds,
    )


def _checkpoint(module, tmp_path, name="run.ckpt", compiled=False):
    """A checkpoint in the Lightning layout, optionally with ``torch.compile`` key names."""
    state_dict = module.state_dict()
    if compiled:
        state_dict = {
            ("model._orig_mod." + key[len("model."):] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }
    path = tmp_path / name
    torch.save(
        {
            "state_dict": state_dict,
            "epoch": 9,
            "global_step": 31380,
            # Deliberately unusable: fine-tuning must not read these back.
            "optimizer_states": ["not an optimiser state"],
            "lr_schedulers": [{"_last_lr": [3e-06]}],
        },
        path,
    )
    return path


def test_weights_come_from_the_checkpoint(tmp_path, reg_means, reg_stds):
    trained = _module(reg_means, reg_stds, seed=1)
    fresh = _module(reg_means, reg_stds, seed=2)
    head = "model.jet_cls_head.3.weight"
    assert not torch.equal(trained.state_dict()[head], fresh.state_dict()[head])

    info = load_checkpoint_weights(fresh, _checkpoint(trained, tmp_path))

    for key, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key
    assert info == {"epoch": 9, "global_step": 31380}


def test_a_compiled_checkpoint_loads_into_an_eager_module(tmp_path, reg_means, reg_stds):
    trained = _module(reg_means, reg_stds, seed=1)
    fresh = _module(reg_means, reg_stds, seed=2)

    load_checkpoint_weights(fresh, _checkpoint(trained, tmp_path, compiled=True))

    for key, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key


def test_a_checkpoint_of_another_model_is_rejected(tmp_path, reg_means, reg_stds):
    trained = _module(reg_means, reg_stds, seed=1)
    path = _checkpoint(trained, tmp_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    del checkpoint["state_dict"]["model.jet_cls_head.3.weight"]
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="jet_cls_head"):
        load_checkpoint_weights(_module(reg_means, reg_stds, seed=2), path)


def test_the_training_cli_exposes_the_flag(monkeypatch):
    import train

    monkeypatch.setattr("sys.argv", ["train.py"])
    assert train._parse_args().init_from_checkpoint is None

    monkeypatch.setattr("sys.argv", ["train.py", "--init-from-checkpoint", "/tmp/best.ckpt"])
    assert train._parse_args().init_from_checkpoint == "/tmp/best.ckpt"


def _lr_curve(module, total_steps):
    """The LR the optimiser sees at every step of a run of `total_steps` steps."""
    module._trainer = SimpleNamespace(estimated_stepping_batches=total_steps)
    config = module.configure_optimizers()
    optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
    curve = []
    for _ in range(total_steps):
        curve.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # no gradients, so this only keeps the scheduler's step order happy
        scheduler.step()
    return curve


def test_the_lr_schedule_spans_the_new_run(tmp_path, reg_means, reg_stds):
    """The cosine of the fine-tune covers the new `--max-epochs`, not the tail of the old run.

    The checkpoint was written at the floor of its own schedule (`_last_lr` 3e-6); a schedule
    restored from it would fine-tune there. A fresh one peaks at 10% of whatever the new run is.
    """
    trained = _module(reg_means, reg_stds, seed=1)
    path = _checkpoint(trained, tmp_path)
    checkpoint_floor = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint_floor["lr_schedulers"][0]["_last_lr"] == [3e-06]

    short, long = (_module(reg_means, reg_stds, seed=2) for _ in range(2))
    for module in (short, long):
        load_checkpoint_weights(module, path)
        module.lr = 1e-3
    short_curve, long_curve = _lr_curve(short, 100), _lr_curve(long, 200)

    assert short_curve[10] == pytest.approx(1e-3)
    assert long_curve[20] == pytest.approx(1e-3)
    assert short_curve[-1] == pytest.approx(1e-5, rel=1e-3)
    assert long_curve[-1] == pytest.approx(1e-5, rel=1e-3)
    # Same shape stretched over the longer run, and nowhere near the 3e-6 the checkpoint left.
    assert short_curve[50] == pytest.approx(long_curve[100], rel=1e-6)
    assert min(short_curve[1:]) > 3e-06


def test_the_optimiser_moments_do_not_come_from_the_checkpoint(tmp_path, reg_means, reg_stds):
    """AdamW starts empty: the checkpoint's optimiser state is never read."""
    fresh = _module(reg_means, reg_stds, seed=2)
    load_checkpoint_weights(fresh, _checkpoint(_module(reg_means, reg_stds, seed=1), tmp_path))

    fresh._trainer = SimpleNamespace(estimated_stepping_batches=100)
    optimizer = fresh.configure_optimizers()["optimizer"]

    assert optimizer.state_dict()["state"] == {}
    assert all(optimizer.state[p] == {} for group in optimizer.param_groups for p in group["params"])


def test_the_training_cli_fine_tunes_from_the_checkpoint(
    tmp_path, monkeypatch, mlflow_logger, reg_means, reg_stds
):
    """`--init-from-checkpoint` reaches the model that is handed to the trainer, and the run
    records where those weights came from."""
    import train

    trained = _module(reg_means, reg_stds, seed=1)
    path = _checkpoint(trained, tmp_path)
    fitted = {}

    def _fake_create_trainer(*args, **kwargs):
        return SimpleNamespace(logger=mlflow_logger, fit=lambda model, **kw: fitted.update(model=model))

    monkeypatch.setattr(train, "load_cache", lambda **kwargs: {
        "train_loader": None, "test_loader": None,
        "reg_means": reg_means, "reg_stds": reg_stds,
    })
    monkeypatch.setattr(train, "create_trainer", _fake_create_trainer)
    monkeypatch.setattr("sys.argv", [
        "train.py", "--version", "v5", "--v5-embed-dim", "32", "--v5-n-heads", "2",
        "--v5-n-enc-layers", "1", "--v5-n-dec-layers", "1", "--max-epochs", "1",
        "--no-compile-model", "--init-from-checkpoint", str(path),
    ])

    train.main()

    for key, value in trained.state_dict().items():
        assert torch.equal(fitted["model"].state_dict()[key], value), key
    assert str(path) in mlflow_logger.experiment.tags["cli"]
    assert mlflow_logger.experiment.params["init_from_checkpoint"] == str(path)
    assert mlflow_logger.experiment.params["init_from_checkpoint_global_step"] == 31380
