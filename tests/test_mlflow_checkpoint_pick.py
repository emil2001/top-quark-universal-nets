"""Checkpoint path resolution for MLflow artifact layouts."""

import pytest

from pat_embedder.mlflow_utils import pick_mlflow_checkpoint_path


def test_pick_prefers_best_over_higher_step():
    ckpts = [
        "best-epoch=01-step=3138/best-epoch=01-step=3138.ckpt",
        "best-epoch=49-step=156900/best-epoch=49-step=156900.ckpt",
        "epoch=49-step=156900/epoch=49-step=156900.ckpt",
    ]
    out = pick_mlflow_checkpoint_path(ckpts, preference="best")
    assert "best-epoch=49" in out
    assert "156900" in out

def test_pick_last_step_legacy():
    ckpts = [
        "best-epoch=01-step=3138/x.ckpt",
        "epoch=49-step=156900/epoch=49-step=156900.ckpt",
    ]
    out = pick_mlflow_checkpoint_path(ckpts, preference="last_step")
    assert "156900" in out

def test_pick_last_prefers_last_stem():
    ckpts = [
        "best-epoch=01-step=3138/x.ckpt",
        "last.ckpt",
    ]
    assert pick_mlflow_checkpoint_path(ckpts, preference="last") == "last.ckpt"

def test_pick_best_model_dir_preferred():
    """Canonical ``best-model/best.ckpt`` wins over legacy per-epoch names."""
    ckpts = [
        "best-model/best.ckpt",
        "best-model/best-epoch=03-step=9000.ckpt",
        "best-epoch=05-step=15000/best-epoch=05-step=15000.ckpt",
    ]
    assert pick_mlflow_checkpoint_path(ckpts, preference="best") == "best-model/best.ckpt"


def test_pick_best_model_legacy_still_works():
    """Runs logged before ``best.ckpt`` rename: pick newest under ``best-model/``."""
    ckpts = [
        "best-model/best-epoch=01-step=6276.ckpt",
        "best-model/best-epoch=02-step=9414.ckpt",
    ]
    out = pick_mlflow_checkpoint_path(ckpts, preference="best")
    assert "9414" in out


def test_pick_last_model_canonical():
    assert (
        pick_mlflow_checkpoint_path(
            ["best-model/best.ckpt", "last-model/last.ckpt"],
            preference="last",
        )
        == "last-model/last.ckpt"
    )


def test_pick_best_model_dir_not_used_for_last():
    """``best-model/*`` must not satisfy ``last`` when a root ``last.ckpt`` exists."""
    ckpts = [
        "best-model/best-epoch=03-step=9000.ckpt",
        "last.ckpt",
    ]
    assert pick_mlflow_checkpoint_path(ckpts, preference="last") == "last.ckpt"


def test_pick_empty_raises():
    with pytest.raises(ValueError, match="No checkpoint"):
        pick_mlflow_checkpoint_path([], preference="best")
