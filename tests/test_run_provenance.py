"""What an MLflow run says about the command that produced it.

MLflow records `mlflow.source.git.commit` from the working tree at the moment the run starts,
so a run started before its own code was committed is tagged with the previous commit, and
Lightning's `save_hyperparameters` only stores module arguments. Without the command line and
the checkpoint the weights came from, a fine-tune is indistinguishable from a run from scratch.
"""

import shlex

from pat_embedder.lightning import log_run_provenance


def test_the_command_line_is_recorded_on_the_run(mlflow_logger):
    argv = ["train.py", "--version", "v5", "--max-epochs", "1", "--precision", "32-true"]

    log_run_provenance(mlflow_logger, argv)

    assert mlflow_logger.experiment.tags["cli"] == (
        "train.py --version v5 --max-epochs 1 --precision 32-true"
    )
    assert mlflow_logger.experiment.run_ids == {mlflow_logger.run_id}


def test_a_quoted_argument_survives_the_round_trip(mlflow_logger):
    argv = ["train.py", "--run-name", "probe844 v5 s2025"]

    log_run_provenance(mlflow_logger, argv)

    assert shlex.split(mlflow_logger.experiment.tags["cli"]) == argv


def test_a_long_command_line_is_recorded_whole(mlflow_logger):
    """`MLFlowLogger.log_hyperparams` truncates values at 250 characters; a training
    command line is longer than that, so the provenance must not go through it."""
    argv = ["train.py"] + [f"--flag-{i}=value-{i}" for i in range(60)]

    log_run_provenance(mlflow_logger, argv)

    assert shlex.split(mlflow_logger.experiment.tags["cli"]) == argv
    assert len(mlflow_logger.experiment.tags["cli"]) > 250


def test_the_checkpoint_the_weights_came_from_is_recorded(mlflow_logger):
    log_run_provenance(
        mlflow_logger,
        ["train.py", "--init-from-checkpoint", "checkpoints/pretrain/best.ckpt"],
        checkpoint_path="checkpoints/pretrain/best.ckpt",
        checkpoint_state={"epoch": 9, "global_step": 31380},
    )

    assert mlflow_logger.experiment.params == {
        "init_from_checkpoint": "checkpoints/pretrain/best.ckpt",
        "init_from_checkpoint_epoch": 9,
        "init_from_checkpoint_global_step": 31380,
    }


def test_a_run_from_scratch_records_no_checkpoint(mlflow_logger):
    log_run_provenance(mlflow_logger, ["train.py", "--max-epochs", "10"])

    assert mlflow_logger.experiment.params == {}
    assert "cli" in mlflow_logger.experiment.tags
