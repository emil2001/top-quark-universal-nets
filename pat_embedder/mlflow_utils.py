import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from .lightning import PatEmbedderLightning


def _iter_checkpoint_artifact_paths(client, run_id: str) -> list[str]:
    """All run artifact paths ending in ``.ckpt`` (recursive)."""
    found: list[str] = []

    def walk(prefix: str) -> None:
        for ent in client.list_artifacts(run_id, path=prefix if prefix else None):
            if ent.is_dir:
                walk(ent.path)
            elif ent.path.endswith(".ckpt"):
                found.append(ent.path)

    walk("")
    return found


def _step_from_ckpt_path(p: str) -> int:
    m = re.search(r"step[=_](\d+)", p)
    return int(m.group(1)) if m else -1


def _resolve_mlflow_checkpoint_path(
    client,
    run_id: str,
    *,
    preference: str = "best",
) -> str:
    """Pick one ``.ckpt`` artifact: best val loss (default), last step, or last file.

    Parameters
    ----------
    preference
        ``"best"`` — prefer Lightning best checkpoint (``best-...`` in path).
        ``"last"`` — prefer ``last`` checkpoint, else highest ``step=``.
        ``"last_step"`` — ignore ``best-``; take highest ``step=`` (old behaviour).
    """
    ckpts = _iter_checkpoint_artifact_paths(client, run_id)
    return pick_mlflow_checkpoint_path(ckpts, preference=preference)


def pick_mlflow_checkpoint_path(ckpts: list[str], *, preference: str = "best") -> str:
    """Pure selection logic for tests and :func:`_resolve_mlflow_checkpoint_path`."""
    if not ckpts:
        raise ValueError("No checkpoint paths given")

    pref = preference.lower().strip()
    if pref == "last_step":
        scored = [p for p in ckpts if "epoch=" in p or "step=" in p]
        if scored:
            return sorted(scored, key=_step_from_ckpt_path)[-1]
        return ckpts[-1]

    if pref == "last":
        canonical_last = "last-model/last.ckpt"
        if canonical_last in ckpts:
            return canonical_last
        last_paths = [
            p for p in ckpts
            if Path(p).stem == "last" or Path(p).name.startswith("last-")
        ]
        if last_paths:
            return sorted(last_paths)[-1]
        scored = [p for p in ckpts if "epoch=" in p or "step=" in p]
        if scored:
            return sorted(scored, key=_step_from_ckpt_path)[-1]
        return ckpts[-1]

    # default: best val checkpoint when present
    canonical_best = "best-model/best.ckpt"
    if canonical_best in ckpts:
        return canonical_best
    bm_paths = [p for p in ckpts if p.startswith("best-model/")]
    if bm_paths:
        fixed = [p for p in bm_paths if Path(p).name == "best.ckpt"]
        if fixed:
            return fixed[0]
        return sorted(bm_paths, key=_step_from_ckpt_path)[-1]
    best_paths = [p for p in ckpts if "best-" in p.split("/")[-1]]
    if best_paths:
        return sorted(best_paths, key=_step_from_ckpt_path)[-1]
    scored = [p for p in ckpts if "epoch=" in p or "step=" in p]
    if scored:
        return sorted(scored, key=_step_from_ckpt_path)[-1]
    return ckpts[-1]


def load_from_mlflow(
    run_name,
    experiment_name="event-reconstruction",
    tracking_uri="http://localhost:5050",
    lightning_cls=None,
    checkpoint_preference: str = "best",
    weights_only: bool = False,
):
    """Load a Lightning checkpoint from MLflow by run name.

    Returns the Lightning module with weights restored.
    If lightning_cls is None, uses PatEmbedderLightning.

    checkpoint_preference
        ``"best"`` (default) prefers ``best-model/best.ckpt`` when present (single
        file overwritten each upload), else Lightning-style ``best-epoch=...`` paths.
        ``"last"`` prefers ``last-model/last.ckpt``, then a root ``last.ckpt``.
        ``"last_step"`` picks the highest training step (legacy behaviour).
    weights_only
        Forwarded to ``torch.load`` via ``LightningModule.load_from_checkpoint``.
        Defaults to ``False`` because our checkpoints come from our own MLflow
        server (trusted source) and contain numpy arrays (e.g. standardisation
        stats in ``hparams``) that the PyTorch 2.6+ weights-only unpickler
        rejects by default.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_name}'")

    run = runs[0]
    print(f"Found run: {run.info.run_id} ({run.info.status})")

    ckpt_path = _resolve_mlflow_checkpoint_path(
        client, run.info.run_id, preference=checkpoint_preference,
    )
    local_path = client.download_artifacts(run.info.run_id, ckpt_path)
    print(f"Downloaded: {local_path}")

    cls = lightning_cls or PatEmbedderLightning
    pl_model = cls.load_from_checkpoint(local_path, weights_only=weights_only)
    print(f"Loaded {cls.__name__} from '{run_name}' (artifact={ckpt_path}, preference={checkpoint_preference})")
    return pl_model


def load_from_mlflow_by_run_id(
    run_id,
    tracking_uri="http://localhost:5050",
    lightning_cls=None,
    checkpoint_preference: str = "best",
    weights_only: bool = False,
):
    """Load a Lightning checkpoint from MLflow by run UUID (not run name tag).

    Use this when you have a specific run id (e.g. from the MLflow UI) and the
    experiment name is unknown or you want to avoid name collisions.

    See :func:`load_from_mlflow` for ``checkpoint_preference`` and
    ``weights_only`` semantics.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    print(f"Found run: {run_id} ({run.info.status})")

    ckpt_path = _resolve_mlflow_checkpoint_path(
        client, run_id, preference=checkpoint_preference,
    )
    local_path = client.download_artifacts(run_id, ckpt_path)
    print(f"Downloaded: {local_path}")

    cls = lightning_cls or PatEmbedderLightning
    pl_model = cls.load_from_checkpoint(local_path, weights_only=weights_only)
    print(f"Loaded {cls.__name__} from run_id={run_id} (artifact={ckpt_path}, preference={checkpoint_preference})")
    return pl_model


def plot_loss_curves(run_id=None, trainer=None, tracking_uri="http://localhost:5050"):
    """Fetch and plot loss curves from MLflow for a given run.

    Either pass run_id directly or a trainer with an attached logger.
    """
    if run_id is None and trainer is not None:
        run_id = trainer.logger.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    print(f"Fetching metrics from run: {run_id}")

    metric_keys = [
        "loss/train_epoch", "loss/val",
        "loss/train_regression_epoch", "loss/train_classification_epoch",
        "loss/train_assignment_epoch", "loss/train_hlr_epoch",
        "loss/val_regression", "loss/val_classification",
        "loss/val_assignment", "loss/val_hlr",
        "task_weight/regression", "task_weight/classification",
        "task_weight/assignment", "task_weight/hlr",
    ]

    metrics_data = {}
    for key in metric_keys:
        try:
            history = client.get_metric_history(run_id, key)
            if history:
                metrics_data[key] = [(m.step, m.value) for m in history]
        except Exception:
            pass

    available = [k for k in metric_keys if k in metrics_data]
    print(f"Available metrics: {available}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for key, label, ls in [("loss/train_epoch", "Train", "-"), ("loss/val", "Val", "--")]:
        if key in metrics_data:
            steps, vals = zip(*metrics_data[key])
            ax.plot(steps, vals, ls, label=label)
    ax.set_title("Total Loss: Train vs Val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for task in ["regression", "classification", "assignment", "hlr"]:
        key = f"loss/train_{task}_epoch"
        if key in metrics_data:
            steps, vals = zip(*metrics_data[key])
            ax.plot(steps, vals, label=task)
    ax.set_title("Per-Task Train Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1, 0]
    for task in ["regression", "classification", "assignment", "hlr"]:
        key = f"loss/val_{task}"
        if key in metrics_data:
            steps, vals = zip(*metrics_data[key])
            ax.plot(steps, vals, label=task)
    ax.set_title("Per-Task Val Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1, 1]
    for task in ["regression", "classification", "assignment", "hlr"]:
        key = f"task_weight/{task}"
        if key in metrics_data:
            steps, vals = zip(*metrics_data[key])
            ax.plot(steps, vals, label=task)
    ax.set_title("Learned Task Weights (precision = exp(-log_var))")
    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.show()

    for key in ["loss/train_epoch", "loss/val"]:
        if key in metrics_data:
            steps, vals = zip(*metrics_data[key])
            if len(vals) >= 3:
                last_3 = vals[-3:]
                trend = "DECREASING" if last_3[-1] < last_3[0] else "FLAT/INCREASING"
                print(f"{key}: last 3 values = {[f'{v:.4f}' for v in last_3]} -> {trend}")

    return metrics_data
