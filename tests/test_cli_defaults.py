"""Training and inference share the default local MLflow experiment."""

import importlib.util
import inspect
from pathlib import Path

from pat_embedder import create_trainer

from .test_detector_mask import _load_infer_module


def test_training_and_inference_share_mlflow_defaults(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "public_train_cli", Path(__file__).resolve().parents[1] / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["train.py"])
    train_args = module._parse_args()
    monkeypatch.setattr("sys.argv", ["infer_and_plot.py", "--version", "v1", "--run-name", "demo"])
    inference = _load_infer_module()
    infer_args = inference._parse_args()
    default_experiment = inspect.signature(create_trainer).parameters["experiment_name"].default
    assert train_args.experiment_name == infer_args.experiment_name == default_experiment
    assert train_args.tracking_uri == infer_args.tracking_uri
