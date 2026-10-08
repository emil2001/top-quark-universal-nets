"""Shared fixtures generated from deterministic synthetic events."""

import pytest
import torch

from pat_embedder.synthetic import make_synthetic_data

MAX_LEPTONS = 4
MAX_JETS = 12
MAX_TOPS = 4
MAX_WS = 4
NUM_HLR_VARS = 51
BATCH_SIZE = 8
INPUT_DIM = 130
REG_DIM_V1 = 7
REG_DIM_V5 = 5


@pytest.fixture(scope="session")
def fixture_data():
    data = make_synthetic_data()
    return {**data, **data["train_batches"][0]}


@pytest.fixture(scope="session")
def synthetic_batch(fixture_data):
    """A single batch dict exactly as the training loop sees it."""
    return {
        "features": fixture_data["features"],
        "masks": fixture_data["masks"],
        "targets": fixture_data["targets"],
    }


@pytest.fixture(scope="session")
def reg_means(fixture_data):
    return fixture_data["reg_means"]


@pytest.fixture(scope="session")
def reg_stds(fixture_data):
    return fixture_data["reg_stds"]


@pytest.fixture(scope="session")
def cls_targets_long(synthetic_batch):
    """Classification targets remapped to class indices (as Lightning does)."""
    lepton_map = {0: 0, -11: 1, 11: 2, -13: 3, 13: 4, -15: 5, 15: 6}
    neutrino_map = {0: 0, 12: 1, -12: 2, 14: 3, -14: 4, 16: 5, -16: 6}
    top_map = {0: 0, 6: 1, -6: 2}
    w_map = {0: 0, 24: 1, -24: 2}
    # Jets carry a b-tag sign, not a PDG id; spelled out here so the fixture
    # stays an oracle independent of `pat_embedder.data.jet_class_labels`.
    jet_map = {0: 0, -1: 1, 1: 2}

    def _remap(t, m):
        return torch.tensor([m.get(int(v), 0) for v in t.flatten()], dtype=torch.long)

    cls = synthetic_batch["targets"]["classification"]
    return {
        "lepton_classes": _remap(cls["leptons"], lepton_map),
        "neutrino_classes": _remap(cls["neutrinos"], neutrino_map),
        "jet_classes": _remap(cls["jets"], jet_map),
        "top_classes": _remap(cls["tops"], top_map),
        "w_classes": _remap(cls["ws"], w_map),
    }


class _RecordingMlflowClient:
    """The two `MlflowClient` calls the run provenance uses."""

    def __init__(self):
        self.tags = {}
        self.params = {}
        self.run_ids = set()

    def set_tag(self, run_id, key, value):
        self.run_ids.add(run_id)
        self.tags[key] = value

    def log_param(self, run_id, key, value):
        self.run_ids.add(run_id)
        self.params[key] = value


class _FakeMlflowLogger:
    """The surface `MLFlowLogger` exposes: the tracking client and the id of the open run."""

    def __init__(self):
        self.experiment = _RecordingMlflowClient()
        self.run_id = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def mlflow_logger():
    return _FakeMlflowLogger()
