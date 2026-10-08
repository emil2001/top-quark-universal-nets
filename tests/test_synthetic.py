"""Synthetic examples preserve padding, topology and train-only statistics."""

import numpy as np
import pytest
import torch

from pat_embedder.data import detector_enc_padding_mask, load_cache
from pat_embedder.synthetic import make_synthetic_data, save_synthetic_cache


def test_seed_is_deterministic_and_test_split_cannot_change_training_statistics():
    first = make_synthetic_data(seed=17, test_batches=1)
    second = make_synthetic_data(seed=17, test_batches=2)
    torch.testing.assert_close(first["train_batches"][0]["features"], second["train_batches"][0]["features"])
    np.testing.assert_array_equal(first["feat_mean"], second["feat_mean"])
    for key in first["reg_means"]:
        np.testing.assert_array_equal(first["reg_means"][key], second["reg_means"][key])
    changed = make_synthetic_data(seed=18)
    assert not torch.equal(first["train_batches"][0]["features"], changed["train_batches"][0]["features"])


def test_detector_padding_and_decay_assignment_follow_the_generated_topology(fixture_data):
    padding = detector_enc_padding_mask(fixture_data["features"], 4, 12)
    masks = fixture_data["masks"]
    assert torch.equal(padding[:, 1:5], ~masks["leptons"])
    assert not (padding[:, 5:] & masks["jets"]).any()
    assert ((~padding[:, 5:]) & (~masks["jets"])).any()
    assignments = fixture_data["targets"]["assignments"]
    child_masks = {
        "top_to_w": masks["ws"], "top_to_bjet": masks["jets"],
        "w_to_lepton": masks["leptons"], "w_to_neutrino": masks["neutrinos"],
        "w_to_jet1": masks["jets"], "w_to_jet2": masks["jets"],
    }
    for relation, target in assignments.items():
        child = child_masks[relation]
        matched = target < child.shape[1]
        rows, slots = matched.nonzero(as_tuple=True)
        assert child[rows, target[rows, slots].long()].all()
    for values in fixture_data["targets"]["regression"].values():
        assert torch.isfinite(values).all()


def test_synthetic_cache_round_trip_and_metadata(tmp_path):
    path = tmp_path / "synthetic.pkl"
    save_synthetic_cache(path, seed=23)
    loaded = load_cache(str(tmp_path), path.name)
    assert loaded["build_metadata"] == {"kind": "synthetic", "seed": 23}
    assert len(loaded["train_batches"]) == 4
    assert len(loaded["test_batches"]) == 1


@pytest.mark.parametrize("kwargs", [{"batch_size": 1}, {"train_batches": 0}, {"test_batches": 0}])
def test_invalid_split_sizes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_data(**kwargs)
