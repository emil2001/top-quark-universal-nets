"""Check the bundled research sample and its independently rebuilt statistics."""

from pathlib import Path

import numpy as np
import torch

from pat_embedder.data import load_cache
from pat_embedder.example import make_example_data, save_example_cache

SAMPLE = Path(__file__).resolve().parents[1] / "examples/mini_events.npz"


def test_sample_format_and_train_only_normalization():
    data = make_example_data(SAMPLE)
    assert [batch["features"].shape[0] for batch in data["train_batches"]] == [3, 3]
    assert data["test_batches"][0]["features"].shape == (2, 130)
    with np.load(SAMPLE, allow_pickle=False) as archive:
        # MET has eight observations; its training mean must exclude test events.
        np.testing.assert_allclose(data["feat_mean"][:2], archive["features"][:6, :2].mean(0))
        assert not np.allclose(data["feat_mean"][:2], archive["features"][:, :2].mean(0))
        # The four-lepton slot is unobserved in this small sample.
        assert np.isnan(archive["regression_leptons"][:, 3]).all()
    np.testing.assert_array_equal(data["reg_means"]["leptons"][3], 0)
    np.testing.assert_array_equal(data["reg_stds"]["leptons"][3], 1)
    for batches in (data["train_batches"], data["test_batches"]):
        for batch in batches:
            assert torch.isfinite(batch["features"]).all()
            for key, values in batch["targets"]["regression"].items():
                assert torch.isfinite(values).all()
                assert torch.equal(batch["masks"][key], batch["targets"]["classification"][key] != 0)


def test_example_cache_roundtrip(tmp_path):
    output = tmp_path / "example.pkl"
    save_example_cache(SAMPLE, output)
    data = load_cache(str(tmp_path), output.name)
    assert data["build_metadata"]["kind"] == "miniature_research_sample"
    assert len(data["train_loader"]) == 2
    assert len(data["test_loader"]) == 1
    assert torch.isfinite(next(iter(data["train_loader"]))["features"]).all()
