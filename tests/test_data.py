"""Data utilities: save/load cache round-trip."""

import os
import pickle

import numpy as np
import pytest
import torch

from pat_embedder.data import (
    CachedBatchDataset,
    LegacyCacheSidecarError,
    load_cache,
    rebuild_cache,
    save_cache,
)


class TestCachedBatchDataset:
    def test_len(self, synthetic_batch):
        ds = CachedBatchDataset([synthetic_batch, synthetic_batch])
        assert len(ds) == 2

    def test_getitem(self, synthetic_batch):
        ds = CachedBatchDataset([synthetic_batch])
        batch = ds[0]
        assert "features" in batch
        assert "targets" in batch


class TestSaveLoadRoundTrip:
    def test_round_trip(self, synthetic_batch, fixture_data, tmp_path):
        train_batches = [synthetic_batch]
        test_batches = [synthetic_batch]

        save_cache(
            train_batches, test_batches,
            feat_mean=fixture_data["feat_mean"],
            feat_std=fixture_data["feat_std"],
            reg_means=fixture_data["reg_means"],
            reg_stds=fixture_data["reg_stds"],
            hlr_mean=fixture_data["hlr_mean"],
            hlr_std=fixture_data["hlr_std"],
            cache_dir=str(tmp_path),
            filename="test_cache.pkl",
        )

        assert os.path.exists(tmp_path / "test_cache.pkl")

        loaded = load_cache(cache_dir=str(tmp_path), filename="test_cache.pkl")
        assert "train_loader" in loaded
        assert "test_loader" in loaded

        reloaded_batch = next(iter(loaded["train_loader"]))
        assert torch.allclose(reloaded_batch["features"], synthetic_batch["features"])

    def test_stats_preserved(self, fixture_data, synthetic_batch, tmp_path):
        save_cache(
            [synthetic_batch], [synthetic_batch],
            feat_mean=fixture_data["feat_mean"],
            feat_std=fixture_data["feat_std"],
            reg_means=fixture_data["reg_means"],
            reg_stds=fixture_data["reg_stds"],
            hlr_mean=fixture_data["hlr_mean"],
            hlr_std=fixture_data["hlr_std"],
            cache_dir=str(tmp_path),
            filename="test_cache2.pkl",
        )
        loaded = load_cache(cache_dir=str(tmp_path), filename="test_cache2.pkl")
        np.testing.assert_array_equal(loaded["feat_mean"], fixture_data["feat_mean"])
        np.testing.assert_array_equal(loaded["hlr_mean"], fixture_data["hlr_mean"])


class TestCacheContract:
    def test_rebuild_replaces_normalized_targets_and_stats_together(self, tmp_path):
        def batch(values):
            values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
            size = len(values)
            return {
                "features": torch.zeros(size, 1),
                "masks": {"ws": torch.ones(size, 1, dtype=torch.bool)},
                "targets": {
                    "regression": {"ws": values},
                    "classification": {"ws": torch.zeros(size, 1, dtype=torch.int8)},
                },
            }

        source = save_cache(
            [batch([0.0, 2e-8, 1e3])], [batch([4e-8])],
            feat_mean=np.zeros(1, dtype=np.float32), feat_std=np.ones(1, dtype=np.float32),
            reg_means={"ws": np.zeros((1, 1), dtype=np.float32)},
            reg_stds={"ws": np.full((1, 1), 1e8, dtype=np.float32)},
            hlr_mean=None, hlr_std=None,
            cache_dir=str(tmp_path), filename="legacy.pkl",
        )

        rebuilt, metadata = rebuild_cache(source, str(tmp_path / "rebuilt.pkl"))
        loaded = load_cache(cache_dir=str(tmp_path), filename="rebuilt.pkl")
        rebuilt_targets = torch.cat(
            [batch["targets"]["regression"]["ws"] for batch in loaded["train_batches"]]
            + [batch["targets"]["regression"]["ws"] for batch in loaded["test_batches"]]
        )
        raw_targets = rebuilt_targets * torch.as_tensor(loaded["reg_stds"]["ws"]) + torch.as_tensor(
            loaded["reg_means"]["ws"]
        )

        assert rebuilt == str(tmp_path / "rebuilt.pkl")
        assert metadata["dropped_corrupted_samples"] == 1
        torch.testing.assert_close(raw_targets.flatten().sort().values, torch.tensor([0.0, 2.0, 4.0]))

    def test_saved_cache_records_contract_version(self, fixture_data, synthetic_batch, tmp_path):
        path = save_cache(
            [synthetic_batch], [synthetic_batch],
            feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
            reg_means=fixture_data["reg_means"], reg_stds=fixture_data["reg_stds"],
            hlr_mean=fixture_data["hlr_mean"], hlr_std=fixture_data["hlr_std"],
            cache_dir=str(tmp_path), filename="v2.pkl",
        )

        with open(path, "rb") as cache_file:
            assert pickle.load(cache_file)["contract_version"] == 2

    def test_sidecar_only_cache_is_rejected(self, fixture_data, synthetic_batch, tmp_path):
        save_cache(
            [synthetic_batch], [synthetic_batch],
            feat_mean=fixture_data["feat_mean"], feat_std=fixture_data["feat_std"],
            reg_means=fixture_data["reg_means"], reg_stds=fixture_data["reg_stds"],
            hlr_mean=fixture_data["hlr_mean"], hlr_std=fixture_data["hlr_std"],
            cache_dir=str(tmp_path), filename="legacy.pkl",
        )
        (tmp_path / "legacy.pkl.reg_stats_fix.json").write_text("[]")

        with pytest.raises(LegacyCacheSidecarError, match="rebuild"):
            load_cache(cache_dir=str(tmp_path), filename="legacy.pkl")
