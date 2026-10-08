"""Smoke tests: every public symbol imports without error."""


def test_import_package():
    import pat_embedder  # noqa: F401


def test_import_models():
    pass


def test_import_losses():
    pass


def test_import_lightning():
    pass


def test_import_data():
    from pat_embedder.data import (  # noqa: F401
        CachedBatchDataset,
        collate_fn,
        load_cache,
        save_cache,
    )


def test_import_mlflow_and_eval_helpers():
    from pat_embedder.evaluation import quick_eval_v5_style_metrics  # noqa: F401
    from pat_embedder.mlflow_utils import load_from_mlflow_by_run_id  # noqa: F401
