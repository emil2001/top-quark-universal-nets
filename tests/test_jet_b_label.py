"""Jet labels use signed b-tag flags rather than PDG IDs."""

import numpy as np
import torch

from pat_embedder.data import jet_class_labels
from pat_embedder.evaluation import _jet_label_func
from pat_embedder.lightning import PatEmbedderLightningV5

MAX_LEPTONS = 4
MAX_JETS = 12
MAX_TOPS = 4
MAX_WS = 4
NUM_HLR_VARS = 51

JET_CLASS_EMPTY, JET_CLASS_LIGHT, JET_CLASS_B = 0, 1, 2


def _labels_at_top_decay_jets(labels, top_to_bjet):
    """Labels of the gen jets the tops decayed into (those are b jets)."""
    n_slots = labels.shape[1]
    assigned = top_to_bjet < n_slots  # n_slots is the "no match" index
    rows = torch.nonzero(assigned, as_tuple=True)[0]
    return labels[rows, top_to_bjet[assigned]]


def test_jet_class_labels_decodes_the_b_flag():
    flags = torch.tensor([[0, -1, 1]], dtype=torch.int8)
    expected = [JET_CLASS_EMPTY, JET_CLASS_LIGHT, JET_CLASS_B]

    assert jet_class_labels(flags).tolist() == [expected]
    assert jet_class_labels(flags.numpy()).tolist() == [expected]


def test_top_decay_jets_carry_the_b_class(synthetic_batch):
    # Strict on the pinned fixture; across the whole cache the invariant holds on
    # 99.95% of the assigned slots (task 840), so a rebuilt fixture can miss it by
    # gen-matching noise rather than by a code change.
    cls = synthetic_batch["targets"]["classification"]["jets"]
    top_to_bjet = synthetic_batch["targets"]["assignments"]["top_to_bjet"].long()

    picked = _labels_at_top_decay_jets(jet_class_labels(cls), top_to_bjet)

    assert picked.numel() > 0
    assert (picked == JET_CLASS_B).all()


def test_evaluation_decodes_jets_like_training(synthetic_batch):
    cls = synthetic_batch["targets"]["classification"]["jets"]

    evaluated = _jet_label_func(cls.numpy())

    assert np.array_equal(evaluated, jet_class_labels(cls).numpy())
    assert (evaluated == JET_CLASS_B).any()


def test_training_targets_reach_the_b_class(synthetic_batch, reg_means, reg_stds):
    module = PatEmbedderLightningV5(
        embed_dim=32, n_heads=2, n_enc_layers=1, n_dec_layers=1,
        max_leptons=MAX_LEPTONS, max_jets=MAX_JETS, max_tops=MAX_TOPS, max_ws=MAX_WS,
        num_hlr_vars=NUM_HLR_VARS, reg_means=reg_means, reg_stds=reg_stds,
    ).eval()
    batch_size = synthetic_batch["features"].size(0)

    _, cls_tgt, _, _, _ = module._prepare_targets(
        synthetic_batch["targets"], synthetic_batch["masks"], batch_size
    )
    jet_classes = cls_tgt["jet_classes"].view(batch_size, MAX_JETS)

    top_to_bjet = synthetic_batch["targets"]["assignments"]["top_to_bjet"].long()
    picked = _labels_at_top_decay_jets(jet_classes.cpu(), top_to_bjet)
    assert (picked == JET_CLASS_B).all()
    assert set(jet_classes.unique().tolist()) == {
        JET_CLASS_EMPTY, JET_CLASS_LIGHT, JET_CLASS_B
    }
