"""Set matching must compare targets in the receiving query's normalization."""

import pytest
import torch

from pat_embedder.lightning import PatEmbedderLightningV5
from pat_embedder.losses import (
    PatEmbedderLossV5,
    _apply_perm_with_query_normalization,
    _target_in_query_normalization,
)


def test_pairwise_targets_are_renormalized_into_each_query_slot():
    # Two target slots describe the same physical log-pT=10 with different z-score stats.
    target = torch.zeros(1, 2, 5)
    target[0, 0, 0] = 10.0
    target[0, 1, 0] = 0.0
    mean = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    std = torch.ones(2, 3)

    pairwise = _target_in_query_normalization(target, mean, std)
    assert pairwise[0, 0, 1, 0] == 10.0
    assert pairwise[0, 1, 0, 0] == 0.0


@pytest.mark.parametrize("lightning_cls", [PatEmbedderLightningV5])
def test_v5_constructors_require_matching_statistics(lightning_cls):
    with pytest.raises(ValueError, match="reg_means and reg_stds are required"):
        lightning_cls()


def test_selected_permutation_keeps_target_physics_in_query_coordinates():
    target = torch.zeros(1, 2, 5)
    target[0, 0, [0, 1, 4]] = torch.tensor([3.0, -1.0, 5.0])
    target[0, 1, [0, 1, 4]] = torch.tensor([7.0, 2.0, 11.0])
    mean = torch.tensor([[0.0, 0.0, 0.0], [10.0, 5.0, 20.0]])
    std = torch.tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 10.0]])

    aligned = _apply_perm_with_query_normalization(target, torch.tensor([[1, 0]]), mean, std)
    # target slot 1 raw values are (24, 15, 130), represented in query slot 0 stats.
    assert torch.allclose(aligned[0, 0, [0, 1, 4]], torch.tensor([24.0, 7.5, 32.5]))
    # target slot 0 raw values are (3, -2, 20), represented in query slot 1 stats.
    assert torch.allclose(aligned[0, 1, [0, 1, 4]], torch.tensor([-3.5, -1.4, 0.0]))


@pytest.mark.parametrize("loss_cls, reg_dim", [(PatEmbedderLossV5, 5)])
def test_w_pairwise_matching_uses_query_normalization(loss_cls, reg_dim):
    criterion = loss_cls(max_tops=2, max_ws=2, aux_loss_weight=1.0)
    mean = torch.tensor([[0.0, 0.0, 0.0], [10.0, 5.0, 20.0]])
    std = torch.tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 10.0]])
    target = torch.zeros(1, 2, reg_dim)
    target[0, 0, [0, 1, reg_dim - 1]] = torch.tensor([3.0, -1.0, 5.0])
    target[0, 1, [0, 1, reg_dim - 1]] = torch.tensor([7.0, 2.0, 11.0])
    # Query 0 predicts target 1 and query 1 predicts target 0, each in the
    # receiving query's normalization. This is zero regression cost only with
    # the pairwise conversion used by the main and auxiliary matching paths.
    prediction = _apply_perm_with_query_normalization(target, torch.tensor([[1, 0]]), mean, std)
    cls_logits = torch.full((1, 2, 3), -20.0)
    cls_logits[..., 0] = 20.0
    assign_out = {
        "w_to_lepton": torch.full((1, 2, 5), -20.0),
        "w_to_neutrino": torch.full((1, 2, 5), -20.0),
        "w_to_jet1": torch.full((1, 2, 13), -20.0),
        "w_to_jet2": torch.full((1, 2, 13), -20.0),
    }
    for value in assign_out.values():
        value[..., 0] = 20.0
    assign_tgt = {key: torch.zeros(1, 2, dtype=torch.long) for key in assign_out}
    costs = criterion._w_pairwise_cost(
        cls_logits, prediction, assign_out, torch.zeros(1, 2, dtype=torch.long), target,
        assign_tgt, torch.ones(1, 2, dtype=torch.bool), mean, std,
    )
    assert costs[0, 0, 1] < 1e-6
    assert costs[0, 1, 0] < 1e-6
    assert costs[0, 0, 0] > 1.0


@pytest.mark.parametrize("loss_cls, reg_dim", [(PatEmbedderLossV5, 5)])
def test_main_and_aux_loss_align_swapped_top_and_w_regression(loss_cls, reg_dim):
    criterion = loss_cls(
        max_tops=2,
        max_ws=2,
        task_weights=[1.0, 0.0, 0.0, 0.0],
        aux_loss_weight=1.0,
        presence_loss_weight=0.0,
    )
    mean = torch.tensor([[0.0, 0.0, 0.0], [10.0, 5.0, 20.0]])
    std = torch.tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 10.0]])
    perm = torch.tensor([[1, 0]])
    target = torch.zeros(1, 2, reg_dim)
    target[0, 0, [0, 1, reg_dim - 1]] = torch.tensor([3.0, -1.0, 5.0])
    target[0, 1, [0, 1, reg_dim - 1]] = torch.tensor([7.0, 2.0, 11.0])
    pred = _apply_perm_with_query_normalization(target, perm, mean, std).requires_grad_()
    reg_out = {key: torch.zeros(1, 2, reg_dim) for key in ("leptons", "neutrinos", "jets")}
    reg_out.update({"tops": pred, "ws": pred})
    reg_tgt = {key: torch.zeros(1, 2, reg_dim) for key in reg_out}
    reg_tgt.update({"tops": target, "ws": target})
    masks = {key: torch.zeros(1, 2, dtype=torch.bool) for key in reg_out}
    masks.update({"tops": torch.ones(1, 2, dtype=torch.bool), "ws": torch.ones(1, 2, dtype=torch.bool)})
    cls_out = {key: torch.zeros(1, 2, n) for key, n in [("lepton_logits", 7), ("neutrino_logits", 7), ("jet_logits", 3), ("top_logits", 3), ("w_logits", 3)]}
    cls_tgt = {key: torch.zeros(2, dtype=torch.long) for key in ("lepton_classes", "neutrino_classes", "jet_classes", "top_classes", "w_classes")}
    assign_out = {key: torch.zeros(1, 2, n) for key, n in [("top_to_w", 3), ("top_to_bjet", 13), ("w_to_lepton", 5), ("w_to_neutrino", 5), ("w_to_jet1", 13), ("w_to_jet2", 13)]}
    assign_tgt = {key: torch.zeros(1, 2, dtype=torch.long) for key in assign_out}
    aux = {"top_reg": pred, "w_reg": pred, "top_cls": cls_out["top_logits"], "w_cls": cls_out["w_logits"], "assign": assign_out}
    kwargs = {"matching_reg_means": {"tops": mean, "ws": mean}, "matching_reg_stds": {"tops": std, "ws": std}}
    result = criterion.forward_with_aux(reg_out, cls_out, reg_tgt, cls_tgt, masks, assign_out, assign_tgt, aux_head_outputs=[aux], **kwargs)
    assert result["regression"] < 1e-6
    assert result["aux"] < 1e-6
    assert result["total"] < 1e-6
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert torch.isfinite(pred.grad).all()
