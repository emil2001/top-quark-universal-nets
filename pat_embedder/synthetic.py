"""Deterministic illustrative events for examples and tests, not a physics simulation."""

import numpy as np

from .config import DEFAULT_HIGH_LEVEL_BRANCHES, SEED
from .data import collate_fn, compute_stats, make_collate_and_scale, save_cache
from .losses import _M_TOP_GEV, _M_W_GEV

SLOTS = {"leptons": 4, "neutrinos": 4, "jets": 12, "tops": 4, "ws": 4}


def _particle(rng, mass=0.0):
    pt = rng.uniform(15.0, 100.0)
    eta = rng.uniform(-1.5, 1.5)
    phi = rng.uniform(-np.pi, np.pi)
    momentum = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])
    return np.concatenate([[np.sqrt(momentum @ momentum + mass**2)], momentum])


def _kinematics(p4):
    energy, px, py, pz = p4
    pt = np.hypot(px, py)
    return np.array(
        [np.log1p(pt), np.arcsinh(pz / max(pt, 1e-6)), np.arctan2(py, px),
         px, py, pz, np.log1p(energy)],
        dtype=np.float32,
    )


def _boost(p4, frame):
    beta = frame[1:] / frame[0]
    beta_sq = beta @ beta
    if beta_sq == 0.0:
        return p4.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    dot = beta @ p4[1:]
    momentum = p4[1:] + ((gamma - 1.0) * dot / beta_sq + gamma * p4[0]) * beta
    return np.concatenate([[gamma * (p4[0] + dot)], momentum])


def _decay(parent, first_mass, second_mass, rng):
    """Isotropic two-body toy decay conserving the parent's four-momentum."""
    mass_sq = parent[0] ** 2 - parent[1:] @ parent[1:]
    mass = np.sqrt(mass_sq)
    momentum = np.sqrt(
        (mass_sq - (first_mass + second_mass) ** 2)
        * (mass_sq - (first_mass - second_mass) ** 2)
    ) / (2.0 * mass)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    p = momentum * direction
    first = np.concatenate([[np.sqrt(momentum**2 + first_mass**2)], p])
    second = np.concatenate([[np.sqrt(momentum**2 + second_mass**2)], -p])
    return _boost(first, parent), _boost(second, parent)


def _event(rng, index):
    regression = {key: np.full((n, 7), np.nan, dtype=np.float32) for key, n in SLOTS.items()}
    classification = {key: np.zeros(n, dtype=np.int8) for key, n in SLOTS.items()}
    assignments = {
        "top_to_w": np.full(4, 4, dtype=np.int8),
        "top_to_bjet": np.full(4, 12, dtype=np.int8),
        "w_to_lepton": np.full(4, 4, dtype=np.int8),
        "w_to_neutrino": np.full(4, 4, dtype=np.int8),
        "w_to_jet1": np.full(4, 12, dtype=np.int8),
        "w_to_jet2": np.full(4, 12, dtype=np.int8),
    }
    n_tops = 1 + index % 4
    hadronic = index % 3 == 0
    n_leptons = 0 if hadronic else n_tops
    n_jets = min(12, (3 * n_tops if hadronic else n_tops) + index % 3)
    jets = [_particle(rng, mass=5.0 if j < n_tops else 0.0) for j in range(n_jets)]
    neutrinos = []
    for j in range(n_tops):
        charge = 1 if (index + j) % 2 == 0 else -1
        top = _particle(rng, mass=_M_TOP_GEV)
        w, jets[j] = _decay(top, _M_W_GEV, 5.0, rng)
        first, second = _decay(w, 0.0, 0.0, rng)
        if hadronic:
            j1, j2 = n_tops + 2 * j, n_tops + 2 * j + 1
            jets[j1], jets[j2] = first, second
            assignments["w_to_jet1"][j] = j1
            assignments["w_to_jet2"][j] = j2
        else:
            lepton, neutrino = first, second
            neutrinos.append(neutrino)
            flavour = (11, 13, 15)[(index + j) % 3]
            regression["leptons"][j] = _kinematics(lepton)
            regression["neutrinos"][j] = _kinematics(neutrino)
            classification["leptons"][j] = -charge * flavour
            classification["neutrinos"][j] = charge * (flavour + 1)
            assignments["w_to_lepton"][j] = j
            assignments["w_to_neutrino"][j] = j
        regression["ws"][j] = _kinematics(w)
        regression["tops"][j] = _kinematics(top)
        classification["tops"][j] = 6 * charge
        classification["ws"][j] = 24 * charge
        assignments["top_to_w"][j] = j
        assignments["top_to_bjet"][j] = j
    for j, p4 in enumerate(jets):
        regression["jets"][j] = _kinematics(p4)
        classification["jets"][j] = 1 if j < n_tops else -1

    met = sum((p[1:3] for p in neutrinos), start=np.zeros(2))
    if not neutrinos:
        met = rng.uniform(-20.0, 20.0, 2)
    features = np.full(130, np.nan, dtype=np.float32)
    features[:2] = [np.log1p(np.linalg.norm(met)), np.arctan2(met[1], met[0])]
    for key, offset in [("leptons", 2), ("jets", 34)]:
        slots = features[offset:offset + SLOTS[key] * 8].reshape(SLOTS[key], 8)
        slots[:, :7] = regression[key]
        slots[:, 7] = classification[key]
    # A detector-only jet exercises the separation between detector masks and
    # parton truth. It has no regression target or decay assignment.
    if n_jets < 12:
        extra_slot = features[34:].reshape(12, 8)[n_jets]
        extra_slot[:7] = _kinematics(_particle(rng))
        extra_slot[7] = -1
    # These exercise the 51-output HLR interface; they do not implement the
    # named observables of the research dataset.
    hlr = rng.normal(size=len(DEFAULT_HIGH_LEVEL_BRANCHES)).astype(np.float32)
    assert np.count_nonzero(classification["leptons"]) == n_leptons
    return {
        "features": features,
        "targets": {
            "regression": regression,
            "classification": classification,
            "assignments": assignments,
            "high_level_regression": hlr,
        },
    }


def make_synthetic_data(seed=SEED, batch_size=8, train_batches=4, test_batches=1):
    """Return contract-v2 batches and train-only normalization statistics.

    At least eight training events are needed so all configured particle slots
    have observations. Test events are generated independently after the train split.
    """
    if batch_size < 2 or train_batches < 1 or test_batches < 1 or batch_size * train_batches < 8:
        raise ValueError("Require batch_size >= 2, positive split sizes and >= 8 training events")
    rng = np.random.default_rng(seed)
    samples = [_event(rng, i) for i in range(batch_size * (train_batches + test_batches))]
    n_train = batch_size * train_batches
    statistics = compute_stats([collate_fn(samples[:n_train])])
    collate = make_collate_and_scale(*statistics)
    batches = [collate(samples[i:i + batch_size]) for i in range(0, len(samples), batch_size)]
    return {
        "train_batches": batches[:train_batches],
        "test_batches": batches[train_batches:],
        **dict(zip(
            ["feat_mean", "feat_std", "reg_means", "reg_stds", "hlr_mean", "hlr_std"],
            statistics,
        )),
    }


def save_synthetic_cache(output, seed=SEED):
    """Write a small locally generated cache compatible with training and inference."""
    from pathlib import Path

    output = Path(output)
    data = make_synthetic_data(seed=seed)
    return save_cache(
        **data,
        cache_dir=str(output.parent),
        filename=output.name,
        build_metadata={"kind": "synthetic", "seed": seed},
    )
