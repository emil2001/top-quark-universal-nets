# Data interface

The entry points consume a contract-v2 pickle cache. `save_cache` writes the
version marker; `load_cache` validates it and returns train/test loaders with
`batch_size=None`, because each stored item is already a batch.

## Batch structure

| Field | Shape | Meaning |
| --- | --- | --- |
| `features` | `(B, 130)` | Standardized detector input |
| `targets.regression[key]` | `(B, N_key, 7)` | Standardized regression targets |
| `targets.classification[key]` | `(B, N_key)` | Particle IDs or jet flags |
| `targets.assignments[key]` | `(B, 4)` | Child-slot indices, including no-match |
| `targets.high_level_regression` | `(B, H)` | Standardized high-level targets |
| `masks[key]` | `(B, N_key)` | True for occupied target slots |
| `masks.hlr` | `(B, H)` | True for valid high-level targets |

Particle keys and slot counts are `leptons: 4`, `neutrinos: 4`, `jets: 12`,
`tops: 4`, `ws: 4`. Features, regression and HLR targets use float32;
classification and assignment targets are integer tensors; masks are Boolean.

The input starts with `[log1p(MET pT), MET phi]`, followed by four lepton slots
and twelve jet slots. Each object slot has eight values:
`[log1p(pT), eta, phi, px, py, pz, log1p(E), identifier]`.
The identifier is the signed lepton PDG ID or a jet b-tagging value. Physical
momenta are in GeV and angles are in radians.

Regression targets use the first seven values of the same kinematic layout.
Store absent raw kinematic slots as NaN. `make_collate_and_scale` creates
occupancy masks and zeroes invalid standardized values. Detector attention
masks are computed from input kinematic features, not from parton truth.

## Labels and assignments

- Leptons: `0`, `±11`, `±13`, `±15`; neutrinos: `0`, `±12`, `±14`, `±16`.
- Top quarks: `0`, `±6`; W bosons: `0`, `±24`.
- Jets: `0` for empty, `-1` for light and `+1` for b. Jet flags are not PDG IDs.

Assignments use zero-based child-slot indices. The final index, equal to the
number of child slots, means no match:

| Assignment | Child slots | No-match index |
| --- | --- | --- |
| `top_to_w` | 4 | 4 |
| `top_to_bjet` | 12 | 12 |
| `w_to_lepton`, `w_to_neutrino` | 4 | 4 |
| `w_to_jet1`, `w_to_jet2` | 12 | 12 |

Set matching uses regression statistics for the receiving query slot. Labels
and their statistics must originate from the same cache build.

## Creating a cache

Prepare raw sample dictionaries with `features` and `targets`, split events,
then calculate statistics on training events only. For example, given lists
named `train_samples` and `test_samples`:

```python
from pat_embedder.data import collate_fn, compute_stats, make_collate_and_scale, save_cache

statistics = compute_stats([collate_fn(train_samples)])
collate = make_collate_and_scale(*statistics)
feat_mean, feat_std, reg_means, reg_stds, hlr_mean, hlr_std = statistics
save_cache(
    train_batches=[collate(train_samples)],
    test_batches=[collate(test_samples)],
    feat_mean=feat_mean, feat_std=feat_std,
    reg_means=reg_means, reg_stds=reg_stds,
    hlr_mean=hlr_mean, hlr_std=hlr_std,
    cache_dir="data", filename="custom.pkl",
)
```

For larger datasets, process samples in batches. Every normalized dimension
needs finite training observations and a finite, positive standard deviation;
retain a single set of training statistics for both splits. Feature statistics
have shape `(130,)`; regression statistics have shape `(N_key, 7)`; HLR
statistics have shape `(H,)`. Set `--num-hlr-vars H` to match your cache.

The bundled [eight-event research sample](../examples/README.md) is converted
with `scripts/make_example_cache.py`; it is not needed by the synthetic tests.
Its miniature normalization uses documented fallback statistics for dimensions
with fewer than two training observations. Full datasets should provide
adequate training observations for every modeled slot.

`pat_embedder.synthetic` provides an executable example of raw events,
normalization, batch construction and cache writing. Its HLR values are mock
targets and should not be interpreted as values of the research observables.

For a legacy cache needing a normalization rebuild, use
`python scripts/rebuild_cache.py --help`. A statistics-only sidecar is rejected:
targets and statistics must be rebuilt together. A rebuild cannot recover
missing physical information or change the meaning of source labels.
