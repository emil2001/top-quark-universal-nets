# Model contracts and relation to the manuscript

This release accompanies the multiprocess reconstruction task in Section 5.
It provides the current corrected implementation rather than an immutable
archive of the manuscript experiments. The miniature research sample, synthetic
examples and default hyperparameters do not reproduce its reported scores.

## Architecture settings

| Setting | Manuscript | Current training CLI default |
| --- | --- | --- |
| MLP dense blocks | 4 | 4 |
| DETR encoder/decoder layers | 4 / 4 | 6 / 4 |
| DETR attention heads | 4 | 4 |
| LGaTr blocks / channels / heads | 4 / 16 / 4 | 6 / 32 / 4 |
| LGaTr beam/time references | Enabled | Opt-in with `--spurions` |

`PatEmbedderModelVXFull` and its Lightning wrapper use eight blocks when
constructed directly without an override; the CLI supplies its explicit
six-layer default. The README examples use smaller one-layer models for CPU
checks. The dimensions and loss weights of historical runs are not inferred
from the architecture table.

## Data and regression

Cache contract v2 couples labels with their normalization statistics. Jet
labels are signed b-tag flags. Set matching converts targets from source-slot
statistics into the receiving query's statistics before calculating costs.
Statistics-only repairs and unversioned old caches are rejected.

LGaTr reconstructs raw detector four-momenta in physical GeV, applying `expm1`
to cached log-energy and MET log-transverse-momentum features. Padded detector
slots remain zero after denormalization.

The MLP predicts the seven standardized cache coordinates. DETR predicts
`[log1p(pT), eta, sin(phi), cos(phi), log1p(E)]`, with the relevant coordinates
standardized using persistent cache statistics.

LGaTr output contract v3 uses `sign(E) * log1p(abs(E))` for energy, preserving
the sign of negative predictions. The encoding has derivative one at zero;
its inverse is `sign(x) * expm1(abs(x))`. DETR retains unsigned log-energy.
Evaluation and the mass-shell checker respect the model's energy contract.
LGaTr checkpoints predating physical-input v2 or signed-output v3 are rejected.
To inspect such checkpoints, use their corresponding historical code or retrain.

## Symmetry and limitations

The Clifford algebra uses signature `(+,-,-,-)` and the rotor action
`X -> R X reverse(R)`. Equivariant layers are tested under Lorentz rotations
and boosts. Full-network claims also depend on reference tokens, input
features, normalization and readouts; a layer test does not prove arbitrary
end-to-end covariance.

The current LGaTr parton-query parameters remain grade-zero scalars throughout
training. The manuscript describes unconstrained query multivectors initialized
at grade zero. This is a further implementation difference from that experiment
configuration. Fixed beam-plane/time references select physical directions;
holding them fixed restricts the transformation symmetry being tested.

Training examples verify finite gradients, checkpoint round trips and execution
of all output heads. The public wrapper omits initialization of an unused
hybrid network, which changes the random-number stream for fresh weights.
A seed alone does not reproduce historical parameter initialization; saved
weights remain compatible with the current contracts. The execution tests
establish no physics performance or equivalence to
the manuscript's trained models. Raw grade-one readouts can have negative
energy or space-like four-momenta; preserving their sign makes these failures
visible rather than enforcing a physical mass shell.
