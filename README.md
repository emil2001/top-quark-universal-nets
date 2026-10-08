# Universal event reconstruction

Companion code for **Section 5, “Universal event reconstruction model”**, of
*Lorentz-Equivariant Geometric Algebra Transformers for Top-Quark Event
Reconstruction*, by E. Abasov, L. Dudko, A. Markina, M. Perfilov, P. Volkov,
G. Vorotnikov and A. Zaborenko.

The models map detector-level objects to a structured description of an event:
particle identities, four-momenta, decay relations and high-level observables.
The reconstruction study uses a common interface across single-top, top-pair,
three-top and four-top processes.

This distribution contains the current corrected implementation. It does not
include the separate HiGEN or FCNC classifiers, full research datasets, trained
weights, or precomputed results. A miniature sample of eight simulated events
is included to demonstrate the code and data interface; reproducing the
manuscript's numerical results requires the original datasets, trained weights and experiment configurations.

## Models and tasks

| Model | CLI `--version` | Implementation |
| --- | --- | --- |
| MLP | `v1` | Residual dense blocks over 130 flat input features |
| DETR | `v5` | Detector-token encoder and learned parton-query decoder |
| LGaTr | `vx-full` | Single stack of Clifford-algebra attention blocks with learned parton queries |

Inputs contain missing transverse momentum, up to four charged leptons and
twelve jets. The token models use 17 detector tokens and eight parton queries
(four top-quark slots and four W-boson slots). LGaTr can additionally use fixed
beam-plane and time-reference tokens with `--spurions`.

All three models jointly predict:

- **Regression:** four-momenta of charged leptons, neutrinos, jets, top quarks and W bosons.
- **Classification:** particle identities, including empty slots.
- **Assignment:** top → W/b jet and W → lepton/neutrino/two jets, including no-match choices.
- **High-level regression:** 51 event-observable targets by default.

The DETR backbone retains the internal class name `PatEmbedderModelV3`; its
supported training configuration is `v5`. Earlier independent model variants
and research ablations are excluded.

## Installation

Python 3.10–3.12 is covered by CI. The lock file provides CPU PyTorch on
Linux and Windows and native PyTorch on macOS; CUDA is not required.

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
uv run python train.py --help
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The requirements file is exported from `uv.lock`. On Windows, activate the
environment with `.venv\Scripts\activate`.

For CUDA training, use a separate environment, install the appropriate PyTorch
build using the [official installation selector](https://pytorch.org/get-started/locally/),
then install this package with `python -m pip install -e .`. The research
configuration is not tied to a particular CUDA wheel index. Select precision
and compilation settings for the hardware you use.

## Quick start on CPU

Prepare a cache from the bundled eight-event research sample:

```bash
uv run python scripts/make_example_cache.py --output data/example.pkl
```

It uses six training and two test events, with detector features, parton
labels, four-momenta, decay assignments and high-level observables from the
research sample. Normalization is calculated only from the training split.
Unobserved dimensions use neutral statistics; singly observed dimensions use
unit standard deviation. See [sample provenance](examples/README.md).
The cache builder refuses to overwrite an existing cache.

An independent synthetic alternative is also available:

```bash
uv run python scripts/make_synthetic_cache.py --output data/synthetic.pkl --seed 2025
```

This produces 32 training and 8 test events. It is a deterministic illustrative
generator, not a Monte Carlo simulation. Its 51 HLR targets exercise the
interface and do not implement the research observables. Tests use this
generator, independently of the supplied sample.

Run a small CPU example for each model:

```bash
uv run python train.py --cache-path data/example.pkl --version v1 \
  --accelerator cpu --precision 32-true --no-compile-model --max-epochs 1 \
  --v1-hidden-dim 32 --v1-num-layers 1 --run-name demo-mlp \
  --checkpoint-dir checkpoints/v1

uv run python train.py --cache-path data/example.pkl --version v5 \
  --accelerator cpu --precision 32-true --no-compile-model --max-epochs 1 \
  --v5-embed-dim 32 --v5-n-heads 2 --v5-n-enc-layers 1 --v5-n-dec-layers 1 \
  --run-name demo-detr --checkpoint-dir checkpoints/v5

uv run python train.py --cache-path data/example.pkl --version vx-full \
  --accelerator cpu --precision 32-true --no-compile-model --max-epochs 1 \
  --v5-embed-dim 32 --v5-n-heads 2 --v5-n-enc-layers 1 --v5-n-dec-layers 1 \
  --ga-channels 4 --ga-heads 1 --run-name demo-lgatr \
  --checkpoint-dir checkpoints/vx-full
```

These reduced configurations and eight-event sample are intended for a quick
check. Metrics on two test events cannot measure physics performance. MLflow writes to local `mlruns/` by
default; no tracking server or account is needed. Each checkpoint directory
contains a best-validation checkpoint and `last.ckpt`. Generated files are
ignored by Git.

Load a checkpoint and produce regression, classification and assignment plots:

```bash
uv run python scripts/infer_and_plot.py --version vx-full \
  --checkpoint checkpoints/vx-full/last.ckpt --strict-load \
  --cache-path data/example.pkl --device cpu --max-samples 2 \
  --outdir outputs/lgatr --quick-metrics
```

Repeat with `--version v1 --checkpoint checkpoints/v1/last.ckpt` or
`--version v5 --checkpoint checkpoints/v5/last.ckpt`, choosing a separate output
directory. Pass `--hlr` to include high-level regression plots.

The same command can load a locally logged best checkpoint with `--run-name
demo-lgatr` in place of `--checkpoint`. Use the same tracking URI and experiment
name as training if you override either of them.

Check the physical character of LGaTr's predicted four-momenta:

```bash
uv run python scripts/mass_shell_sanity.py --version vx-full \
  --checkpoint checkpoints/vx-full/last.ckpt --cache-path data/example.pkl \
  --device cpu --max-samples 2 --outdir outputs/mass-shell
```

## Using your own data

Training and evaluation consume a local pickle cache containing prebatched
PyTorch tensors and normalization statistics. See [the data format](docs/data-format.md)
for shapes, units, label encodings, padding and an example of creating a cache.
ROOT readers and laboratory-specific dataset paths are not part of this
distribution. Only load pickle caches and checkpoints from sources you trust.

The cache uses **contract v2**. LGaTr uses physical detector four-momenta and
**signed-energy output contract v3**; incompatible older checkpoints are
rejected. Details and the differences from the manuscript are described in
[the model contracts](docs/model-contracts.md).

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q pat_embedder scripts train.py
```

Tests generate their own synthetic data and require neither a research cache
nor an MLflow server. They cover model shapes, finite losses and gradients,
checkpoint loading, detector masks, set matching, energy decoding, physical
constraints and Lorentz-equivariant layers. End-to-end training and checkpoint
inference are exercised for all three supported model families.

After changing dependencies, regenerate both installation manifests:

```bash
uv lock
uv export --locked --no-dev --no-emit-project --emit-index-url \
  --no-annotate --no-header --no-hashes --output-file requirements.txt
```

## Citation and license

Use [CITATION.cff](CITATION.cff) or the manuscript citation below. Publication
identifiers should be added when available.

```bibtex
@unpublished{abasov2026lgatr,
  title = {Lorentz-Equivariant Geometric Algebra Transformers for Top-Quark Event Reconstruction},
  author = {Abasov, E. and Dudko, L. and Markina, A. and Perfilov, M. and
            Volkov, P. and Vorotnikov, G. and Zaborenko, A.},
  year = {2026},
  note = {Manuscript}
}
```

Code is distributed under the [MIT license](LICENSE). The provenance of the
included geometric-algebra components is recorded in [NOTICE.md](NOTICE.md).
