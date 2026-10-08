# Code provenance

The Clifford algebra primitives in `pat_embedder/gatr/algebra.py` originate
from Top-Lab's HiGEN `gatr-lite` implementation. The optimized geometric
algebra layers follow the related `CMS/cms-nets` implementation, including
cached algebra tensors, scaled dot-product attention, equivariant RMS
normalization, grade dropout and the optional ten-map linear basis.

These components are included in this repository; no sibling repositories
are required. Their original source attribution is preserved in the module
documentation. See the accompanying manuscript for the scientific references
and `LICENSE` for the license of this distribution.
