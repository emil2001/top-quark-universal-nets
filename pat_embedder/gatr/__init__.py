"""GATr primitives for Cl(1,3) — copied from higen with padding-mask patch."""
from .algebra import (
    CAYLEY_DENSE,
    GRADE1_INDICES,
    GRADE_INDICES,
    N_BASIS,
    apply_rotor,
    clifford_reverse,
    geometric_product,
    inner_product,
    mv_exp,
    random_rotor,
)
from .layers import (
    BlockConfig,
    EquivariantLinear,
    GatedActivation,
    GATrBlock,
    GeometricProduct,
    MultiHeadScalarAttention,
    ScalarAttention,
)

__all__ = [
    "CAYLEY_DENSE",
    "GRADE_INDICES",
    "GRADE1_INDICES",
    "N_BASIS",
    "apply_rotor",
    "clifford_reverse",
    "geometric_product",
    "inner_product",
    "mv_exp",
    "random_rotor",
    "BlockConfig",
    "EquivariantLinear",
    "GATrBlock",
    "GatedActivation",
    "GeometricProduct",
    "MultiHeadScalarAttention",
    "ScalarAttention",
]
