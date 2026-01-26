"""
Modules pour les modèles RealNVP.
"""

from .coupling import AffineCoupling
from .realnvp import RealNVP_2D
from .layers import BatchNorm1d
from .permutations import (
    generate_alternating_masks,
    generate_checkerboard_masks,
    generate_random_masks,
)

__all__ = [
    "AffineCoupling",
    "RealNVP_2D",
    "BatchNorm1d",
    "generate_alternating_masks",
    "generate_checkerboard_masks",
    "generate_random_masks",
]
