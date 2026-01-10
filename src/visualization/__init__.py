"""
Modules pour la visualisation des résultats.
"""

from .plots import (
    plot_transformation,
    plot_training_curve,
    plot_samples_comparison,
    plot_multiple_distributions
)
from .density import (
    compute_density_grid,
    plot_density_heatmap,
    plot_density_comparison
)

__all__ = [
    'plot_transformation',
    'plot_training_curve',
    'plot_samples_comparison',
    'plot_multiple_distributions',
    'compute_density_grid',
    'plot_density_heatmap',
    'plot_density_comparison'
]

