"""
Configuration et constantes par défaut pour le projet RealNVP.
"""

DEFAULT_CONFIG = {
    'hidden_dim': 128,
    'learning_rate': 0.0001,
    'n_steps': 5000,
    'batch_size': 512,
    'print_every': 100,
    'device': 'auto',
}

DISTRIBUTION_CONFIG = {
    'two_moons': {
        'noise': 0.05,
    },
    'circles': {
        'noise': 0.05,
        'factor': 0.5,
    },
    'gaussian_mixture': {
        'n_components': 3,
    },
}

VISUALIZATION_CONFIG = {
    'density_resolution': 100,
    'xlim': (-4, 4),
    'ylim': (-4, 4),
    'figsize': (12.8, 4.8),
}

