"""
Configuration et constantes par défaut pour le projet RealNVP.
"""

# Configuration par défaut pour l'entraînement
DEFAULT_CONFIG = {
    'hidden_dim': 128,           # Dimension des couches cachées
    'learning_rate': 0.0001,      # Learning rate pour Adam
    'n_steps': 5000,              # Nombre d'étapes d'entraînement
    'batch_size': 512,            # Taille du batch
    'print_every': 100,           # Fréquence d'affichage
    'device': 'auto',             # Device ('auto', 'cpu', ou 'cuda')
}

# Configuration pour les distributions
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

# Configuration pour la visualisation
VISUALIZATION_CONFIG = {
    'density_resolution': 100,    # Résolution pour le calcul de densité
    'xlim': (-4, 4),              # Limites en x
    'ylim': (-4, 4),              # Limites en y
    'figsize': (12.8, 4.8),       # Taille par défaut des figures
}

