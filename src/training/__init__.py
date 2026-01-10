"""
Modules pour l'entraînement des modèles.
"""

from .losses import negative_log_likelihood
from .trainer import Trainer

__all__ = [
    'negative_log_likelihood',
    'Trainer'
]

