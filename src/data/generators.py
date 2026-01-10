"""
Générateurs de distributions 2D pour l'entraînement de RealNVP.

Ces fonctions génèrent des échantillons de différentes distributions 2D
qui serviront de données d'entraînement pour les modèles RealNVP.
"""

import numpy as np
import torch
from sklearn import datasets


def generate_two_moons(n_samples=1000, noise=0.05):
    """
    Génère des échantillons de la distribution two-moons.
    
    La distribution two-moons consiste en deux demi-cercles entrelacés,
    créant une distribution non-linéaire et non-convexe.
    
    Args:
        n_samples (int): Nombre d'échantillons à générer
        noise (float): Niveau de bruit (écart-type du bruit gaussien)
    
    Returns:
        tuple: (X, labels) où
            - X (numpy.ndarray): Échantillons de forme (n_samples, 2)
            - labels (numpy.ndarray): Labels de classe (0 ou 1) de forme (n_samples,)
    
    Example:
        >>> X, labels = generate_two_moons(n_samples=1000, noise=0.05)
        >>> print(X.shape)  # (1000, 2)
    """
    X, labels = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, labels


def generate_circles(n_samples=1000, noise=0.05, factor=0.5):
    """
    Génère des échantillons de la distribution cercles concentriques.
    
    La distribution consiste en deux cercles concentriques, créant une
    distribution avec une structure circulaire complexe.
    
    Args:
        n_samples (int): Nombre d'échantillons à générer
        noise (float): Niveau de bruit (écart-type du bruit gaussien)
        factor (float): Facteur de séparation entre les cercles (0 < factor < 1)
    
    Returns:
        tuple: (X, labels) où
            - X (numpy.ndarray): Échantillons de forme (n_samples, 2)
            - labels (numpy.ndarray): Labels de classe (0 ou 1) de forme (n_samples,)
    
    Example:
        >>> X, labels = generate_circles(n_samples=1000, noise=0.05)
        >>> print(X.shape)  # (1000, 2)
    """
    X, labels = datasets.make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=42
    )
    return X, labels


def generate_gaussian_mixture(n_samples=1000, n_components=3, means=None, covs=None, weights=None):
    """
    Génère des échantillons d'une mixture de gaussiennes 2D.
    
    Crée une distribution complexe en mélangeant plusieurs distributions
    gaussiennes 2D avec des moyennes et covariances différentes.
    
    Args:
        n_samples (int): Nombre d'échantillons à générer
        n_components (int): Nombre de composantes gaussiennes dans le mélange
        means (list, optional): Liste de moyennes, chaque moyenne est un array de forme (2,)
            Si None, génère des moyennes aléatoires dans [-2, 2]²
        covs (list, optional): Liste de matrices de covariance, chaque matrice est (2, 2)
            Si None, utilise des matrices identité
        weights (array, optional): Poids de chaque composante (doit sommer à 1)
            Si None, utilise des poids uniformes
    
    Returns:
        tuple: (X, labels) où
            - X (numpy.ndarray): Échantillons de forme (n_samples, 2)
            - labels (numpy.ndarray): Labels de composante (0 à n_components-1) de forme (n_samples,)
    
    Example:
        >>> # Mixture avec 3 composantes
        >>> X, labels = generate_gaussian_mixture(n_samples=1000, n_components=3)
        >>> 
        >>> # Mixture personnalisée
        >>> means = [[-2, -2], [2, 2], [0, 0]]
        >>> covs = [np.eye(2)*0.5, np.eye(2)*0.5, np.eye(2)*0.3]
        >>> X, labels = generate_gaussian_mixture(n_samples=1000, means=means, covs=covs)
    """
    # Générer des moyennes aléatoires si non fournies
    if means is None:
        np.random.seed(42)
        means = [
            np.random.uniform(-2, 2, size=2) for _ in range(n_components)
        ]
    
    # Utiliser des matrices identité si non fournies
    if covs is None:
        covs = [np.eye(2) * 0.3 for _ in range(n_components)]
    
    # Utiliser des poids uniformes si non fournis
    if weights is None:
        weights = np.ones(n_components) / n_components
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normaliser
    
    # Générer les échantillons
    np.random.seed(42)
    samples_per_component = np.random.multinomial(n_samples, weights)
    
    X_list = []
    labels_list = []
    
    for i, (mean, cov, n) in enumerate(zip(means, covs, samples_per_component)):
        # Générer des échantillons pour cette composante
        X_component = np.random.multivariate_normal(mean, cov, size=n)
        X_list.append(X_component)
        labels_list.append(np.full(n, i))
    
    # Concaténer tous les échantillons
    X = np.vstack(X_list)
    labels = np.hstack(labels_list)
    
    # Mélanger les échantillons
    indices = np.random.permutation(len(X))
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def generate_custom_gaussian_mixture_2d():
    """
    Génère une mixture de gaussiennes 2D avec une configuration prédéfinie.
    
    Cette fonction crée une mixture avec 3 composantes bien séparées
    pour des visualisations claires.
    
    Returns:
        tuple: (X, labels) avec 1000 échantillons par défaut
    """
    means = [
        [-1.5, -1.5],  # Composante en bas à gauche
        [1.5, 1.5],    # Composante en haut à droite
        [0, 0]         # Composante au centre
    ]
    
    covs = [
        np.eye(2) * 0.4,  # Covariance plus large
        np.eye(2) * 0.4,
        np.eye(2) * 0.25  # Covariance plus serrée pour le centre
    ]
    
    return generate_gaussian_mixture(
        n_samples=1000,
        n_components=3,
        means=means,
        covs=covs,
        weights=[0.3, 0.3, 0.4]
    )

