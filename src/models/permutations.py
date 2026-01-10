"""
Fonctions utilitaires pour générer différents types de masques pour RealNVP.

Les masques définissent quelles dimensions sont transformées dans chaque
couche de couplage affine. Différents patterns de masques peuvent avoir
un impact significatif sur la capacité du modèle à capturer des distributions
complexes.
"""

import torch
import random


def generate_alternating_masks(n_layers, dim=2):
    """
    Génère des masques alternés pour RealNVP.
    
    Pattern alterné : [1, 0], [0, 1], [1, 0], [0, 1], ...
    Ce pattern assure que chaque dimension est alternativement fixe et transformée,
    permettant à toutes les dimensions d'interagir au fil des couches.
    
    C'est le pattern le plus couramment utilisé et recommandé pour RealNVP.
    
    Args:
        n_layers (int): Nombre de couches de couplage à créer
        dim (int): Dimension de l'espace (2 pour 2D)
    
    Returns:
        list: Liste de masques, chaque masque est une liste de longueur dim
    
    Example:
        >>> masks = generate_alternating_masks(4, dim=2)
        >>> print(masks)
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    """
    masks = []
    for i in range(n_layers):
        mask = [0.0] * dim
        mask[i % dim] = 1.0
        masks.append(mask)
    return masks


def generate_checkerboard_masks(n_layers, dim=2):
    """
    Génère des masques en damier (checkerboard) pour RealNVP.
    
    Pour 2D, cela revient au même que les masques alternés.
    Pour des dimensions plus élevées, cela créerait un pattern en damier.
    
    Args:
        n_layers (int): Nombre de couches de couplage à créer
        dim (int): Dimension de l'espace (2 pour 2D)
    
    Returns:
        list: Liste de masques en pattern damier
    
    Note:
        Pour dim=2, cette fonction est équivalente à generate_alternating_masks
    """
    if dim == 2:
        return generate_alternating_masks(n_layers, dim)
    
    masks = []
    for i in range(n_layers):
        mask = []
        for j in range(dim):
            value = 1.0 if ((i + j) % 2 == 0) else 0.0
            mask.append(value)
        masks.append(mask)
    return masks


def generate_random_masks(n_layers, dim=2, seed=None):
    """
    Génère des masques aléatoires pour RealNVP.
    
    Chaque masque est généré aléatoirement avec exactement une dimension
    masquée (valeur 1) et les autres transformées (valeur 0).
    
    Attention : Les masques aléatoires peuvent ne pas garantir que toutes
    les dimensions sont transformées de manière équilibrée, ce qui peut
    affecter les performances du modèle.
    
    Args:
        n_layers (int): Nombre de couches de couplage à créer
        dim (int): Dimension de l'espace (2 pour 2D)
        seed (int, optional): Graine pour la reproductibilité
    
    Returns:
        list: Liste de masques aléatoires
    
    Example:
        >>> masks = generate_random_masks(4, dim=2, seed=42)
        >>> print(masks)
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    """
    if seed is not None:
        random.seed(seed)
    
    masks = []
    for i in range(n_layers):
        mask = [0.0] * dim
        masked_dim = random.randint(0, dim - 1)
        mask[masked_dim] = 1.0
        masks.append(mask)
    
    return masks


def generate_custom_masks(mask_patterns):
    """
    Génère des masques à partir d'un pattern personnalisé.
    
    Permet de définir manuellement les masques pour des expériences spécifiques.
    
    Args:
        mask_patterns (list): Liste de patterns, chaque pattern est une liste
            de valeurs 0.0 ou 1.0
    
    Returns:
        list: Liste de masques correspondant aux patterns fournis
    
    Example:
        >>> custom = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        >>> masks = generate_custom_masks(custom)
    """
    return mask_patterns.copy()

