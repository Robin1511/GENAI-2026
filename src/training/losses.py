"""
Fonctions de perte pour l'entraînement de RealNVP.

La fonction de perte principale est la negative log-likelihood (NLL)
basée sur la formule de changement de variables pour les flots normalisants.
"""

import torch
import math


def negative_log_likelihood(z, logdet):
    """
    Calcule la negative log-likelihood pour RealNVP.
    
    Formule complète :
        NLL = log(2π) + mean(0.5 * ||z||² - logdet)
    
    Dérivation théorique :
    ----------------------
    Soit p_z(z) la distribution de base (gaussienne standard) et f la transformation
    RealNVP qui mappe z -> x. Par la formule de changement de variables :
    
        p_x(x) = p_z(z) * |det(J_f^{-1}(x))|
    
    où J_f^{-1} est le Jacobien de la transformation inverse.
    
    En prenant le log :
        log p_x(x) = log p_z(z) + log |det(J_f^{-1}(x))|
    
    Pour une gaussienne standard 2D :
        log p_z(z) = -log(2π) - 0.5 * ||z||²
    
    Le log-déterminant du Jacobien inverse est donné par logdet (calculé par le modèle).
    
    Donc :
        log p_x(x) = -log(2π) - 0.5 * ||z||² + logdet
    
    La negative log-likelihood est donc :
        NLL = -log p_x(x) = log(2π) + 0.5 * ||z||² - logdet
    
    Le terme -logdet corrige le changement de volume dans l'espace lors de la
    transformation. Si la transformation étire l'espace (det > 1), logdet > 0 et
    on soustrait pour compenser. Si elle compresse (det < 1), logdet < 0 et on
    ajoute pour compenser.
    
    Args:
        z (torch.Tensor): Variables latentes de forme (batch_size, dim)
            Typiquement des échantillons transformés depuis l'espace observé
        logdet (torch.Tensor): Log-déterminant du Jacobien de forme (batch_size,)
            Calculé par le modèle lors de la transformation inverse
    
    Returns:
        torch.Tensor: Negative log-likelihood moyenne (scalaire)
    
    Example:
        >>> z = torch.randn(100, 2)
        >>> logdet = torch.randn(100)
        >>> loss = negative_log_likelihood(z, logdet)
        >>> print(loss.item())
    """
    z_norm_sq = torch.sum(0.5 * z ** 2, dim=-1)
    
    log_2pi = torch.log(z.new_tensor([2 * math.pi]))
    nll_per_sample = log_2pi + z_norm_sq - logdet
    
    return torch.mean(nll_per_sample)


def compute_log_likelihood(z, logdet):
    """
    Calcule la log-likelihood (sans le signe négatif).
    
    Utile pour l'évaluation et le monitoring pendant l'entraînement.
    
    Args:
        z (torch.Tensor): Variables latentes de forme (batch_size, dim)
        logdet (torch.Tensor): Log-déterminant du Jacobien de forme (batch_size,)
    
    Returns:
        torch.Tensor: Log-likelihood moyenne (scalaire)
    """
    return -negative_log_likelihood(z, logdet)

