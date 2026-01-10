"""
Implémentation du modèle RealNVP pour distributions 2D.

RealNVP (Real-valued Non-Volume Preserving) est un type de flot normalisant
qui compose plusieurs transformations de couplage affine pour créer une
transformation complexe et inversible entre un espace latent simple (gaussien)
et un espace observé complexe.
"""

import torch
import torch.nn as nn
from .coupling import AffineCoupling


class RealNVP_2D(nn.Module):
    """
    Modèle RealNVP pour modéliser des distributions 2D.
    
    Ce modèle compose plusieurs couches de couplage affine pour créer une
    transformation inversible entre un espace latent gaussien standard et
    un espace observé complexe. Une couche de normalisation finale limite
    les valeurs à l'intervalle [-4, 4] pour éviter les valeurs extrêmes.
    
    Attributes:
        hidden_dim (int): Dimension des couches cachées dans les réseaux de couplage
        masks (nn.ParameterList): Liste des masques pour chaque couche de couplage
        affine_couplings (nn.ModuleList): Liste des couches de couplage affine
    """
    
    def __init__(self, masks, hidden_dim):
        """
        Initialise le modèle RealNVP_2D.
        
        Args:
            masks (list): Liste de masques, chaque masque définit une couche de couplage
                Chaque masque est une liste de longueur 2 (pour 2D)
                Exemple : [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], ...]
            hidden_dim (int): Dimension des couches cachées dans les réseaux de couplage
        """
        super(RealNVP_2D, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.masks = nn.ParameterList([
            nn.Parameter(torch.Tensor(m), requires_grad=False)
            for m in masks
        ])
        
        self.affine_couplings = nn.ModuleList([
            AffineCoupling(self.masks[i], self.hidden_dim)
            for i in range(len(self.masks))
        ])
    
    def forward(self, x):
        """
        Transforme x de l'espace latent vers l'espace observé.
        
        La transformation consiste en :
        1. Application séquentielle de toutes les couches de couplage affine
        2. Application d'une couche de normalisation finale : y = 4*tanh(x)
           pour limiter les valeurs à l'intervalle [-4, 4]
        
        Le log-déterminant total est la somme des log-déterminants de chaque
        transformation.
        
        Args:
            x (torch.Tensor): Variables latentes de forme (batch_size, 2)
                Typiquement des échantillons d'une distribution normale standard
        
        Returns:
            tuple: (y, logdet_tot) où
                - y (torch.Tensor): Variables observées de forme (batch_size, 2)
                - logdet_tot (torch.Tensor): Log-déterminant total de forme (batch_size,)
        """
        y = x
        logdet_tot = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet
        
        logdet_normalization = torch.sum(
            torch.log(torch.abs(4 * (1 - torch.tanh(y) ** 2))),
            dim=-1
        )
        y = 4 * torch.tanh(y)
        logdet_tot = logdet_tot + logdet_normalization
        
        return y, logdet_tot
    
    def inverse(self, y):
        """
        Transforme y de l'espace observé vers l'espace latent (transformation inverse).
        
        La transformation inverse consiste en :
        1. Inversion de la couche de normalisation : x = atanh(y/4)
        2. Application inverse séquentielle de toutes les couches de couplage
        
        Args:
            y (torch.Tensor): Variables observées de forme (batch_size, 2)
                Valeurs dans l'intervalle [-4, 4]
        
        Returns:
            tuple: (x, logdet_tot) où
                - x (torch.Tensor): Variables latentes de forme (batch_size, 2)
                - logdet_tot (torch.Tensor): Log-déterminant total inverse de forme (batch_size,)
        """
        x = y
        logdet_tot = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        
        logdet_normalization = torch.sum(
            torch.log(torch.abs(1.0 / 4.0 * 1 / (1 - (x / 4) ** 2))),
            dim=-1
        )
        x = 0.5 * torch.log((1 + x / 4) / (1 - x / 4))
        logdet_tot = logdet_tot + logdet_normalization
        
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
        
        return x, logdet_tot

