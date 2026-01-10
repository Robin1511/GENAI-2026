"""
Implémentation de la couche de couplage affine pour RealNVP.

Cette couche implémente la transformation affine de couplage qui est au cœur
de RealNVP. Elle permet de créer des transformations inversibles avec un
calcul efficace du déterminant du Jacobien.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class AffineCoupling(nn.Module):
    """
    Couche de couplage affine pour RealNVP.
    
    Cette couche divise l'entrée en deux parties selon un masque :
    - La partie masquée (mask=1) reste inchangée
    - La partie non masquée (mask=0) subit une transformation affine
      paramétrée par la partie masquée
    
    La transformation affine est de la forme :
        y = mask*x + (1-mask)*(x*exp(s) + t)
    où s (scale) et t (translation) sont calculés par des réseaux de neurones
    à partir de la partie masquée de x.
    
    Attributes:
        input_dim (int): Dimension de l'entrée (2 pour 2D)
        hidden_dim (int): Dimension des couches cachées des réseaux
        mask (torch.Tensor): Masque binaire définissant quelles dimensions sont fixes
        scale_fc1, scale_fc2, scale_fc3 (nn.Linear): Réseau pour calculer le scale
        translation_fc1, translation_fc2, translation_fc3 (nn.Linear): Réseau pour calculer la translation
        scale (nn.Parameter): Paramètre de scale initialisé aléatoirement
    """
    
    def __init__(self, mask, hidden_dim):
        """
        Initialise la couche de couplage affine.
        
        Args:
            mask (list or torch.Tensor): Masque binaire de forme (input_dim,)
                où 1 signifie que la dimension reste fixe et 0 qu'elle est transformée
            hidden_dim (int): Dimension des couches cachées des réseaux de neurones
        """
        super(AffineCoupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        
        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)
        
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)
        
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
    
    def _compute_scale(self, x):
        """
        Calcule le facteur de scale s à partir de la partie masquée de x.
        
        Le réseau prend en entrée uniquement la partie masquée (x * mask)
        et produit un vecteur de scale pour toutes les dimensions.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Facteur de scale de forme (batch_size, input_dim)
        """
        masked_x = x * self.mask
        
        s = torch.relu(self.scale_fc1(masked_x))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s))
        
        s = s * self.scale
        
        return s
    
    def _compute_translation(self, x):
        """
        Calcule le vecteur de translation t à partir de la partie masquée de x.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Vecteur de translation de forme (batch_size, input_dim)
        """
        masked_x = x * self.mask
        
        t = torch.relu(self.translation_fc1(masked_x))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        
        return t
    
    def forward(self, x):
        """
        Transforme x de l'espace latent vers l'espace observé.
        
        Transformation : y = mask*x + (1-mask)*(x*exp(s) + t)
        
        Le log-déterminant du Jacobien est simplement sum((1-mask)*s) car :
        - Les dimensions masquées ont un Jacobien de 1 (pas de changement)
        - Les dimensions transformées ont un Jacobien de exp(s)
        - Le log-det est donc sum((1-mask)*s)
        
        Args:
            x (torch.Tensor): Variables latentes de forme (batch_size, input_dim)
        
        Returns:
            tuple: (y, logdet) où
                - y (torch.Tensor): Variables observées de forme (batch_size, input_dim)
                - logdet (torch.Tensor): Log-déterminant du Jacobien de forme (batch_size,)
        """
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        
        logdet = torch.sum((1 - self.mask) * s, dim=-1)
        
        return y, logdet
    
    def inverse(self, y):
        """
        Transforme y de l'espace observé vers l'espace latent (transformation inverse).
        
        Transformation inverse : x = mask*y + (1-mask)*((y-t)*exp(-s))
        
        Le log-déterminant est -sum((1-mask)*s) car on inverse la transformation.
        
        Args:
            y (torch.Tensor): Variables observées de forme (batch_size, input_dim)
        
        Returns:
            tuple: (x, logdet) où
                - x (torch.Tensor): Variables latentes de forme (batch_size, input_dim)
                - logdet (torch.Tensor): Log-déterminant du Jacobien inverse de forme (batch_size,)
        """
        s = self._compute_scale(y)
        t = self._compute_translation(y)
        
        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        
        logdet = torch.sum((1 - self.mask) * (-s), dim=-1)
        
        return x, logdet

