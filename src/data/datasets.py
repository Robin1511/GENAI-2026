"""
Wrappers PyTorch pour les datasets de distributions.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class DistributionDataset(Dataset):
    """
    Dataset PyTorch pour les distributions 2D générées dynamiquement.
    
    Ce dataset permet de générer des échantillons à la volée à chaque
    accès, ce qui est utile pour l'entraînement où on veut toujours
    avoir des échantillons frais.
    
    Attributes:
        generator_func (callable): Fonction qui génère les échantillons
        generator_args (dict): Arguments à passer à la fonction génératrice
        n_samples (int): Nombre d'échantillons à générer à chaque accès
    """
    
    def __init__(self, generator_func, n_samples=512, **generator_args):
        """
        Initialise le dataset.
        
        Args:
            generator_func (callable): Fonction qui génère (X, labels)
            n_samples (int): Nombre d'échantillons à générer
            **generator_args: Arguments additionnels pour la fonction génératrice
        """
        self.generator_func = generator_func
        self.generator_args = generator_args
        self.n_samples = n_samples
    
    def __len__(self):
        """Retourne le nombre d'échantillons."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Génère un batch d'échantillons.
        
        Note: Pour un dataset dynamique, on génère toujours un nouveau batch
        complet. L'index idx est ignoré mais nécessaire pour l'interface Dataset.
        
        Args:
            idx (int): Index (ignoré pour génération dynamique)
        
        Returns:
            torch.Tensor: Échantillons de forme (n_samples, 2)
        """
        # Générer de nouveaux échantillons à chaque accès
        X, _ = self.generator_func(n_samples=self.n_samples, **self.generator_args)
        
        # Convertir en tensor PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float64)
        
        return X_tensor


class StaticDistributionDataset(Dataset):
    """
    Dataset PyTorch pour des distributions statiques (pré-générées).
    
    Utile pour l'évaluation où on veut utiliser le même ensemble de données.
    
    Attributes:
        data (torch.Tensor): Données pré-générées de forme (n_samples, 2)
        labels (torch.Tensor): Labels correspondants de forme (n_samples,)
    """
    
    def __init__(self, X, labels=None):
        """
        Initialise le dataset avec des données pré-générées.
        
        Args:
            X (numpy.ndarray or torch.Tensor): Données de forme (n_samples, 2)
            labels (numpy.ndarray or torch.Tensor, optional): Labels correspondants
        """
        if isinstance(X, np.ndarray):
            self.data = torch.tensor(X, dtype=torch.float64)
        else:
            self.data = X
        
        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.tensor(labels, dtype=torch.long)
            else:
                self.labels = labels
        else:
            self.labels = None
    
    def __len__(self):
        """Retourne le nombre d'échantillons."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retourne un échantillon à l'index donné.
        
        Args:
            idx (int): Index de l'échantillon
        
        Returns:
            torch.Tensor or tuple: Échantillon (et labels si disponibles)
        """
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

