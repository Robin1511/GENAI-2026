"""
Couches personnalisées pour RealNVP.
"""

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """
    Batch Normalization pour les Normalizing Flows (1D).

    Cette couche implémente la batch normalization sans paramètres affines apprenables
    par défaut (pour préserver l'inversibilité simple), mais maintient les statistiques
    mobiles pour l'inférence.

    Elle retourne également le log-déterminant du Jacobien de la transformation,
    nécessaire pour le calcul de la vraisemblance dans les flows.
    """

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Statistiques mobiles (non apprenables par backprop, mais mises à jour)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, reverse=False):
        """
        Applique la batch normalization ou son inverse.

        Args:
            x (torch.Tensor): Entrée de forme (batch_size, num_features)
            reverse (bool): Si True, applique la transformation inverse

        Returns:
            tuple: (y, logdet)
                y: Sortie transformée
                logdet: Log-déterminant du Jacobien (somme sur les dimensions)
        """
        if reverse:
            # Inverse : x = y * std + mean
            # On utilise toujours les statistiques mobiles en mode inverse (génération)
            mean = self.running_mean
            var = self.running_var

            x = x * torch.sqrt(var + self.eps) + mean

            # Log-det pour l'inverse est sum(log(std))
            logdet = 0.5 * torch.sum(torch.log(var + self.eps))
            # On étend logdet à la taille du batch
            logdet = logdet.expand(x.size(0))

            return x, logdet

        else:
            # Forward : y = (x - mean) / std
            if self.training:
                # En entraînement, on utilise les stats du batch courant et on met à jour les running stats
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)

                with torch.no_grad():
                    self.running_mean.mul_(self.momentum).add_(
                        (1 - self.momentum) * batch_mean
                    )
                    self.running_var.mul_(self.momentum).add_(
                        (1 - self.momentum) * batch_var
                    )

                mean = batch_mean
                var = batch_var
            else:
                # En évaluation, on utilise les statistiques mobiles
                mean = self.running_mean
                var = self.running_var

            y = (x - mean) / torch.sqrt(var + self.eps)

            # Log-det pour forward est -sum(log(std))
            logdet = -0.5 * torch.sum(torch.log(var + self.eps))
            # On étend logdet à la taille du batch
            logdet = logdet.expand(x.size(0))

            return y, logdet
