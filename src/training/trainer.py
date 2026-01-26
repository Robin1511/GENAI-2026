"""
Classe Trainer pour gérer l'entraînement et l'évaluation des modèles RealNVP.
"""

import torch
import torch.optim as optim
import time
from typing import Callable, Optional, List, Dict


class Trainer:
    """
    Classe pour entraîner et évaluer des modèles RealNVP.

    Cette classe encapsule la logique d'entraînement, incluant la boucle
    d'entraînement, le suivi des pertes, et la sauvegarde/chargement des modèles.

    Attributes:
        model: Modèle RealNVP à entraîner
        optimizer: Optimiseur PyTorch
        device: Device (CPU ou GPU) utilisé pour l'entraînement
        loss_history: Historique des pertes pendant l'entraînement
    """

    def __init__(self, model, lr=0.0001, device="auto"):
        """
        Initialise le trainer.

        Args:
            model: Modèle RealNVP à entraîner
            lr (float): Learning rate pour l'optimiseur Adam
            device (str): Device à utiliser ('auto', 'cpu', ou 'cuda')
                Si 'auto', détecte automatiquement GPU/CPU
        """
        self.model = model

        # Détection automatique du device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Déplacer le modèle sur le device
        self.model = self.model.to(self.device)

        # Initialiser l'optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Historique des pertes
        self.loss_history = []

    def train_step(self, data_generator: Callable):
        """
        Effectue une étape d'entraînement.

        Args:
            data_generator (callable): Fonction qui génère un batch de données
                Doit retourner un tensor de forme (batch_size, 2)

        Returns:
            float: Valeur de la perte pour cette étape
        """
        X = data_generator()
        if isinstance(X, tuple):
            X = X[0]
        X = X.to(self.device)

        z, logdet = self.model.inverse(X)

        from .losses import negative_log_likelihood

        loss = negative_log_likelihood(z, logdet)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        data_generator: Callable,
        n_steps: int,
        print_every: int = 100,
        verbose: bool = True,
    ):
        """
        Entraîne le modèle pendant n_steps étapes.

        Args:
            data_generator (callable): Fonction qui génère des batches de données
            n_steps (int): Nombre d'étapes d'entraînement
            print_every (int): Fréquence d'affichage de la perte
            verbose (bool): Si True, affiche la progression

        Returns:
            list: Historique des pertes
        """
        self.loss_history = []
        start_time = time.time()

        for step in range(n_steps):
            loss = self.train_step(data_generator)
            self.loss_history.append(loss)

            if verbose and (step + 1) % print_every == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Step {step + 1}/{n_steps}, Loss: {loss:.5f}, "
                    f"Time: {elapsed_time:.2f}s"
                )

        if verbose:
            total_time = time.time() - start_time
            print(f"\nEntraînement terminé en {total_time:.2f}s")
            print(f"Perte finale: {self.loss_history[-1]:.5f}")

        return self.loss_history

    def evaluate(self, X: torch.Tensor) -> Dict[str, float]:
        """
        Évalue le modèle sur un ensemble de données.

        Args:
            X (torch.Tensor): Données d'évaluation de forme (n_samples, 2)

        Returns:
            dict: Dictionnaire contenant les métriques d'évaluation
        """
        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            z, logdet = self.model.inverse(X)

            from .losses import negative_log_likelihood

            nll = negative_log_likelihood(z, logdet)

            z_mean = torch.mean(z, dim=0)
            z_std = torch.std(z, dim=0)

        self.model.train()

        return {
            "nll": nll.item(),
            "z_mean": z_mean.cpu().numpy(),
            "z_std": z_std.cpu().numpy(),
        }

    def sample(self, n_samples: int = 1000) -> torch.Tensor:
        """
        Génère des échantillons depuis le modèle.

        Args:
            n_samples (int): Nombre d'échantillons à générer

        Returns:
            torch.Tensor: Échantillons générés de forme (n_samples, 2)
        """
        self.model.eval()

        with torch.no_grad():
            z = torch.normal(0, 1, size=(n_samples, 2)).to(self.device)

            x, _ = self.model(z)

        self.model.train()

        return x.cpu()

    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle et l'optimiseur.

        Args:
            filepath (str): Chemin où sauvegarder le modèle
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            filepath,
        )
        print(f"Modèle sauvegardé dans {filepath}")

    def load_model(self, filepath: str):
        """
        Charge un modèle et un optimiseur sauvegardés.

        Args:
            filepath (str): Chemin vers le fichier de sauvegarde
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint.get("loss_history", [])
        print(f"Modèle chargé depuis {filepath}")
