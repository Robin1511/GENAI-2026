"""
Fonctions de visualisation pour les résultats de RealNVP.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_transformation(x, z, labels=None, title="Transformation", save_path=None):
    """
    Visualise la transformation x -> z côte à côte.
    
    Args:
        x (torch.Tensor or numpy.ndarray): Données originales de forme (n_samples, 2)
        z (torch.Tensor or numpy.ndarray): Données transformées de forme (n_samples, 2)
        labels (array, optional): Labels pour colorer les points
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(z, torch.Tensor):
        z = z.cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    ax = axes[0]
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(x[mask, 0], x[mask, 1], alpha=0.6, s=20, label=f'Classe {label}')
        ax.legend()
    else:
        ax.scatter(x[:, 0], x[:, 1], alpha=0.6, s=20)
    ax.set_title("X (données originales)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    ax = axes[1]
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(z[mask, 0], z[mask, 1], alpha=0.6, s=20, label=f'Classe {label}')
        ax.legend()
    else:
        ax.scatter(z[:, 0], z[:, 1], alpha=0.6, s=20)
    ax.set_title("Z (transformé vers espace latent)")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()


def plot_training_curve(losses, title="Courbe d'entraînement", save_path=None):
    """
    Visualise l'évolution de la perte pendant l'entraînement.
    
    Args:
        losses (list): Liste des valeurs de perte à chaque étape
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.7, linewidth=1)
    plt.xlabel("Étape d'entraînement")
    plt.ylabel("Negative Log-Likelihood")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if len(losses) > 50:
        window = min(50, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 
                label=f'Moyenne mobile (fenêtre={window})', linewidth=2)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()


def plot_samples_comparison(real_samples, generated_samples, 
                           labels_real=None, title="Comparaison échantillons",
                           save_path=None):
    """
    Compare les échantillons réels et générés côte à côte.
    
    Args:
        real_samples (torch.Tensor or numpy.ndarray): Échantillons réels
        generated_samples (torch.Tensor or numpy.ndarray): Échantillons générés
        labels_real (array, optional): Labels pour les échantillons réels
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().detach().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    ax = axes[0]
    if labels_real is not None:
        unique_labels = np.unique(labels_real)
        for label in unique_labels:
            mask = labels_real == label
            ax.scatter(real_samples[mask, 0], real_samples[mask, 1], 
                      alpha=0.6, s=20, label=f'Classe {label}')
        ax.legend()
    else:
        ax.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.6, s=20)
    ax.set_title("Échantillons réels")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    ax = axes[1]
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
              alpha=0.6, s=20, color='orange')
    ax.set_title("Échantillons générés")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()


def plot_multiple_distributions(distributions_dict, title="Distributions multiples",
                                save_path=None):
    """
    Visualise plusieurs distributions dans une grille.
    
    Args:
        distributions_dict (dict): Dictionnaire {nom: (X, labels)} où X est (n_samples, 2)
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    n_distributions = len(distributions_dict)
    n_cols = min(3, n_distributions)
    n_rows = (n_distributions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_distributions == 1:
        axes_list = [axes]
    elif n_rows == 1:
        axes_list = list(axes.flatten()) if hasattr(axes, 'flatten') else [axes]
    else:
        axes_list = list(axes.flatten())
    
    for idx, (name, (X, labels)) in enumerate(distributions_dict.items()):
        ax = axes_list[idx]
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], alpha=0.6, s=20, label=f'Classe {label}')
            if len(unique_labels) <= 5:
                ax.legend()
        else:
            ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
        
        ax.set_title(name)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    for idx in range(n_distributions, len(axes_list)):
        axes_list[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()

