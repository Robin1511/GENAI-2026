"""
Fonctions pour visualiser les densités apprises par RealNVP.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def compute_density_grid(model, xlim=(-4, 4), ylim=(-4, 4), resolution=100):
    """
    Calcule la densité apprise sur une grille 2D.
    
    La densité est calculée en transformant chaque point de la grille vers
    l'espace latent, puis en calculant la densité selon la formule de
    changement de variables.
    
    Args:
        model: Modèle RealNVP entraîné
        xlim (tuple): Limites en x (x_min, x_max)
        ylim (tuple): Limites en y (y_min, y_max)
        resolution (int): Résolution de la grille (resolution x resolution)
    
    Returns:
        tuple: (X_grid, Y_grid, density) où
            - X_grid, Y_grid: Grilles de coordonnées
            - density: Densité calculée sur la grille
    """
    # Créer la grille
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Aplatir la grille pour le traitement par batch
    grid_points = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float64)
    
    # Déterminer le device
    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        # Transformation inverse : x -> z
        z, logdet = model.inverse(grid_tensor)
        
        # Calculer la densité selon la formule de changement de variables
        # p_x(x) = p_z(z) * exp(logdet)
        # où p_z(z) = (2π)^(-d/2) * exp(-0.5 * ||z||²)
        log_p_z = -0.5 * torch.sum(z ** 2, dim=-1) - np.log(2 * np.pi)
        log_p_x = log_p_z + logdet
        density = torch.exp(log_p_x)
    
    model.train()
    
    # Reshaper en grille
    density_grid = density.cpu().numpy().reshape(resolution, resolution)
    
    return X_grid, Y_grid, density_grid


def plot_density_heatmap(X_grid, Y_grid, density, title="Densité apprise",
                         xlim=None, ylim=None, save_path=None):
    """
    Visualise la densité sous forme de heatmap.
    
    Args:
        X_grid (numpy.ndarray): Grille de coordonnées X
        Y_grid (numpy.ndarray): Grille de coordonnées Y
        density (numpy.ndarray): Densité sur la grille
        title (str): Titre de la figure
        xlim (tuple, optional): Limites en x pour l'affichage
        ylim (tuple, optional): Limites en y pour l'affichage
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(8, 8))
    
    # Normaliser la densité pour l'affichage
    density_normalized = density / density.max()
    
    # Créer la heatmap
    im = plt.imshow(density_normalized.T, origin='lower', 
                   extent=[X_grid.min(), X_grid.max(), 
                          Y_grid.min(), Y_grid.max()],
                   cmap='viridis', aspect='equal', interpolation='bilinear')
    
    plt.colorbar(im, label='Densité normalisée')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier de destination s'il n'existe pas
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Si le chemin contient un dossier
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()


def plot_density_comparison(true_density, learned_density, X_grid, Y_grid,
                            title="Comparaison des densités", save_path=None):
    """
    Compare la vraie densité et la densité apprise côte à côte.
    
    Args:
        true_density (numpy.ndarray): Vraie densité sur la grille (si disponible)
        learned_density (numpy.ndarray): Densité apprise sur la grille
        X_grid (numpy.ndarray): Grille de coordonnées X
        Y_grid (numpy.ndarray): Grille de coordonnées Y
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5))
    
    # Normaliser les densités
    if true_density is not None:
        true_density_norm = true_density / true_density.max()
    learned_density_norm = learned_density / learned_density.max()
    
    # Plot de la vraie densité
    if true_density is not None:
        ax = axes[0]
        im = ax.imshow(true_density_norm.T, origin='lower',
                      extent=[X_grid.min(), X_grid.max(), 
                             Y_grid.min(), Y_grid.max()],
                      cmap='viridis', aspect='equal', interpolation='bilinear')
        ax.set_title("Vraie densité")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        plt.colorbar(im, ax=ax, label='Densité normalisée')
    else:
        axes[0].text(0.5, 0.5, 'Vraie densité\nnon disponible', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Vraie densité")
    
    # Plot de la densité apprise
    ax = axes[1]
    im = ax.imshow(learned_density_norm.T, origin='lower',
                  extent=[X_grid.min(), X_grid.max(), 
                         Y_grid.min(), Y_grid.max()],
                  cmap='viridis', aspect='equal', interpolation='bilinear')
    ax.set_title("Densité apprise")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.colorbar(im, ax=ax, label='Densité normalisée')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        # Créer le dossier de destination s'il n'existe pas
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Si le chemin contient un dossier
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()


def plot_density_3d(X_grid, Y_grid, density, title="Densité 3D", save_path=None):
    """
    Visualise la densité en 3D (optionnel).
    
    Args:
        X_grid (numpy.ndarray): Grille de coordonnées X
        Y_grid (numpy.ndarray): Grille de coordonnées Y
        density (numpy.ndarray): Densité sur la grille
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normaliser la densité
    density_norm = density / density.max()
    
    # Créer la surface 3D
    surf = ax.plot_surface(X_grid, Y_grid, density_norm, cmap='viridis',
                          alpha=0.9, linewidth=0, antialiased=True)
    
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("Densité")
    ax.set_title(title)
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_path:
        # Créer le dossier de destination s'il n'existe pas
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Si le chemin contient un dossier
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.show()

