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
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X_grid, Y_grid = np.meshgrid(x, y)

    grid_points = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float64)

    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)

    model.eval()
    with torch.no_grad():
        z, logdet = model.inverse(grid_tensor)

        log_p_z = -0.5 * torch.sum(z**2, dim=-1) - np.log(2 * np.pi)
        log_p_x = log_p_z + logdet
        density = torch.exp(log_p_x)

    model.train()

    density_grid = density.cpu().numpy().reshape(resolution, resolution)
    
    # Vérifications et nettoyage
    density_grid = np.nan_to_num(density_grid, nan=0.0, posinf=0.0, neginf=0.0)
    density_grid = np.clip(density_grid, 0, None)  # S'assurer que toutes les valeurs sont >= 0

    return X_grid, Y_grid, density_grid


def plot_density_heatmap(
    X_grid,
    Y_grid,
    density,
    title="Densité apprise",
    xlim=None,
    ylim=None,
    save_path=None,
):
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
    # Nettoyage des valeurs invalides
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.clip(density, 0, None)
    
    # Vérifications
    density_max = np.max(density)
    density_min = np.min(density)
    
    print(f"Debug densité: min={density_min:.2e}, max={density_max:.2e}, mean={np.mean(density):.2e}")
    
    if density_max <= 0 or np.isclose(density_max, 0):
        print("Warning: La densité maximale est zéro ou très proche de zéro. Le graphique sera vide.")
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, "Densité vide ou invalide", 
                ha="center", va="center", transform=plt.gca().transAxes)
        plt.title(title)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        return
    
    # Normalisation robuste
    density_normalized = density / density_max
    
    plt.figure(figsize=(8, 8))
    
    # Utiliser contourf pour une meilleure visualisation
    contour = plt.contourf(
        X_grid,
        Y_grid,
        density_normalized,
        levels=20,
        cmap="viridis",
        extend='both'
    )
    
    plt.colorbar(contour, label="Densité normalisée")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)
    plt.axis('equal')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout() failed ({type(e).__name__}). Continuing anyway.")

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure sauvegardée dans {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure ({type(e).__name__}). Continuing anyway.")

    plt.show()
    plt.close()  # Close figure to prevent Jupyter rendering issues


def plot_density_comparison(
    true_density,
    learned_density,
    X_grid,
    Y_grid,
    title="Comparaison des densités",
    save_path=None,
):
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

    # Nettoyage et normalisation robuste
    if true_density is not None:
        true_density = np.nan_to_num(true_density, nan=0.0, posinf=0.0, neginf=0.0)
        true_density = np.clip(true_density, 0, None)
        true_max = np.max(true_density)
        if true_max > 0:
            true_density_norm = true_density / true_max
        else:
            true_density_norm = true_density
    
    learned_density = np.nan_to_num(learned_density, nan=0.0, posinf=0.0, neginf=0.0)
    learned_density = np.clip(learned_density, 0, None)
    learned_max = np.max(learned_density)
    if learned_max > 0:
        learned_density_norm = learned_density / learned_max
    else:
        learned_density_norm = learned_density

    if true_density is not None:
        ax = axes[0]
        if true_max > 0:
            contour = ax.contourf(
                X_grid,
                Y_grid,
                true_density_norm,
                levels=20,
                cmap="viridis",
                extend='both'
            )
            plt.colorbar(contour, ax=ax, label="Densité normalisée")
        else:
            ax.text(0.5, 0.5, "Densité vide", ha="center", va="center", 
                   transform=ax.transAxes)
        ax.set_title("Vraie densité")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.axis('equal')
    else:
        axes[0].text(
            0.5,
            0.5,
            "Vraie densité\nnon disponible",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Vraie densité")

    ax = axes[1]
    if learned_max > 0:
        contour = ax.contourf(
            X_grid,
            Y_grid,
            learned_density_norm,
            levels=20,
            cmap="viridis",
            extend='both'
        )
        plt.colorbar(contour, ax=ax, label="Densité normalisée")
    else:
        ax.text(0.5, 0.5, "Densité vide", ha="center", va="center", 
               transform=ax.transAxes)
    ax.set_title("Densité apprise")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.axis('equal')

    plt.suptitle(title, fontsize=14, y=1.02)
    
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout() failed ({type(e).__name__}). Using subplots_adjust instead.")
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure sauvegardée dans {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure ({type(e).__name__}). Continuing anyway.")

    plt.show()
    plt.close()  # Close figure to prevent Jupyter rendering issues


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

    # Nettoyage
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.clip(density, 0, None)
    
    density_max = np.max(density)
    if density_max <= 0:
        print("Warning: La densité maximale est zéro. Impossible de visualiser en 3D.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    density_norm = density / density_max

    surf = ax.plot_surface(
        X_grid,
        Y_grid,
        density_norm,
        cmap="viridis",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("Densité")
    ax.set_title(title)

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure sauvegardée dans {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure ({type(e).__name__}). Continuing anyway.")

    plt.show()
    plt.close()  # Close figure to prevent Jupyter rendering issues
