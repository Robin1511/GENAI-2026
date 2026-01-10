# Projet RealNVP 2D - Estimation de Densité avec Flots Normalisants

Ce projet implémente RealNVP (Real-valued Non-Volume Preserving) pour modéliser des distributions 2D complexes à l'aide de flots normalisants.

## Structure du Projet

```
GENAI-2026/
├── src/
│   ├── models/          # Modèles RealNVP (couplage affine, RealNVP_2D)
│   ├── data/            # Générateurs de distributions 2D
│   ├── training/        # Entraînement et fonctions de perte
│   ├── visualization/   # Visualisation des résultats
│   └── utils/           # Configuration et utilitaires
├── notebooks/
│   └── RealNVP_Tutorial.ipynb  # Notebook principal très commenté
├── output/              # Figures générées
├── requirements.txt     # Dépendances Python
├── RAPPORT.md          # Rapport du projet
├── IMAGES_REFERENCE.md # Référence des images pour LaTeX
└── README.md           # Ce fichier
```

## Installation

1. Cloner ou télécharger le projet

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Vérifier l'installation :
```bash
python -c "import torch; import numpy; print('Installation réussie!')"
```

## Utilisation

### Notebook Principal

Le notebook `notebooks/RealNVP_Tutorial.ipynb` contient une implémentation complète et très commentée avec :

- **Introduction théorique** : Formule de changement de variables, log-déterminant du Jacobien
- **Implémentation** : Couches de couplage affine et modèle RealNVP
- **Génération de données** : Two-moons, cercles, mixtures gaussiennes
- **Entraînement** : Maximum de vraisemblance avec visualisation
- **Comparaisons** : Impact de la profondeur, nombre de couches, types de masques
- **Visualisation** : Densités apprises et échantillons générés

Pour lancer le notebook :
```bash
jupyter notebook notebooks/RealNVP_Tutorial.ipynb
```

### Utilisation en Python

```python
from src.models import RealNVP_2D, generate_alternating_masks
from src.data import generate_two_moons
from src.training import Trainer
import torch

# Créer un modèle
masks = generate_alternating_masks(n_layers=8, dim=2)
model = RealNVP_2D(masks, hidden_dim=128)

# Créer un trainer
trainer = Trainer(model, lr=0.0001)

# Générer des données
def data_generator():
    X, _ = generate_two_moons(n_samples=512)
    return torch.tensor(X, dtype=torch.float64)

# Entraîner
loss_history = trainer.train(data_generator, n_steps=5000)

# Échantillonner
samples = trainer.sample(n_samples=1000)
```

## Distributions Supportées

- **Two-Moons** : Distribution en forme de deux demi-cercles entrelacés
- **Cercles** : Distribution en cercles concentriques
- **Mixtures Gaussiennes** : Mélange de plusieurs distributions gaussiennes 2D

## Fonctionnalités

### Modèles
- `AffineCoupling` : Couche de couplage affine avec transformation inversible
- `RealNVP_2D` : Modèle complet composant plusieurs couches de couplage

### Entraînement
- Entraînement par maximum de vraisemblance
- Fonction de perte : Negative Log-Likelihood
- Support GPU/CPU automatique

### Visualisation
- Transformation x → z et z → x
- Courbes d'entraînement
- Comparaison échantillons réels vs générés
- Visualisation des densités apprises (heatmaps)

### Comparaisons
- Impact de la profondeur (nombre de couches)
- Impact du type de masques (alternés, checkerboard, aléatoires)
- Métriques : NLL, temps d'entraînement

## Configuration

Les paramètres par défaut sont définis dans `src/utils/config.py` :
- `hidden_dim` : 128
- `learning_rate` : 0.0001
- `n_steps` : 5000
- `batch_size` : 512

## Références

- **RealNVP** : Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv:1605.08803
- **GLOW** : Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1×1 convolutions. arXiv:1807.03039

## Auteur

Projet réalisé dans le cadre du module GENAI-2026.

## Licence

Ce projet est fourni à des fins éducatives.

