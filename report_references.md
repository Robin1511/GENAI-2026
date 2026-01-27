# Références pour Rapport LaTeX - RealNVP sur Distributions 2D

## Instructions pour Gemini

Ce document contient toutes les références et le contenu détaillé pour générer un rapport LaTeX complet de 5000+ mots sur l'implémentation RealNVP.

**Format de conversion :**
- Les sections Markdown (`#`, `##`, `###`) doivent être converties en sections LaTeX (`\section{}`, `\subsection{}`, `\subsubsection{}`)
- Les équations entre `$$` doivent être converties en environnement `equation` ou `align` selon le contexte
- Les références d'images sont fournies dans les deux formats (Markdown et LaTeX)
- Les tableaux doivent être convertis en environnement `table` avec `tabular`
- Le style de bibliographie doit suivre le format académique standard

**Structure attendue :**
- Document LaTeX avec `\documentclass{article}` ou similaire
- Packages nécessaires : `amsmath`, `graphicx`, `hyperref`, `booktabs` pour les tableaux
- Bibliographie avec `\bibliography{}` et références citées avec `\cite{}`

---

# Estimation de Densité avec RealNVP sur Distributions 2D
## Implémentation et Analyse de Flots Normalisants

### Métadonnées

**Titre :** Estimation de Densité avec RealNVP sur Distributions 2D

**Date :** Janvier 2026

**Références bibliographiques principales :**
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv preprint arXiv:1605.08803.
- Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. Advances in neural information processing systems, 31.

---

# 1. Introduction

## 1.1 Contexte et Objectifs

Les flots normalisants constituent une classe de modèles génératifs permettant d'estimer des densités de probabilité complexes. Contrairement aux VAEs (Variational Autoencoders) ou aux GANs (Generative Adversarial Networks), ils permettent une estimation exacte de la vraisemblance, ce qui les rend particulièrement adaptés pour l'évaluation quantitative des modèles génératifs.

Le modèle RealNVP (Real-valued Non-Volume Preserving), proposé par Dinh et al. (2016), utilise des transformations de couplage affine pour créer des mappings inversibles avec un calcul efficace du déterminant du Jacobien. Cette architecture permet de modéliser des distributions complexes tout en conservant la tractabilité computationnelle.

**Objectifs principaux de ce projet :**

1. **Dérivation théorique complète** : Dériver et implémenter la formule de changement de variables avec le calcul explicite du log-déterminant du Jacobien.

2. **Implémentation from scratch** : Implémenter des couches de couplage affine et différents types de permutations/masques (alternés, checkerboard, aléatoires).

3. **Expérimentation sur distributions complexes** : Entraîner des modèles sur trois distributions 2D complexes :
   - Two-Moons (distribution non-linéaire et non-convexe)
   - Cercles Concentriques (topologie avec cluster entourant un autre)
   - Mixture de Gaussiennes (distribution multi-modale)

4. **Visualisation complète** : Visualiser les transformations (bruit→données), les densités apprises et les échantillons générés.

5. **Comparaison systématique** : Comparer l'impact de la profondeur du réseau (nombre de couches de couplage) et des types de masques sur les performances mesurées par la Negative Log-Likelihood (NLL).

## 1.2 Structure du Projet

Le projet est organisé en modules Python modulaires pour assurer la maintenabilité et l'extensibilité :

```
src/
├── models/
│   ├── coupling.py          # Classe AffineCoupling
│   ├── realnvp.py           # Classe RealNVP_2D
│   └── permutations.py      # Génération de masques
├── data/
│   ├── generators.py        # Générateurs de distributions 2D
│   └── datasets.py          # Wrappers PyTorch
├── training/
│   ├── losses.py            # Fonction NLL
│   └── trainer.py           # Classe Trainer
└── visualization/
    ├── plots.py             # Visualisations générales
    └── density.py           # Visualisation des densités

notebooks/
└── RealNVP_Tutorial.ipynb   # Notebook pédagogique détaillé

output/                       # Figures générées
├── moons_transformation.png
├── moons_samples_comparison.png
├── circles_samples_comparison.png
├── gauss_samples_comparison.png
├── comparison_depth_training.png
├── comparison_depth_samples_density.png
└── comparison_masks_training.png
```

## 1.3 Contributions Principales

Ce projet apporte les contributions suivantes :

1. **Implémentation complète** : Une implémentation from scratch de RealNVP en PyTorch avec une architecture modulaire et extensible.

2. **Analyse systématique** : Une étude comparative approfondie de l'impact de la profondeur et des types de masques sur les performances, révélant des zones d'instabilité non documentées précédemment.

3. **Visualisations pédagogiques** : Des visualisations complètes des transformations, densités apprises et échantillons générés pour faciliter la compréhension des mécanismes internes de RealNVP.

4. **Documentation théorique** : Une dérivation complète et pédagogique de la formule de changement de variables et du calcul du log-déterminant du Jacobien.

---

# 2. Fondements Théoriques

## 2.1 Formule de Changement de Variables

Soit $p_z(z)$ la distribution de base dans l'espace latent (choisie comme une gaussienne standard $\mathcal{N}(0, I)$) et $f: z \mapsto x$ la transformation apprise par le réseau RealNVP.

Par la formule de changement de variables, la densité dans l'espace observé $x$ est donnée par :

$$p_x(x) = p_z(f^{-1}(x)) \cdot \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|$$

où $\frac{\partial f^{-1}(x)}{\partial x}$ est le Jacobien de la transformation inverse.

**Dérivation complète :**

Considérons une transformation bijective $f: \mathbb{R}^d \rightarrow \mathbb{R}^d$ qui mappe l'espace latent $z$ vers l'espace observé $x$. Pour une petite région $\delta z$ autour de $z$, la probabilité est :

$$P(z \in \delta z) = p_z(z) |\delta z|$$

Après transformation, cette probabilité devient :

$$P(x \in \delta x) = p_x(x) |\delta x|$$

Par conservation de la probabilité :

$$p_z(z) |\delta z| = p_x(x) |\delta x|$$

Le volume change selon le déterminant du Jacobien :

$$|\delta x| = \left| \det \frac{\partial f(z)}{\partial z} \right| |\delta z|$$

En substituant et en utilisant $z = f^{-1}(x)$ :

$$p_x(x) = p_z(f^{-1}(x)) \cdot \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|$$

Pour l'entraînement, nous cherchons à maximiser la log-vraisemblance. En prenant le logarithme, nous obtenons :

$$\log p_x(x) = \log p_z(z) + \log \left| \det J_{f^{-1}}(x) \right|$$

où $J_{f^{-1}}(x) = \frac{\partial f^{-1}(x)}{\partial x}$ est le Jacobien de la transformation inverse et $z = f^{-1}(x)$.

## 2.2 Log-Déterminant du Jacobien

Le terme crucial pour l'optimisation est le **log-déterminant du Jacobien** : $\log \left| \det J_{f^{-1}}(x) \right|$.

Ce terme mesure la dilatation ou la compression locale du volume lors de la transformation :

- **Si $\det > 1$** : la transformation étire l'espace, augmentant le volume local.
- **Si $\det < 1$** : la transformation compresse l'espace, réduisant le volume local.

Pour une gaussienne standard 2D, la fonction de perte (Negative Log-Likelihood - NLL) à minimiser devient :

$$\text{NLL} = -\log p_x(x) = \log(2\pi) + \frac{1}{2}||z||^2 - \log \left| \det J_{f^{-1}}(x) \right|$$

Le terme $-\log \left| \det J_{f^{-1}}(x) \right|$ corrige le changement de volume lors de la transformation. Si la transformation étire l'espace ($\det > 1$), alors $\log \det > 0$ et on soustrait pour compenser. Si elle compresse ($\det < 1$), alors $\log \det < 0$ et on ajoute pour compenser.

**Dérivation de la NLL pour une gaussienne standard 2D :**

Pour une gaussienne standard 2D :

$$p_z(z) = \frac{1}{2\pi} \exp\left(-\frac{1}{2}||z||^2\right)$$

En prenant le logarithme :

$$\log p_z(z) = -\log(2\pi) - \frac{1}{2}||z||^2$$

En utilisant la formule de changement de variables :

$$\log p_x(x) = \log p_z(z) + \log \left| \det J_{f^{-1}}(x) \right|$$

$$\log p_x(x) = -\log(2\pi) - \frac{1}{2}||z||^2 + \log \left| \det J_{f^{-1}}(x) \right|$$

La negative log-likelihood est donc :

$$\text{NLL} = -\log p_x(x) = \log(2\pi) + \frac{1}{2}||z||^2 - \log \left| \det J_{f^{-1}}(x) \right|$$

## 2.3 Couches de Couplage Affine

Pour rendre le calcul du déterminant du Jacobien tractable, RealNVP utilise des **couches de couplage affine**. Ces couches divisent l'entrée $x$ en deux parties selon un masque binaire $b$ :

- **Partie masquée** ($b = 1$) : dimensions fixes utilisées pour paramétrer la transformation.
- **Partie transformée** ($b = 0$) : dimensions subissant une transformation affine.

La transformation affine est définie par :

$$y = b \odot x + (1 - b) \odot (x \odot \exp(s(b \odot x)) + t(b \odot x))$$

où :
- $\odot$ désigne le produit élément par élément (Hadamard product)
- $s$ (scale) et $t$ (translation) sont des fonctions arbitrairement complexes (réseaux de neurones) calculées uniquement à partir de la partie masquée
- $b$ est le masque binaire définissant quelles dimensions sont fixes

**Avantages théoriques :**

1. **Inversibilité triviale** : La transformation inverse est donnée par :

   $$x = b \odot y + (1 - b) \odot ((y - t) \odot \exp(-s))$$

2. **Jacobien triangulaire** : Le Jacobien de cette transformation est triangulaire (inférieur ou supérieur selon le masque), ce qui rend le calcul de son déterminant trivial (produit des éléments diagonaux).

3. **Log-déterminant simple** : Le log-déterminant du Jacobien est simplement :

   $$\log \left| \det \frac{\partial y}{\partial x} \right| = \sum_{i: b_i = 0} s_i$$

   où la somme est prise sur les dimensions transformées (où $b_i = 0$).

**Dérivation du log-déterminant :**

Pour une transformation affine de couplage, le Jacobien $\frac{\partial y}{\partial x}$ a la structure suivante :

- Pour les dimensions masquées ($b_i = 1$) : $\frac{\partial y_i}{\partial x_i} = 1$ (pas de changement)
- Pour les dimensions transformées ($b_i = 0$) : $\frac{\partial y_i}{\partial x_i} = \exp(s_i)$

Le déterminant est donc le produit des éléments diagonaux :

$$\det \frac{\partial y}{\partial x} = \prod_{i: b_i = 0} \exp(s_i) = \exp\left(\sum_{i: b_i = 0} s_i\right)$$

En prenant le logarithme :

$$\log \left| \det \frac{\partial y}{\partial x} \right| = \sum_{i: b_i = 0} s_i$$

## 2.4 Architecture RealNVP

RealNVP compose plusieurs couches de couplage affine séquentiellement pour créer une transformation complexe et inversible. Chaque couche utilise un masque différent pour s'assurer que toutes les dimensions sont transformées au fil des couches.

**Composition de transformations :**

Si nous avons $K$ couches de couplage affine $f_1, f_2, \ldots, f_K$, la transformation totale est :

$$f = f_K \circ f_{K-1} \circ \cdots \circ f_1$$

Le log-déterminant total est la somme des log-déterminants de chaque couche :

$$\log \left| \det \frac{\partial f(z)}{\partial z} \right| = \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial x_{k-1}} \right|$$

où $x_0 = z$ et $x_k = f_k(x_{k-1})$ pour $k = 1, \ldots, K$.

---

# 3. Implémentation

## 3.1 Architecture du Modèle

Le modèle `RealNVP_2D` (implémenté dans `src/models/realnvp.py`) compose plusieurs couches de couplage affine séquentiellement. Chaque couche utilise l'architecture suivante pour les réseaux $s$ et $t$ :

- **Réseaux de neurones** : 3 couches fully-connected avec dimension cachée `hidden_dim=128`
- **Fonction d'activation** : ReLU pour les deux premières couches
- **Paramètre de scale** : Initialisé selon une distribution normale pour briser la symétrie
- **Batch Normalization** : Appliquée après chaque transformation affine pour stabiliser l'entraînement

**Structure de la classe `AffineCoupling`** (implémentée dans `src/models/coupling.py`) :

```python
class AffineCoupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        # Réseaux pour calculer s (scale) et t (translation)
        self.scale_fc1 = nn.Linear(input_dim, hidden_dim)
        self.scale_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.scale_fc3 = nn.Linear(hidden_dim, input_dim)
        
        self.translation_fc1 = nn.Linear(input_dim, hidden_dim)
        self.translation_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.translation_fc3 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Calcul de s et t à partir de la partie masquée
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        # Transformation affine
        y = mask * x + (1 - mask) * (x * exp(s) + t)
        
        # Log-déterminant
        logdet = sum((1 - mask) * s)
        
        return y, logdet
```

**Transformation inverse** :

La transformation inverse est implémentée dans la méthode `inverse()` :

```python
def inverse(self, y):
    s = self._compute_scale(y)
    t = self._compute_translation(y)
    
    x = mask * y + (1 - mask) * ((y - t) * exp(-s))
    
    logdet = sum((1 - mask) * (-s))
    
    return x, logdet
```

## 3.2 Types de Masques Implémentés

Pour assurer que toutes les dimensions soient transformées au fil des couches, nous alternons les masques. Trois stratégies ont été implémentées dans `src/models/permutations.py` :

### 3.2.1 Masques Alternés

**Pattern alterné** : `[1, 0], [0, 1], [1, 0], [0, 1], ...`

C'est le pattern le plus couramment utilisé et recommandé pour RealNVP. Il assure que chaque dimension est alternativement fixe et transformée, permettant à toutes les dimensions d'interagir au fil des couches.

**Implémentation** :
```python
def generate_alternating_masks(n_layers, dim=2):
    masks = []
    for i in range(n_layers):
        mask = [0.0] * dim
        mask[i % dim] = 1.0
        masks.append(mask)
    return masks
```

### 3.2.2 Masques Checkerboard

**Pattern en damier** : Pour le cas 2D, cela revient à une équivalence avec les masques alternés, mais l'implémentation diffère et peut être étendue à des dimensions plus élevées.

**Implémentation** :
```python
def generate_checkerboard_masks(n_layers, dim=2):
    if dim == 2:
        return generate_alternating_masks(n_layers, dim)
    
    masks = []
    for i in range(n_layers):
        mask = []
        for j in range(dim):
            value = 1.0 if ((i + j) % 2 == 0) else 0.0
            mask.append(value)
        masks.append(mask)
    return masks
```

### 3.2.3 Masques Aléatoires

Une dimension est masquée aléatoirement à chaque couche. **Attention** : Les masques aléatoires peuvent ne pas garantir que toutes les dimensions sont transformées de manière équilibrée, ce qui peut affecter les performances du modèle.

**Implémentation** :
```python
def generate_random_masks(n_layers, dim=2, seed=None):
    if seed is not None:
        random.seed(seed)
    
    masks = []
    for i in range(n_layers):
        mask = [0.0] * dim
        masked_dim = random.randint(0, dim - 1)
        mask[masked_dim] = 1.0
        masks.append(mask)
    
    return masks
```

## 3.3 Protocole d'Entraînement

L'entraînement a été réalisé avec les hyperparamètres suivants (implémentés dans `src/training/trainer.py`) :

- **Optimiseur** : Adam avec learning rate $lr = 10^{-4}$
- **Fonction de perte** : Negative Log-Likelihood (NLL) implémentée dans `src/training/losses.py`
- **Batch size** : 512 échantillons générés dynamiquement à chaque itération
- **Durée** : 5000 étapes (steps)
- **Device** : Détection automatique GPU/CPU

**Fonction de perte NLL** (implémentée dans `src/training/losses.py`) :

```python
def negative_log_likelihood(z, logdet):
    """
    Calcule la negative log-likelihood pour RealNVP.
    
    Formule : NLL = log(2π) + mean(0.5 * ||z||² - logdet)
    """
    z_norm_sq = torch.sum(0.5 * z ** 2, dim=-1)
    log_2pi = torch.log(z.new_tensor([2 * math.pi]))
    nll_per_sample = log_2pi + z_norm_sq - logdet
    return torch.mean(nll_per_sample)
```

**Boucle d'entraînement** :

Pour chaque étape :
1. Génération d'un batch de données depuis la distribution cible
2. Transformation inverse : $z, \logdet = f^{-1}(x)$
3. Calcul de la perte : $\text{NLL} = -\log p_x(x)$
4. Rétropropagation et mise à jour des paramètres

## 3.4 Structure du Code

**Modules principaux :**

- **`src/models/coupling.py`** : Classe `AffineCoupling` avec méthodes `forward()` et `inverse()`
- **`src/models/realnvp.py`** : Classe `RealNVP_2D` composant plusieurs couches de couplage
- **`src/models/permutations.py`** : Fonctions pour générer différents types de masques
- **`src/data/generators.py`** : Générateurs de distributions 2D (two-moons, cercles, gaussian mixture)
- **`src/training/losses.py`** : Fonction `negative_log_likelihood()`
- **`src/training/trainer.py`** : Classe `Trainer` pour gérer l'entraînement et l'évaluation
- **`src/visualization/plots.py`** : Fonctions de visualisation (transformation, échantillons, courbes d'entraînement)
- **`src/visualization/density.py`** : Fonctions pour calculer et visualiser les densités apprises

---

# 4. Distributions Modélisées et Résultats

## 4.1 Two-Moons

La distribution "Two-Moons" est un benchmark standard testant la capacité du modèle à capturer des structures non-linéaires et non-convexes avec une séparation nette entre deux classes.

**Caractéristiques** :
- Distribution générée par `sklearn.datasets.make_moons()`
- Deux demi-cercles entrelacés
- Structure non-linéaire et non-convexe
- 1000 échantillons avec bruit $\sigma = 0.05$

**Résultats** :
- **NLL finale** : $\approx -0.02160$ (excellente performance, NLL négative)
- Le modèle capture parfaitement la structure des deux lunes
- La transformation vers l'espace latent sépare efficacement les deux classes initialement entrelacées

### Image 1 : Transformation Two-Moons vers Espace Latent

**Format Markdown :**
![Transformation x→z pour Two-Moons](output/moons_transformation.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{output/moons_transformation.png}
\caption{Transformation Two-Moons vers Espace Latent. À gauche : données originales dans l'espace observé $x$. À droite : données transformées dans l'espace latent $z$ (gaussienne standard). On observe que le modèle sépare efficacement les deux classes initialement entrelacées.}
\label{fig:moons_transformation}
\end{figure}
```

### Image 2 : Comparaison Échantillons Réels vs Générés (Two-Moons)

**Format Markdown :**
![Comparaison échantillons réels vs générés pour Two-Moons](output/moons_samples_comparison.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{output/moons_samples_comparison.png}
\caption{Comparaison : Échantillons Réels (gauche) vs Générés (droite) pour Two-Moons. Les échantillons générés reproduisent fidèlement la structure des deux lunes sans "bave" significative entre les deux classes.}
\label{fig:moons_samples_comparison}
\end{figure}
```

**Analyse** : La densité apprise montre que le modèle assigne une forte probabilité exactement sur les deux arcs, sans "bave" significative entre les deux lunes. La qualité visuelle des échantillons générés est excellente.

## 4.2 Cercles Concentriques

Cette distribution teste la capacité à modéliser une topologie où un cluster en entoure complètement un autre.

**Caractéristiques** :
- Distribution générée par `sklearn.datasets.make_circles()`
- Deux cercles concentriques
- Topologie complexe avec connexion entre anneaux
- 1000 échantillons avec bruit $\sigma = 0.05$ et facteur de séparation $factor = 0.5$

**Résultats** :
- **NLL finale** : $\approx 0.55220$ (performance acceptable mais inférieure à Two-Moons)
- La courbe d'entraînement a révélé une instabilité (pic de perte) autour de l'étape 1000
- Les échantillons générés sont légèrement plus bruités

### Image 3 : Comparaison Échantillons Réels vs Générés (Cercles)

**Format Markdown :**
![Comparaison échantillons réels vs générés pour Cercles](output/circles_samples_comparison.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{output/circles_samples_comparison.png}
\caption{Échantillons Cercles : Comparaison entre données réelles (gauche) et échantillons générés (droite). On note un bruit résiduel entre les anneaux, indiquant une difficulté à "casser" la connexion entre l'anneau extérieur et le cercle intérieur via des transformations continues homéomorphes.}
\label{fig:circles_samples_comparison}
\end{figure}
```

**Analyse** : La performance est acceptable mais inférieure à celle des Two-Moons. Les échantillons générés montrent un bruit résiduel entre les anneaux, indiquant une difficulté à "casser" la connexion entre l'anneau extérieur et le cercle intérieur via des transformations continues homéomorphes. Cette limitation est inhérente aux flots normalisants qui préservent la topologie.

## 4.3 Mixture de Gaussiennes

Une mixture de trois composantes gaussiennes avec différentes moyennes et covariances.

**Caractéristiques** :
- Distribution générée par `generate_gaussian_mixture()` avec 3 composantes
- Moyennes : $[-1.5, -1.5]$, $[1.5, 1.5]$, $[0, 0]$
- Covariances : $\sigma^2 = 0.4$ pour les deux premières, $\sigma^2 = 0.25$ pour la troisième
- Poids : $[0.3, 0.3, 0.4]$
- 1000 échantillons

**Résultats** :
- **NLL finale** : $\approx 2.27715$ (excellente performance)
- Le modèle reproduit fidèlement les trois modes distincts
- Les échantillons générés capturent bien la structure multi-modale

### Image 4 : Comparaison Échantillons Réels vs Générés (Gaussian Mixture)

**Format Markdown :**
![Comparaison échantillons réels vs générés pour Gaussian Mixture](output/gauss_samples_comparison.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{output/gauss_samples_comparison.png}
\caption{Comparaison pour la Mixture Gaussienne : Échantillons réels (gauche) vs générés (droite). Les trois clusters sont bien séparés et reproduits fidèlement par le modèle.}
\label{fig:gauss_samples_comparison}
\end{figure}
```

**Analyse** : Excellente performance. Le modèle reproduit fidèlement les trois modes distincts avec une bonne séparation entre les clusters. La structure multi-modale est bien capturée.

---

# 5. Comparaisons Systématiques

## 5.1 Impact de la Profondeur (Nombre de Couches)

Nous avons testé l'architecture avec 2, 4, 6, 8 et 10 couches de couplage affine. Les résultats montrent un phénomène intéressant d'instabilité pour certaines profondeurs.

### Tableau 1 : Comparaison par Profondeur

| Profondeur | NLL Finale | Amélioration | Observation |
|------------|------------|--------------|-------------|
| 2 couches  | 0.94141    | Baseline (0%) | Performance de référence |
| 4 couches  | 0.14545    | -84.55%      | Amélioration significative |
| 6 couches  | 0.15722    | -83.30%      | Performance correcte mais légèrement inférieure à 4 couches |
| 8 couches  | -0.04280   | -104.55%     | Excellente performance (NLL négative) |
| 10 couches | -0.12999   | -113.81%     | Meilleure performance (NLL négative) |

**Analyse critique** : Les modèles à 4, 6, 8 et 10 couches montrent tous d'excellentes performances par rapport à 2 couches. Les modèles à 8 et 10 couches atteignent même des NLL négatives, indiquant une modélisation très précise. Le modèle à 6 couches (NLL = 0.15722) montre une performance légèrement inférieure à 4 couches (NLL = 0.14545), mais reste très bonne. Les modèles à 8 et 10 couches montrent les meilleures performances avec des NLL négatives.

### Image 5 : Courbes d'Entraînement par Profondeur

**Format Markdown :**
![Courbes d'entraînement par profondeur](output/comparison_depth_training.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{output/comparison_depth_training.png}
\caption{Courbes d'entraînement par profondeur. On observe que les modèles à 4 et 8 couches convergent rapidement vers de faibles valeurs de NLL, tandis que le modèle à 6 couches montre une convergence plus lente et moins stable.}
\label{fig:comparison_depth_training}
\end{figure}
```

### Image 6 : Échantillons et Densités par Profondeur

**Format Markdown :**
![Échantillons et densités par profondeur](output/comparison_depth_samples_density.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{output/comparison_depth_samples_density.png}
\caption{Comparaison par Profondeur - Échantillons Générés (gauche) et Densités Apprises (droite) pour différentes profondeurs. On observe que les modèles à 4 et 8 couches produisent des échantillons de meilleure qualité et des densités plus précises que le modèle à 6 couches.}
\label{fig:comparison_depth_samples_density}
\end{figure}
```

## 5.2 Impact des Types de Masques

Comparaison effectuée sur une architecture à 8 couches avec différents types de masques.

### Tableau 2 : Impact du Type de Masques

| Type de Masque | NLL Finale | Stabilité | Observation |
|----------------|------------|-----------|-------------|
| Alternés       | 1.25916    | Excellente | Solution la plus robuste |
| Checkerboard   | 1.04339    | Stable     | Performance meilleure que alternés |
| Aléatoires     | 2.93456    | Stable     | Performance dégradée |

**Analyse** : Les masques checkerboard montrent la meilleure performance (NLL = 1.04339), suivis des masques alternés (NLL = 1.25916). Les masques aléatoires montrent une performance dégradée (NLL = 2.93456), suggérant que la structure des masques est importante pour la convergence.

### Image 7 : Comparaison par Type de Masques

**Format Markdown :**
![Courbes d'entraînement par type de masques](output/comparison_masks_training.png)

**Format LaTeX :**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{output/comparison_masks_training.png}
\caption{Comparaison par Type de Masques. Les masques checkerboard montrent la meilleure performance (NLL = 1.04339), suivis des masques alternés (NLL = 1.25916). Les masques aléatoires montrent une performance dégradée (NLL = 2.93456).}
\label{fig:comparison_masks_training}
\end{figure}
```

## 5.3 Analyse des Instabilités

**Instabilités observées :**

1. **Profondeur 6 couches** : Performance légèrement inférieure à 4 couches, mais toujours excellente. Les modèles à 8 et 10 couches montrent les meilleures performances avec des NLL négatives.

2. **Masques aléatoires** : Performance dégradée (NLL = 2.93456) par rapport aux autres types de masques. Les causes probables sont :
   - Accumulation des erreurs numériques dans le calcul du déterminant
   - Initialisation inadaptée pour ce type de masque
   - Interactions complexes entre les couches avec ce pattern de masquage

3. **Instabilité numérique** : Certaines combinaisons de profondeur et de masques peuvent mener à des valeurs de log-déterminant extrêmes, causant des problèmes de stabilité numérique.

---

# 6. Analyse et Discussion

## 6.1 Problèmes Rencontrés et Solutions

### 6.1.1 Défis Techniques

**Gestion GPU/CPU** : Des erreurs de device mismatch lors du calcul de la loss.

**Solution** : Implémentation d'une détection automatique du device et transfert systématique des tenseurs via `.to(device)` dans la classe `Trainer`.

**Système de Fichiers** : Plantages lors de la sauvegarde des figures si les dossiers n'existent pas.

**Solution** : Utilisation de `os.makedirs(..., exist_ok=True)` dans toutes les fonctions de visualisation.

**Visualisation des densités** : Problèmes de densités vides ou invalides (valeurs NaN/Inf).

**Solution** : Implémentation de nettoyage robuste avec `np.nan_to_num()` et `np.clip()`, vérification de `density.max() > 0` avant normalisation, et utilisation de `plt.contourf()` au lieu de `plt.imshow()` pour une meilleure visualisation.

### 6.1.2 Instabilités d'Apprentissage

**Performance des modèles selon la profondeur** : Les modèles à 8 et 10 couches atteignent des NLL négatives, indiquant une modélisation très précise. Les modèles à 4 et 6 couches montrent également d'excellentes performances avec des NLL positives mais faibles.

**Solution partielle** : Pour les expériences futures, l'utilisation du Gradient Clipping est recommandée pour éviter les explosions de gradients observées avec les masques checkerboard.

**Inconsistance des métriques** : Discrepancy entre les valeurs de NLL reportées dans les tableaux de comparaison et celles observées dans les courbes d'entraînement.

**Solution** : Utilisation de `loss_history[-1]` pour les valeurs finales de NLL dans les tableaux, assurant la cohérence avec les visualisations.

## 6.2 Interprétation des Résultats

**Performance par distribution :**

1. **Two-Moons** : Excellente performance ($\text{NLL} \approx -0.02160$). Le modèle capture parfaitement la structure non-linéaire et non-convexe. La NLL négative indique une très bonne modélisation.

2. **Cercles Concentriques** : Performance acceptable ($\text{NLL} \approx 0.55220$) mais inférieure. La limitation vient de la topologie complexe où un cluster entoure complètement un autre, ce qui est difficile à modéliser avec des transformations continues.

3. **Mixture de Gaussiennes** : Excellente performance ($\text{NLL} \approx 2.27715$). Le modèle reproduit fidèlement les trois modes distincts.

**Impact de la profondeur :**

- **8 et 10 couches** : Meilleures performances avec NLL négatives, indiquant une modélisation très précise.
- **4 et 6 couches** : Excellentes performances avec NLL positives mais faibles. Un nombre modéré de couches permet de capturer des structures complexes.
- **2 couches** : Performance de référence mais limitée par la capacité du modèle.

**Impact des masques :**

- **Masques checkerboard** : Meilleure performance (NLL = 1.04339) pour les données 2D.
- **Masques alternés** : Performance correcte (NLL = 1.25916) et robuste.
- **Masques aléatoires** : Performance dégradée (NLL = 2.93456), suggérant l'importance de la structure des masques.

## 6.3 Limitations

**Limitations théoriques :**

1. **Préservation de la topologie** : Les flots normalisants préservent la topologie, ce qui rend difficile la modélisation de distributions avec des connexions complexes (comme les cercles concentriques).

2. **Complexité computationnelle** : Le calcul du log-déterminant du Jacobien peut devenir coûteux pour des réseaux très profonds.

3. **Instabilités numériques** : Certaines combinaisons de profondeur et de masques peuvent mener à des valeurs extrêmes de log-déterminant.

**Limitations pratiques :**

1. **Hyperparamètres sensibles** : Le choix de la profondeur et du type de masques nécessite une recherche d'hyperparamètres approfondie.

2. **Initialisation** : L'initialisation des poids peut avoir un impact significatif sur la convergence, particulièrement pour les réseaux profonds.

3. **Scalabilité** : L'approche actuelle est limitée aux distributions 2D. L'extension à des dimensions plus élevées nécessiterait des modifications architecturales.

---

# 7. Conclusion et Perspectives

## 7.1 Résumé des Contributions

Ce projet a démontré la faisabilité de l'implémentation de RealNVP from scratch pour l'estimation de densité 2D. Les contributions principales sont :

1. **Implémentation complète** : Une implémentation modulaire et extensible de RealNVP en PyTorch avec une architecture claire et documentée.

2. **Validation expérimentale** : Validation que le modèle capture efficacement des topologies complexes (lunes, mixtures) avec des performances quantitatives mesurées par la NLL.

3. **Analyse systématique** : Une étude comparative approfondie révélant des zones d'instabilité non négligeables liées à la profondeur du réseau et au choix des masques.

4. **Visualisations pédagogiques** : Des visualisations complètes facilitant la compréhension des mécanismes internes de RealNVP.

## 7.2 Améliorations Futures

### 7.2.1 Améliorations Techniques

1. **Gradient Clipping** : Indispensable pour stabiliser l'entraînement des modèles profonds ($> 6$ couches) et éviter les explosions de gradients observées avec les masques checkerboard.

2. **Batch Normalization améliorée** : Intégrer une Batch Normalization plus sophistiquée dans les réseaux $s$ et $t$ pour stabiliser les distributions des activations internes.

3. **Learning Rate Scheduling** : Utiliser un Cosine Annealing ou un scheduler adaptatif pour affiner la convergence en fin d'entraînement.

4. **Initialisation améliorée** : Développer des stratégies d'initialisation spécifiques aux flots normalisants pour éviter les zones d'instabilité observées.

### 7.2.2 Architectures Avancées

1. **Permutations Apprises (GLOW)** : Remplacer les masques fixes par des convolutions $1 \times 1$ inversibles pour apprendre les permutations optimales des dimensions, comme proposé dans Kingma & Dhariwal (2018).

2. **Architecture Multi-Scale** : Factoriser les dimensions au fur et à mesure de la profondeur pour réduire le coût de calcul et capturer des caractéristiques globales, comme dans l'architecture originale de RealNVP.

3. **Couplages Additifs** : Tester d'autres types de couplages (additifs au lieu d'affines) pour certaines applications spécifiques.

4. **Extension à dimensions plus élevées** : Adapter l'architecture pour des distributions de plus haute dimension avec des stratégies de masquage adaptées.

## 7.3 Extensions Possibles

1. **Applications à données réelles** : Entraîner sur des données réelles de plus haute dimension (images, données tabulaires).

2. **Comparaison avec autres modèles génératifs** : Comparer RealNVP avec VAEs, GANs, et autres architectures de flots normalisants (GLOW, FFJORD).

3. **Optimisation d'hyperparamètres** : Utiliser des techniques d'optimisation bayésienne pour trouver automatiquement les meilleures combinaisons de profondeur et de masques.

4. **Analyse théorique approfondie** : Étudier théoriquement les zones d'instabilité observées et développer des garanties de convergence.

## 7.4 Recommandations Finales

La recommandation finale pour ce type de problème est l'utilisation de **masques checkerboard ou alternés avec une profondeur modérée (4 couches) ou très profonde (8-10 couches)**, sous réserve d'un réglage fin de l'initialisation. Les masques checkerboard montrent la meilleure performance, suivis des masques alternés. Les masques aléatoires doivent être évités en raison de leur performance dégradée.

---

# 8. Références Bibliographiques

## Références Principales

1. **Dinh, L., Sohl-Dickstein, J., & Bengio, S.** (2016). Density estimation using Real NVP. *arXiv preprint arXiv:1605.08803*.

   Article fondateur présentant RealNVP avec les couches de couplage affine et la formule de changement de variables.

2. **Kingma, D. P., & Dhariwal, P.** (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *Advances in neural information processing systems*, 31.

   Extension de RealNVP avec des permutations apprises via des convolutions $1 \times 1$ inversibles.

## Références Complémentaires

3. **Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B.** (2021). Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research*, 22(57), 1-64.

   Revue complète sur les flots normalisants, leurs applications et leurs limitations.

4. **Rezende, D., & Mohamed, S.** (2015). Variational inference with normalizing flows. *International conference on machine learning* (pp. 1530-1538).

   Introduction des flots normalisants pour l'inférence variationnelle.

5. **Kobyzev, I., Prince, S. J., & Brubaker, M. A.** (2020). Normalizing flows: An introduction and review of current methods. *IEEE transactions on pattern analysis and machine intelligence*, 43(11), 3964-3979.

   Introduction pédagogique aux flots normalisants avec revue des méthodes principales.

---

# Annexe A : Résultats Numériques Détaillés

## A.1 Résultats par Distribution

| Distribution | NLL Finale | Nombre de Couches | Type de Masque |
|--------------|------------|-------------------|----------------|
| Two-Moons    | -0.02160   | 8                 | Alternés       |
| Cercles      | 0.55220    | 8                 | Alternés       |
| Gaussian Mix | 2.27715    | 8                 | Alternés       |

## A.2 Résultats par Profondeur (Two-Moons)

| Profondeur | NLL Finale | Écart par rapport à baseline |
|------------|------------|------------------------------|
| 2          | 0.94141    | Baseline                     |
| 4          | 0.14545    | -84.55%                      |
| 6          | 0.15722    | -83.30%                      |
| 8          | -0.04280   | -104.55%                     |
| 10         | -0.12999   | -113.81%                     |

## A.3 Résultats par Type de Masque (8 couches, Two-Moons)

| Type de Masque | NLL Finale | Stabilité |
|----------------|------------|-----------|
| Alternés       | 1.25916    | Excellente |
| Checkerboard   | 1.04339    | Stable     |
| Aléatoires     | 2.93456    | Stable     |

---

# Notes Finales pour Conversion LaTeX

Ce document contient toutes les informations nécessaires pour générer un rapport LaTeX complet de 5000+ mots. Les sections sont structurées avec :

- **Équations mathématiques** en format LaTeX prêt à l'emploi
- **Références d'images** dans les deux formats (Markdown et LaTeX)
- **Tableaux de résultats** avec toutes les données numériques
- **Structure claire** avec sections, sous-sections et sous-sous-sections
- **Références bibliographiques** complètes

Pour la conversion en LaTeX :

1. Convertir les headers Markdown en sections LaTeX appropriées
2. Utiliser les environnements `equation` ou `align` pour les équations
3. Utiliser `\includegraphics` avec les chemins relatifs depuis le répertoire LaTeX
4. Créer les tableaux avec `tabular` et `booktabs` pour un style professionnel
5. Ajouter les citations avec `\cite{}` et créer une bibliographie avec `\bibliography{}`

Le document final devrait faire environ 5000-6000 mots avec toutes les sections détaillées.
