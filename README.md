# Mini Projet – Analyse des sentiments sur Tweets (Sentiment140)

Ce projet met en œuvre une chaîne complète d'analyse de sentiments sur des tweets (dataset Sentiment140) en Python : exploration, prétraitement, transformation textuelle, réduction de dimension, clustering et classification. Il produit des métriques et des figures prêtes à être intégrées dans un rapport.

## Objectifs

- Explorer le dataset : statistiques clés et distributions.
- Construire des variables pertinentes (longueur, nombre de mots).
- Transformer le texte (nettoyage, TF‑IDF, n‑grammes).
- Réduire la dimension (TruncatedSVD).
- Regrouper (k‑means) et interpréter les clusters.
- Classifier (k‑NN et arbre de décision) et évaluer les performances.

## Structure du projet

- `data/` : données d’entrée
  - `tweets.csv` (Sentiment140, 6 colonnes : target, ids, date, flag, user, text)
- `src/` : code source (architecture modulaire)
  - [data_loading.py](file:///d:/L3/MATH_511/mini_projet/src/data_loading.py) : chargement du dataset
  - [preprocessing.py](file:///d:/L3/MATH_511/mini_projet/src/preprocessing.py) : nettoyage du texte, variables dérivées
  - [exploration.py](file:///d:/L3/MATH_511/mini_projet/src/exploration.py) : statistiques et visualisations EDA
  - [models_classification.py](file:///d:/L3/MATH_511/mini_projet/src/models_classification.py) : pipelines TF‑IDF + k‑NN / arbre
  - [models_clustering.py](file:///d:/L3/MATH_511/mini_projet/src/models_clustering.py) : pipeline TF‑IDF + SVD + k‑means
  - [main.py](file:///d:/L3/MATH_511/mini_projet/src/main.py) : orchestration (génère tous les livrables)
- `outputs/` : résultats (généré à l’exécution)
  - `basic_stats.csv` : statistiques de base
  - `classification_metrics_knn.csv`, `classification_metrics_decision_tree.csv`
  - `clustering_results.csv` : cluster, cible, texte nettoyé, composantes SVD (2D)
  - `clustering_metrics.csv` : métriques de clustering (silhouette, ARI)
  - `figures/` : figures (distributions et nuages SVD)

## Installation

### Option A : avec `uv` (recommandé)

```bash
cd ./emotion_classification
uv venv .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Option B : avec `pip`

```bash
cd d:\L3\MATH_511\mini_projet
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution

Pour tout générer en une fois (statistiques, figures, clustering, classification) :

```bash
# Avec uv (utilise l'env actif ou isole l'exécution)
uv run python -m src.main

# Ou depuis l'environnement virtuel activé
python -m src.main
```

Par défaut, `src/main.py` échantillonne jusqu’à 50 000 tweets pour la classification et le clustering afin d’accélérer l’exécution. Vous pouvez augmenter ou réduire cette taille en modifiant les paramètres `n_rows` dans `run_classification` et `run_clustering` de [main.py](file:///d:/L3/MATH_511/mini_projet/src/main.py).

## Sorties générées

- Exploration
  - `outputs/basic_stats.csv`
  - `outputs/figures/target_distribution.png`
  - `outputs/figures/text_length_distribution.png`
- Clustering + Réduction de dimension
  - `outputs/clustering_results.csv` (cluster, cible, texte, `component_1`, `component_2`)
  - `outputs/clustering_metrics.csv` (silhouette, ARI)
  - `outputs/figures/svd_scatter_by_cluster.png`
  - `outputs/figures/svd_scatter_by_sentiment.png`
- Classification
  - `outputs/classification_metrics_knn.csv`
  - `outputs/classification_metrics_decision_tree.csv`

## Choix de modélisation (résumé)

- Représentation : `TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")`.
- Classification :
  - k‑NN (`n_neighbors=7`).
  - Arbre de décision (profondeur max 30, feuilles min 5) pour limiter l’overfitting.
- Clustering :
  - `TruncatedSVD(n_components=100)` sur TF‑IDF, puis `KMeans(n_clusters=10, n_init=10)`.

Ces paramètres sont un bon point de départ et peuvent être ajustés selon les ressources et les objectifs.

## Interprétation des résultats

- Comparez les métriques de classification (accuracy, precision_macro, recall_macro, f1_macro) entre k‑NN et arbre pour discuter des compromis.
- Utilisez `clustering_results.csv` avec les figures SVD pour :
  - Visualiser la séparation spatiale des clusters.
  - Vérifier la cohérence grossière avec les sentiments (via ARI / la coloration par cible).
- Appuyez la discussion sur la distribution des labels et la longueur des tweets (biais éventuels).

## Conseils de performance

- Commencez avec `n_rows=50_000` pour la classification et le clustering si votre machine est limitée.
- Augmentez progressivement `n_rows` selon la mémoire disponible.
- Pour de meilleures performances en supervision : Logistic Regression ou Linear SVM (extensions faciles).

## Pistes d’amélioration

- Ajouter un modèle linéaire (LogReg / LinearSVC) pour enrichir la comparaison.
- Intégrer la gestion des emojis, abréviations et négations dans le prétraitement.
- Utiliser des embeddings (FastText, GloVe) ou des modèles transformers (BERT).
- Implémenter une CLI (arguments pour `n_rows`, `n_clusters`, etc.).

## Dépannage

- Problèmes d’installation : assurez-vous que l’environnement virtuel est activé et que `uv` ou `pip` installent bien les paquets de `requirements.txt`.
- Mémoire insuffisante : diminuez `n_rows` dans [main.py](file:///d:/L3/MATH_511/mini_projet/src/main.py).
- Figures non générées : vérifiez que le dossier `outputs/figures/` existe (créé automatiquement) et que `matplotlib` et `seaborn` sont installés.

---
