from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline


def build_clustering_pipeline(
    n_components: int = 100,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    reducer = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    pipeline: Pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("reducer", reducer),
            ("clusterer", kmeans),
        ]
    )
    return pipeline


def evaluate_clustering(
    pipeline: Pipeline,
    texts: np.ndarray,
    labels: np.ndarray | None = None,
) -> Dict[str, float]:
    features = pipeline.named_steps["vectorizer"].transform(texts)
    reduced = pipeline.named_steps["reducer"].transform(features)
    cluster_labels = pipeline.named_steps["clusterer"].predict(reduced)
    metrics: Dict[str, float] = {}
    metrics["silhouette"] = float(silhouette_score(reduced, cluster_labels))
    if labels is not None:
        metrics["adjusted_rand"] = float(adjusted_rand_score(labels, cluster_labels))
    return metrics, cluster_labels


def fit_clustering_model(
    texts: np.ndarray,
    n_components: int = 100,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray]:
    pipeline = build_clustering_pipeline(
        n_components=n_components,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    pipeline.fit(texts)
    features = pipeline.named_steps["vectorizer"].transform(texts)
    reduced = pipeline.named_steps["reducer"].transform(features)
    cluster_labels = pipeline.named_steps["clusterer"].predict(reduced)
    return pipeline, cluster_labels
