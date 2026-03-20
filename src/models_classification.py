from typing import Dict, Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )


def build_knn_classifier() -> ClassifierMixin:
    return KNeighborsClassifier(n_neighbors=7)


def build_decision_tree_classifier() -> ClassifierMixin:
    return DecisionTreeClassifier(
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=5,
        criterion="gini",
    )


def evaluate_classifier(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }
    return metrics


def train_and_evaluate_model(
    texts: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    vectorizer = build_vectorizer()
    if model_name == "knn":
        classifier = build_knn_classifier()
    elif model_name == "decision_tree":
        classifier = build_decision_tree_classifier()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    pipeline: Pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_classifier(pipeline, X_test, y_test)
    return pipeline, metrics
