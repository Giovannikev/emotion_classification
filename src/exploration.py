from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def compute_basic_statistics(df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    stats["n_rows"] = float(len(df))
    stats["n_unique_users"] = float(df["user"].nunique())
    stats["mean_text_length"] = float(df["text"].astype(str).str.len().mean())
    stats["mean_word_count"] = float(
        df["text"].astype(str).str.split().str.len().mean()
    )
    return stats


def plot_target_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Sentiment distribution")
    plt.xlabel("Sentiment label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_text_length_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(6, 4))
    lengths = df["text"].astype(str).str.len()
    sns.histplot(lengths, bins=50)
    plt.title("Tweet length distribution (characters)")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter_by_cluster(
    components: pd.DataFrame,
    clusters: pd.Series,
    output_path: str,
    centroids: Optional[pd.DataFrame] = None,
    var1: Optional[float] = None,
    var2: Optional[float] = None,
) -> None:
    plt.figure(figsize=(7, 6))
    plot_df = components.copy()
    plot_df["cluster"] = clusters.values
    sns.scatterplot(
        data=plot_df,
        x="component_1",
        y="component_2",
        hue="cluster",
        palette="tab10",
        s=12,
        linewidth=0,
        alpha=0.6,
        legend=True,
    )
    if centroids is not None:
        plt.scatter(
            centroids["component_1"],
            centroids["component_2"],
            c="black",
            s=80,
            marker="X",
            label="centroïdes",
            linewidths=0.5,
        )
    if var1 is not None and var2 is not None:
        xlab = f"Composante 1 (ACP, {var1*100:.1f}% var.)"
        ylab = f"Composante 2 (ACP, {var2*100:.1f}% var.)"
    else:
        xlab = "Component 1"
        ylab = "Component 2"
    plt.title("ACP: composantes colorées par cluster")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        title="Cluster",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_top_words(texts: List[str], top_n: int = 20) -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    X = vectorizer.fit_transform(texts)
    words = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(counts)[::-1][:top_n]
    return pd.DataFrame({"word": words[idx], "count": counts[idx]})


def plot_top_words(top_words: pd.DataFrame, output_path: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plot_df = top_words.sort_values("count", ascending=True)
    sns.barplot(data=plot_df, x="count", y="word", color="#4C72B0")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: str, class_names: List[str]) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_accuracy_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plot_df = metrics_df.copy()
    sns.barplot(data=plot_df, x="model", y="accuracy", palette="Set2")
    plt.ylim(0, 1)
    plt.xlabel("Modèle")
    plt.ylabel("Accuracy")
    plt.title("Comparaison des accuracies des modèles")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
