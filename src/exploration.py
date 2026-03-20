from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    components: pd.DataFrame, clusters: pd.Series, output_path: str
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
        legend=False,
    )
    plt.title("SVD components colored by cluster")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scatter_by_sentiment(
    components: pd.DataFrame, targets: pd.Series, output_path: str
) -> None:
    plt.figure(figsize=(7, 6))
    plot_df = components.copy()
    plot_df["target"] = targets.values
    sns.scatterplot(
        data=plot_df,
        x="component_1",
        y="component_2",
        hue="target",
        palette="Set1",
        s=12,
        linewidth=0,
        alpha=0.6,
        legend=False,
    )
    plt.title("SVD components colored by sentiment")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
