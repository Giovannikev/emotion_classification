from typing import Literal

import numpy as np
import pandas as pd

from src import PROJECT_ROOT
from src.data_loading import load_raw_tweets
from src.exploration import (
    compute_basic_statistics,
    plot_target_distribution,
    plot_text_length_distribution,
    plot_scatter_by_cluster,
    plot_scatter_by_sentiment,
)
from src.models_classification import train_and_evaluate_model
from src.models_clustering import evaluate_clustering, fit_clustering_model
from src.preprocessing import add_text_length_features, clean_text_series


def run_exploratory_analysis(df: pd.DataFrame) -> None:
    stats = compute_basic_statistics(df)
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df, str(figures_dir / "target_distribution.png"))
    plot_text_length_distribution(df, str(figures_dir / "text_length_distribution.png"))
    stats_path = PROJECT_ROOT / "outputs" / "basic_stats.csv"
    stats_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in stats.items()]
    )
    stats_df.to_csv(stats_path, index=False)


def run_classification(
    df: pd.DataFrame,
    model_name: Literal["knn", "decision_tree"],
    n_rows: int | None = 50000,
) -> None:
    if n_rows is not None:
        df = df.sample(n=n_rows, random_state=42)
    cleaned_texts = clean_text_series(df["text"].astype(str).to_numpy())
    labels = df["target"].to_numpy()
    _, metrics = train_and_evaluate_model(
        texts=np.array(cleaned_texts),
        labels=labels,
        model_name=model_name,
    )
    metrics_path = (
        PROJECT_ROOT
        / "outputs"
        / f"classification_metrics_{model_name}.csv"
    )
    metrics_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    )
    metrics_df.to_csv(metrics_path, index=False)


def run_clustering(
    df: pd.DataFrame,
    n_components: int = 100,
    n_clusters: int = 10,
    n_rows: int | None = 50000,
) -> None:
    if n_rows is not None:
        df = df.sample(n=n_rows, random_state=42)
    cleaned_texts = clean_text_series(df["text"].astype(str).to_numpy())
    labels = df["target"].to_numpy()
    texts_array = np.array(cleaned_texts)
    pipeline, cluster_labels = fit_clustering_model(
        texts=texts_array,
        n_components=n_components,
        n_clusters=n_clusters,
    )
    features = pipeline.named_steps["vectorizer"].transform(texts_array)
    reduced = pipeline.named_steps["reducer"].transform(features)
    results_df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "target": labels,
            "text": cleaned_texts,
            "component_1": reduced[:, 0],
            "component_2": reduced[:, 1],
        }
    )
    clustering_path = PROJECT_ROOT / "outputs" / "clustering_results.csv"
    clustering_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(clustering_path, index=False)

    # Compute clustering metrics and save
    metrics, _ = evaluate_clustering(pipeline, texts_array, labels=labels)
    metrics_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    )
    metrics_path = PROJECT_ROOT / "outputs" / "clustering_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Scatter plots for visualization
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    components_df = pd.DataFrame(
        {"component_1": reduced[:, 0], "component_2": reduced[:, 1]}
    )
    plot_scatter_by_cluster(
        components_df,
        pd.Series(cluster_labels),
        str(figures_dir / "svd_scatter_by_cluster.png"),
    )
    plot_scatter_by_sentiment(
        components_df,
        pd.Series(labels),
        str(figures_dir / "svd_scatter_by_sentiment.png"),
    )


def main() -> None:
    df = load_raw_tweets()
    df = add_text_length_features(df)
    run_exploratory_analysis(df)
    run_classification(df, model_name="knn")
    run_classification(df, model_name="decision_tree")
    run_clustering(df)


if __name__ == "__main__":
    main()
