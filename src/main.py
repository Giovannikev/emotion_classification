from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src import PROJECT_ROOT
from src.data_loading import load_raw_tweets
from src.exploration import (
    compute_basic_statistics,
    plot_target_distribution,
    plot_text_length_distribution,
    plot_scatter_by_cluster,
    compute_top_words,
    plot_top_words,
    plot_confusion_matrix,
    plot_model_accuracy_comparison,
)
from src.models_classification import train_and_evaluate_model
from src.models_clustering import fit_clustering_model
from src.preprocessing import add_text_length_features, clean_text_series


def run_exploratory_analysis(df: pd.DataFrame) -> None:
    stats = compute_basic_statistics(df)
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df, str(figures_dir / "target_distribution.png"))
    plot_text_length_distribution(df, str(figures_dir / "text_length_distribution.png"))
    stats_path = PROJECT_ROOT / "outputs" / "basic_stats.csv"
    minimal_stats = {k: v for k, v in stats.items() if k in ["n_rows", "mean_text_length"]}
    stats_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in minimal_stats.items()]
    )
    stats_df.to_csv(stats_path, index=False)
    cleaned_texts = clean_text_series(df["text"].astype(str).to_numpy())
    top_overall = compute_top_words(cleaned_texts, top_n=20)
    top_overall.to_csv(PROJECT_ROOT / "outputs" / "top_words_overall.csv", index=False)
    plot_top_words(
        top_overall,
        str(figures_dir / "top_words_overall.png"),
        "Top Words (overall)",
    )


def run_classification(
    df: pd.DataFrame,
    model_name: Literal["knn", "decision_tree", "svm"],
    n_rows: int | None = 50000,
) -> dict:
    if n_rows is not None:
        df = df.sample(n=n_rows, random_state=42)
    cleaned_texts = clean_text_series(df["text"].astype(str).to_numpy())
    labels = df["target"].to_numpy()
    _, metrics, y_test, y_pred = train_and_evaluate_model(
        texts=np.array(cleaned_texts),
        labels=labels,
        model_name=model_name,
    )
    metrics_path = (
        PROJECT_ROOT
        / "outputs"
        / f"classification_metrics_{model_name}.csv"
    )
    acc_only = {"accuracy": metrics.get("accuracy")}
    metrics_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in acc_only.items()]
    )
    metrics_df.to_csv(metrics_path, index=False)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 4])
    cm_df = pd.DataFrame(cm, index=[0, 4], columns=[0, 4])
    cm_csv_path = PROJECT_ROOT / "outputs" / f"confusion_matrix_{model_name}.csv"
    cm_df.to_csv(cm_csv_path, index=True)
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        cm,
        str(figures_dir / f"confusion_matrix_{model_name}.png"),
        ["negatif (0)", "positif (4)"],
    )
    return metrics


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
    pca = PCA(n_components=2, random_state=42)
    components_2d = pca.fit_transform(reduced)
    var_ratio = pca.explained_variance_ratio_
    var1 = float(var_ratio[0]) if len(var_ratio) > 0 else None
    var2 = float(var_ratio[1]) if len(var_ratio) > 1 else None
    comp_df = pd.DataFrame({"component_1": components_2d[:, 0], "component_2": components_2d[:, 1]})
    centroids_df = (
        comp_df.assign(cluster=cluster_labels)
        .groupby("cluster")[["component_1", "component_2"]]
        .mean()
        .reset_index()
    )
    results_df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "target": labels,
            "text": cleaned_texts,
            "component_1": components_2d[:, 0],
            "component_2": components_2d[:, 1],
        }
    )
    clustering_path = PROJECT_ROOT / "outputs" / "clustering_results.csv"
    clustering_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(clustering_path, index=False)

    # Scatter plots for visualization
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    components_df = comp_df
    plot_scatter_by_cluster(
        components_df,
        pd.Series(cluster_labels),
        str(figures_dir / "pca_scatter_by_cluster.png"),
        centroids=centroids_df,
        var1=var1,
        var2=var2,
    )


def main() -> None:
    df = load_raw_tweets()
    df = add_text_length_features(df)
    run_exploratory_analysis(df)
    knn_metrics = run_classification(df, model_name="knn")
    dt_metrics = run_classification(df, model_name="decision_tree")
    svm_metrics = run_classification(df, model_name="svm")
    comparison_df = pd.DataFrame(
        [
            {"model": "knn", "accuracy": knn_metrics.get("accuracy")},
            {"model": "decision_tree", "accuracy": dt_metrics.get("accuracy")},
            {"model": "svm", "accuracy": svm_metrics.get("accuracy")},
        ]
    )
    comparison_path = PROJECT_ROOT / "outputs" / "classification_metrics_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_fig_path = figures_dir / "classification_accuracy_comparison.png"
    plot_model_accuracy_comparison(comparison_df, str(comparison_fig_path))
    run_clustering(df)


if __name__ == "__main__":
    main()
