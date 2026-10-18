"""Task C (Milestone 1): baseline K-Means. k is investigated with silhouette / Davies-Bouldin / elbow.
No labels are used to pick parameters; cancel rate per cluster is a post-hoc profile only."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import path_of
from src.evaluation.cv import save_table
from src.features.engineering import add_engineered
from src.features.feature_sets import FEATURES
from src.preprocessing.pipelines import make_preprocessor


def run(cfg):
    seed = cfg["seed"]
    cl = cfg["clustering"]
    train = add_engineered(pd.read_csv(path_of(cfg, "train")))
    num, cat = FEATURES["clustering"]["numeric"], FEATURES["clustering"]["categorical"]
    prep = make_preprocessor(num, cat, dense=True)
    Z = prep.fit_transform(train[num + cat])  # fitted on TRAIN only (the 20% TEST set is never opened here)

    rows = []
    for k in range(cl["k_range"][0], cl["k_range"][1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(Z, km.labels_, sample_size=cl["silhouette_sample"], random_state=seed),
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            "inertia": km.inertia_,
        })
    scan = pd.DataFrame(rows)
    save_table(scan, cfg, "clustering_k_scan")
    print(scan.round(3).to_string(index=False))

    k0 = cl["baseline_k"]
    km = KMeans(n_clusters=k0, n_init=10, random_state=seed).fit(Z)
    train["cluster"] = km.labels_
    profile = train.groupby("cluster").agg(
        size=("ADR", "size"), LeadTime=("LeadTime", "mean"), TotalNights=("TotalNights", "mean"),
        TotalGuests=("TotalGuests", "mean"), ADR=("ADR", "mean"),
        cancel_rate_posthoc=("IsCanceled", "mean")).round(3).reset_index()
    save_table(profile, cfg, f"clustering_profile_k{k0}")
    print(profile.to_string(index=False))

    fig_dir = path_of(cfg, "fig_dir")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(scan.k, scan.inertia, "o-", color="#2b5d9b")
    ax[0].set(xlabel="k", ylabel="Inertia", title="Elbow")
    ax[1].plot(scan.k, scan.silhouette, "o-", color="#2e8b57")
    ax[1].set(xlabel="k", ylabel="Silhouette (higher = better)", title="Silhouette")
    ax[2].plot(scan.k, scan.davies_bouldin, "o-", color="#c0392b")
    ax[2].set(xlabel="k", ylabel="Davies-Bouldin (lower = better)", title="Davies-Bouldin")
    fig.tight_layout()
    fig.savefig(fig_dir / "clu_k_selection.png", dpi=150)
    plt.close(fig)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Z), size=min(6000, len(Z)), replace=False)
    P = PCA(n_components=2, random_state=seed).fit_transform(Z)[idx]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sc = ax.scatter(P[:, 0], P[:, 1], c=km.labels_[idx], s=5, alpha=0.5, cmap="viridis")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"K-Means k={k0} in PCA 2-D (6,000-point sample)")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "clu_pca_scatter.png", dpi=150)
    plt.close(fig)
    return scan, profile
