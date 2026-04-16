"""Publication-quality visualisation suite (15+ charts)."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE

# Global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PALETTE = sns.color_palette("Set2", 10)
MODEL_COLORS = {}

def _mc(name):
    if name not in MODEL_COLORS:
        MODEL_COLORS[name] = PALETTE[len(MODEL_COLORS) % len(PALETTE)]
    return MODEL_COLORS[name]

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────── EDA ──────────────────────────
def plot_rating_distribution(df, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    df["rating"].value_counts().sort_index().plot.bar(ax=ax, color=PALETTE[0], edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Rating"); ax.set_ylabel("Count"); ax.set_title("Rating Distribution")
    _save(fig, out)

def plot_user_item_activity(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    uc = df["uidx"].value_counts()
    ic = df["iidx"].value_counts()
    axes[0].hist(uc.values, bins=50, color=PALETTE[1], edgecolor="k", linewidth=0.3)
    axes[0].set_xlabel("# Interactions"); axes[0].set_ylabel("# Users")
    axes[0].set_title("User Activity Distribution"); axes[0].set_yscale("log")
    axes[1].hist(ic.values, bins=50, color=PALETTE[2], edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel("# Interactions"); axes[1].set_ylabel("# Items")
    axes[1].set_title("Item Popularity Distribution"); axes[1].set_yscale("log")
    fig.tight_layout()
    _save(fig, out)

def plot_sparsity_matrix(mat, out, sample=500):
    fig, ax = plt.subplots(figsize=(8, 6))
    n_u, n_i = mat.shape
    su = min(sample, n_u); si = min(sample, n_i)
    sub = mat[:su, :si].toarray()
    ax.imshow(sub, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Items (sampled)"); ax.set_ylabel("Users (sampled)")
    ax.set_title(f"Interaction Matrix Sparsity (density={mat.nnz/(n_u*n_i):.5f})")
    _save(fig, out)

def plot_temporal(df, out):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ts = pd.to_datetime(df["timestamp"], unit="s")
    ts.dt.to_period("M").value_counts().sort_index().plot(ax=ax, color=PALETTE[3])
    ax.set_xlabel("Month"); ax.set_ylabel("Interactions"); ax.set_title("Temporal Interaction Volume")
    fig.autofmt_xdate()
    _save(fig, out)


# ──────────────────── Training ─────────────────────────
def plot_training_curves(histories, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in histories.items():
        if losses:
            ax.plot(range(1, len(losses)+1), losses, label=name, color=_mc(name), linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Convergence")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out)


# ────────────────── Performance ────────────────────────
def plot_metric_bars(results_df, metric, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    models = results_df["Model"].values
    vals = results_df[metric].values
    colors = [_mc(m) for m in models]
    bars = ax.bar(models, vals, color=colors, edgecolor="k", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric); ax.set_title(f"Model Comparison – {metric}")
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)

def plot_grouped_bars(results_df, metrics, out):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    models = results_df["Model"].values
    x = np.arange(len(models))
    w = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        vals = results_df[m].values
        ax.bar(x + i * w, vals, w, label=m, color=PALETTE[i], edgecolor="k", linewidth=0.3)
    ax.set_xticks(x + w * (len(metrics)-1)/2)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Accuracy Metrics Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out)

def plot_metrics_at_k(all_k_results, metric_prefix, out):
    """Line chart: metric@K for varying K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    k_vals = sorted(all_k_results.keys())
    for model_name in all_k_results[k_vals[0]]:
        vals = [all_k_results[k][model_name] for k in k_vals]
        ax.plot(k_vals, vals, marker="o", label=model_name, color=_mc(model_name), linewidth=2)
    ax.set_xlabel("K"); ax.set_ylabel(metric_prefix); ax.set_title(f"{metric_prefix} at Different K")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out)

def plot_radar(results_df, metrics, out):
    from math import pi
    n_metrics = len(metrics)
    angles = [i / n_metrics * 2 * pi for i in range(n_metrics)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in results_df.iterrows():
        vals = [row[m] for m in metrics]
        mn, mx = min(vals), max(vals)
        if mx > mn:
            vals_n = [(v - mn) / (mx - mn) for v in vals]
        else:
            vals_n = [0.5] * len(vals)
        vals_n += vals_n[:1]
        ax.plot(angles, vals_n, "o-", label=row["Model"], color=_mc(row["Model"]), linewidth=1.8)
        ax.fill(angles, vals_n, alpha=0.08, color=_mc(row["Model"]))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Multi-Dimensional Radar", pad=20); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, out)

def plot_heatmap(results_df, metrics, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = results_df.set_index("Model")[metrics]
    # per-column normalisation for visibility
    normed = (data - data.min()) / (data.max() - data.min() + 1e-12)
    sns.heatmap(normed, annot=data.values, fmt=".4f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Normalised"})
    ax.set_title("Performance Heat-Map"); ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save(fig, out)


# ────────────── Beyond-Accuracy ────────────────────────
def plot_beyond_accuracy(results_df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, pal in zip(axes, ["Coverage", "Novelty", "Gini"],
                            [PALETTE[4], PALETTE[5], PALETTE[6]]):
        models = results_df["Model"].values
        vals = results_df[col].values
        bars = ax.barh(models, vals, color=pal, edgecolor="k", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.002, b.get_y() + b.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel(col); ax.set_title(col)
        ax.invert_yaxis()
    fig.suptitle("Beyond-Accuracy Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, out)

def plot_accuracy_diversity_tradeoff(results_df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in results_df.iterrows():
        ax.scatter(row["NDCG@10"], row["Coverage"], s=140,
                   color=_mc(row["Model"]), edgecolors="k", linewidths=0.6, zorder=5)
        ax.annotate(row["Model"], (row["NDCG@10"], row["Coverage"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("NDCG@10 (Accuracy)"); ax.set_ylabel("Coverage (Diversity)")
    ax.set_title("Accuracy vs. Diversity Trade-off"); ax.grid(alpha=0.3)
    _save(fig, out)


# ──────────── Embedding visualisation ──────────────────
def plot_tsne_embeddings(models_dict, out):
    fig, axes = plt.subplots(1, min(len(models_dict), 4), figsize=(5*min(len(models_dict),4), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    idx = 0
    for name, model in models_dict.items():
        emb = model.get_item_embeddings()
        if emb is None or idx >= len(axes):
            continue
        n = min(2000, emb.shape[0])
        subset = emb[np.random.choice(emb.shape[0], n, replace=False)]
        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(subset)
        axes[idx].scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.5, c=PALETTE[idx])
        axes[idx].set_title(f"{name} Item Embeddings (t-SNE)")
        axes[idx].set_xticks([]); axes[idx].set_yticks([])
        idx += 1
    fig.tight_layout()
    _save(fig, out)


# ──────────── Architecture diagrams ────────────────────
def plot_pipeline_diagram(out):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    boxes = [
        (0.03, "Product\nreview data", "#FFF3E0"),
        (0.17, "Preprocessing\n& K-core Filter", "#E3F2FD"),
        (0.31, "Leave-One-Out\nSplit", "#E8F5E9"),
        (0.45, "Phase I\nBaselines\n(Pop · KNN)", "#FFECB3"),
        (0.59, "Phase II\nDeep Models\n(NCF · LightGCN\nVAE · SASRec)", "#F3E5F5"),
        (0.73, "Phase III\nEnsemble\n& Reranking", "#FCE4EC"),
        (0.87, "Evaluation\n& Visualisation", "#E0F7FA"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.2), 0.12, 0.6, facecolor=col,
                                   edgecolor="#333", linewidth=1.5, transform=ax.transAxes,
                                   zorder=2))
        ax.text(x + 0.06, 0.5, txt, ha="center", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes, zorder=3)
    for i in range(len(boxes)-1):
        ax.annotate("", xy=(boxes[i+1][0], 0.5), xytext=(boxes[i][0]+0.12, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    ax.set_title("End-to-End Experimental Pipeline", fontsize=14, pad=20)
    _save(fig, out)

def plot_model_architectures(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _draw_ncf_arch(os.path.join(out_dir, "arch_ncf.png"))
    _draw_lightgcn_arch(os.path.join(out_dir, "arch_lightgcn.png"))
    _draw_vae_arch(os.path.join(out_dir, "arch_multivae.png"))
    _draw_sasrec_arch(os.path.join(out_dir, "arch_sasrec.png"))

def _box(ax, x, y, w, h, txt, fc, fs=9):
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, fc=fc, ec="#333", lw=1.2, zorder=2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=3)

def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))

def _draw_ncf_arch(out):
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    _box(ax, 2.5, 0.7, 2, 0.8, "User ID", "#FFF3E0")
    _box(ax, 7.5, 0.7, 2, 0.8, "Item ID", "#FFF3E0")
    _box(ax, 1.5, 2.2, 1.8, 0.8, "GMF\nEmbed", "#E3F2FD"); _box(ax, 3.5, 2.2, 1.8, 0.8, "MLP\nEmbed", "#E3F2FD")
    _box(ax, 6.5, 2.2, 1.8, 0.8, "GMF\nEmbed", "#E3F2FD"); _box(ax, 8.5, 2.2, 1.8, 0.8, "MLP\nEmbed", "#E3F2FD")
    _box(ax, 2, 3.7, 2.4, 0.8, "Element-wise\nProduct", "#E8F5E9")
    _box(ax, 8, 3.7, 2.4, 0.8, "Concat →\nMLP(128→64→32)", "#F3E5F5")
    _box(ax, 5, 5.2, 3, 0.8, "Concatenate GMF + MLP", "#FFECB3")
    _box(ax, 5, 6.3, 2, 0.7, "σ → Score", "#FCE4EC")
    _arrow(ax, 2.5, 1.1, 1.5, 1.8); _arrow(ax, 2.5, 1.1, 3.5, 1.8)
    _arrow(ax, 7.5, 1.1, 6.5, 1.8); _arrow(ax, 7.5, 1.1, 8.5, 1.8)
    _arrow(ax, 1.5, 2.6, 2, 3.3); _arrow(ax, 6.5, 2.6, 2, 3.3)
    _arrow(ax, 3.5, 2.6, 8, 3.3); _arrow(ax, 8.5, 2.6, 8, 3.3)
    _arrow(ax, 2, 4.1, 5, 4.8); _arrow(ax, 8, 4.1, 5, 4.8)
    _arrow(ax, 5, 5.6, 5, 5.95)
    ax.set_title("NeuMF Architecture", fontsize=14)
    _save(fig, out)

def _draw_lightgcn_arch(out):
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    _box(ax, 2, 0.7, 2.5, 0.8, "User Embeddings", "#FFF3E0")
    _box(ax, 8, 0.7, 2.5, 0.8, "Item Embeddings", "#FFF3E0")
    for l in range(3):
        y = 1.8 + l * 1.1
        _box(ax, 5, y, 4, 0.7, f"Light Graph Conv Layer {l+1}", "#E3F2FD")
    _box(ax, 5, 5.1, 4, 0.7, "Layer-wise Mean Pooling → BPR", "#E8F5E9")
    _arrow(ax, 2, 1.1, 5, 1.45); _arrow(ax, 8, 1.1, 5, 1.45)
    _arrow(ax, 5, 2.15, 5, 2.55); _arrow(ax, 5, 3.25, 5, 3.65); _arrow(ax, 5, 4.35, 5, 4.75)
    ax.set_title("LightGCN Architecture (3 layers)", fontsize=14)
    _save(fig, out)

def _draw_vae_arch(out):
    fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 6)
    _box(ax, 1.5, 0.7, 2.2, 0.8, "User Interaction\nVector x", "#FFF3E0")
    _box(ax, 1.5, 2.2, 2.2, 0.8, "Encoder\n600→200", "#E3F2FD")
    _box(ax, 1.5, 3.7, 1, 0.8, "μ", "#F3E5F5"); _box(ax, 3, 3.7, 1, 0.8, "log σ²", "#F3E5F5")
    _box(ax, 4.5, 3.7, 2, 0.8, "z ~ N(μ,σ²)", "#FFECB3")
    _box(ax, 7, 2.2, 2, 0.8, "Decoder\n200→600→N", "#E3F2FD")
    _box(ax, 7, 0.7, 2, 0.8, "Reconstructed\nx̂", "#E8F5E9")
    _box(ax, 4.5, 5.2, 3.5, 0.8, "Loss = CE(x,x̂) + β·KL(q‖p)", "#FCE4EC")
    _arrow(ax, 1.5, 1.1, 1.5, 1.8); _arrow(ax, 1.5, 2.6, 1.5, 3.3); _arrow(ax, 1.5, 2.6, 3, 3.3)
    _arrow(ax, 2, 3.7, 3.5, 3.7); _arrow(ax, 3.5, 3.7, 4.5, 3.7)
    _arrow(ax, 5.5, 3.7, 7, 2.6); _arrow(ax, 7, 1.8, 7, 1.1)
    _arrow(ax, 4.5, 4.1, 4.5, 4.8)
    ax.set_title("Multi-VAE Architecture", fontsize=14)
    _save(fig, out)

def _draw_sasrec_arch(out):
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    _box(ax, 5, 0.5, 6, 0.7, "Item Sequence [i₁, i₂, …, iₜ]", "#FFF3E0")
    _box(ax, 5, 1.5, 5, 0.7, "Item Embed + Positional Embed", "#E3F2FD")
    _box(ax, 5, 2.5, 5, 0.7, "Causal Self-Attention Block ×2", "#F3E5F5")
    _box(ax, 5, 3.5, 5, 0.7, "Feed-Forward + LayerNorm", "#F3E5F5")
    _box(ax, 5, 4.5, 3, 0.7, "Last Position → BPR", "#E8F5E9")
    for y1, y2 in [(0.85,1.15),(1.85,2.15),(2.85,3.15),(3.85,4.15)]:
        _arrow(ax, 5, y1, 5, y2)
    ax.set_title("SASRec Architecture (Transformer-based)", fontsize=14)
    _save(fig, out)


# ────────────── Cold-start analysis ────────────────────
def plot_cold_start(results_by_group, out):
    """results_by_group: {group_label: {model_name: ndcg_value}}"""
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = list(results_by_group.keys())
    x = np.arange(len(groups))
    models = list(next(iter(results_by_group.values())).keys())
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [results_by_group[g].get(m, 0) for g in groups]
        ax.bar(x + i * w, vals, w, label=m, color=_mc(m), edgecolor="k", linewidth=0.3)
    ax.set_xticks(x + w * (len(models)-1)/2)
    ax.set_xticklabels(groups)
    ax.set_xlabel("User Activity Group"); ax.set_ylabel("NDCG@10")
    ax.set_title("Cold-Start Analysis: NDCG@10 by User Activity")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


# ────────────── Summary table image ───────────────────
def plot_results_table(results_df, out):
    fig, ax = plt.subplots(figsize=(14, 3 + 0.35 * len(results_df)))
    ax.axis("off")
    cols = results_df.columns.tolist()
    cell_text = []
    for _, row in results_df.iterrows():
        cell_text.append([f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    tbl = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D6E4F0")
    ax.set_title("Complete Experimental Results", fontsize=14, pad=20)
    _save(fig, out)


# ────────────── Master function ────────────────────────
def generate_all_figures(results_df, all_k_results, histories,
                         dataset, models_dict, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # EDA
    plot_rating_distribution(dataset.df, f"{fig_dir}/eda_rating_dist.png")
    plot_user_item_activity(dataset.df, f"{fig_dir}/eda_activity.png")
    plot_sparsity_matrix(dataset.train_matrix, f"{fig_dir}/eda_sparsity.png")
    plot_temporal(dataset.df, f"{fig_dir}/eda_temporal.png")

    # Training
    plot_training_curves(histories, f"{fig_dir}/training_curves.png")

    # Accuracy
    plot_grouped_bars(results_df, ["HR@5","HR@10","HR@20"], f"{fig_dir}/accuracy_hr.png")
    plot_grouped_bars(results_df, ["NDCG@5","NDCG@10","NDCG@20"], f"{fig_dir}/accuracy_ndcg.png")
    plot_metric_bars(results_df, "NDCG@10", f"{fig_dir}/ndcg10_bars.png")
    plot_metric_bars(results_df, "MRR@10", f"{fig_dir}/mrr10_bars.png")

    # K-curve
    ndcg_k = {}
    for k in sorted(all_k_results.keys()):
        ndcg_k[k] = {row["Model"]: row[f"NDCG@{k}"] for _, row in results_df.iterrows()}
    plot_metrics_at_k(ndcg_k, "NDCG", f"{fig_dir}/ndcg_at_k.png")

    hr_k = {}
    for k in sorted(all_k_results.keys()):
        hr_k[k] = {row["Model"]: row[f"HR@{k}"] for _, row in results_df.iterrows()}
    plot_metrics_at_k(hr_k, "HR", f"{fig_dir}/hr_at_k.png")

    # Radar & Heatmap
    radar_metrics = ["HR@10", "NDCG@10", "MRR@10", "Coverage", "Novelty"]
    plot_radar(results_df, radar_metrics, f"{fig_dir}/radar.png")
    all_m = [c for c in results_df.columns if c != "Model"]
    plot_heatmap(results_df, all_m, f"{fig_dir}/heatmap.png")

    # Beyond accuracy
    plot_beyond_accuracy(results_df, f"{fig_dir}/beyond_accuracy.png")
    plot_accuracy_diversity_tradeoff(results_df, f"{fig_dir}/tradeoff.png")

    # Embeddings
    plot_tsne_embeddings(models_dict, f"{fig_dir}/tsne_embeddings.png")

    # Architecture diagrams
    plot_pipeline_diagram(f"{fig_dir}/pipeline.png")
    plot_model_architectures(fig_dir)

    # Summary table
    plot_results_table(results_df, f"{fig_dir}/results_table.png")

    print(f"\n✓ Generated {len(os.listdir(fig_dir))} figures in {fig_dir}/")
