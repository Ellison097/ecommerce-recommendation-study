#!/usr/bin/env python3
"""Main experiment driver – trains 8 models, evaluates, and generates all outputs."""
import os, sys, json, logging, yaml, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import InteractionDataset
from src.evaluator import evaluate_model
from src.models import (PopularityModel, ItemKNNModel, BPRMFModel,
                         NeuMFModel, LightGCNModel, MultiVAEModel,
                         SASRecModel, EnsembleModel)
from src.visualizer import generate_all_figures

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    set_seed(seed)
    mcfg = cfg["models"]
    k_vals = tuple(cfg["evaluation"]["k_values"])
    fig_dir = cfg["results"]["figures_dir"]
    met_dir = cfg["results"]["metrics_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────
    log.info("Loading interaction dataset …")
    ds = InteractionDataset({**cfg["data"], "seed": seed, "max_seq_len": 50})
    stats = ds.get_stats()
    log.info("Dataset stats: %s", json.dumps(stats, indent=2))
    pd.DataFrame([stats]).to_csv(f"{met_dir}/dataset_stats.csv", index=False)

    # ── 2. Models ────────────────────────────────────────
    models = {
        "Popularity": PopularityModel(ds.n_users, ds.n_items, mcfg),
        "ItemKNN":    ItemKNNModel(ds.n_users, ds.n_items, mcfg),
        "BPR-MF":     BPRMFModel(ds.n_users, ds.n_items, mcfg),
        "NeuMF":      NeuMFModel(ds.n_users, ds.n_items, mcfg),
        "LightGCN":   LightGCNModel(ds.n_users, ds.n_items, mcfg),
        "Multi-VAE":  MultiVAEModel(ds.n_users, ds.n_items, mcfg),
        "SASRec":     SASRecModel(ds.n_users, ds.n_items, mcfg),
    }

    histories = {}
    all_results = {}

    for name, model in models.items():
        log.info("═══ Training %s ═══", name)
        t0 = time.time()
        model.fit(ds, verbose=True)
        train_sec = time.time() - t0
        histories[name] = model.get_training_losses()

        log.info("Evaluating %s …", name)
        metrics = evaluate_model(model, ds, k_values=k_vals)
        metrics["Train_Time_s"] = round(train_sec, 1)
        all_results[name] = metrics
        log.info("%s  NDCG@10=%.4f  HR@10=%.4f", name,
                 metrics.get("NDCG@10", 0), metrics.get("HR@10", 0))

    # ── 3. Ensemble ──────────────────────────────────────
    log.info("═══ Building Ensemble ═══")
    ens = EnsembleModel(ds.n_users, ds.n_items, mcfg)
    deep_models = {k: v for k, v in models.items() if k not in ("Popularity",)}
    ens.set_models(deep_models)
    ens.fit(ds, verbose=True)
    ens_metrics = evaluate_model(ens, ds, k_values=k_vals)
    ens_metrics["Train_Time_s"] = 0
    all_results["Ensemble"] = ens_metrics
    models["Ensemble"] = ens
    log.info("Ensemble  NDCG@10=%.4f  HR@10=%.4f",
             ens_metrics["NDCG@10"], ens_metrics["HR@10"])

    # ── 4. Collect results ───────────────────────────────
    rows = []
    for name, m in all_results.items():
        row = {"Model": name}
        row.update(m)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{met_dir}/all_metrics.csv", index=False)
    log.info("Saved metrics to %s/all_metrics.csv", met_dir)

    all_k_results = {}
    for k in k_vals:
        all_k_results[k] = {row["Model"]: row[f"NDCG@{k}"] for _, row in results_df.iterrows()}

    # ── 5. Cold-start analysis ───────────────────────────
    log.info("Running cold-start analysis …")
    cold_start = _cold_start_analysis(models, ds, k_vals)
    pd.DataFrame(cold_start).to_csv(f"{met_dir}/cold_start.csv", index=False)

    # ── 6. Visualisations ────────────────────────────────
    log.info("Generating figures …")
    generate_all_figures(results_df, all_k_results, histories, ds, models, fig_dir)

    # ── 7. Cold-start chart ──────────────────────────────
    from src.visualizer import plot_cold_start
    cs_by_group = {}
    cs_df = pd.DataFrame(cold_start)
    for _, row in cs_df.iterrows():
        g = row["group"]
        if g not in cs_by_group:
            cs_by_group[g] = {}
        cs_by_group[g][row["model"]] = row["NDCG@10"]
    plot_cold_start(cs_by_group, f"{fig_dir}/cold_start.png")

    log.info("══════════ DONE ══════════")
    print("\n" + results_df.to_string(index=False))


def _cold_start_analysis(models, ds, k_vals):
    """Bin users by training-set activity and evaluate per group."""
    from src.evaluator import evaluate_model as _ev
    counts = {u: len(ds.train_user_items[u]) for u in ds.test_targets}
    bins = [(5, 10, "5-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 9999, "50+")]
    rows = []
    for lo, hi, label in bins:
        subset_users = {u: ds.test_targets[u] for u in counts if lo <= counts[u] <= hi}
        if len(subset_users) < 10:
            continue
        # Sample for speed
        if len(subset_users) > 2000:
            keys = list(subset_users.keys())
            np.random.shuffle(keys)
            subset_users = {k: subset_users[k] for k in keys[:2000]}

        class _SubDS:
            pass
        sub = _SubDS()
        sub.test_targets = subset_users
        sub.train_user_items = ds.train_user_items
        sub.train_matrix = ds.train_matrix
        sub.n_items = ds.n_items
        sub.sample_negatives = ds.sample_negatives

        for mname, model in models.items():
            if mname == "Ensemble":
                continue
            m = _ev(model, sub, k_values=(10,), n_neg=99, mode="test")
            rows.append({"group": label, "model": mname,
                         "NDCG@10": m["NDCG@10"], "HR@10": m["HR@10"]})
    return rows


if __name__ == "__main__":
    main()
