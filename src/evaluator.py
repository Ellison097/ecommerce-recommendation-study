"""Comprehensive evaluation: ranking metrics + beyond-accuracy metrics."""
import numpy as np
from collections import Counter
from tqdm import tqdm


def _hr(ranked, target, k):
    return float(target in ranked[:k])

def _ndcg(ranked, target, k):
    for i, item in enumerate(ranked[:k]):
        if item == target:
            return 1.0 / np.log2(i + 2)
    return 0.0

def _mrr(ranked, target, k):
    for i, item in enumerate(ranked[:k]):
        if item == target:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_model(model, dataset, k_values=(5, 10, 20), n_neg=99, mode="test"):
    """Sampled-ranking evaluation (1 positive + n_neg negatives)."""
    targets = dataset.test_targets if mode == "test" else dataset.val_targets
    max_k = max(k_values)

    per_k = {k: {"HR": [], "NDCG": [], "MRR": []} for k in k_values}
    item_counter = Counter()
    all_recs = []

    for uid, pos in tqdm(targets.items(), desc=f"Eval({mode})", leave=False):
        negs = dataset.sample_negatives(uid, n_neg)
        cands = np.concatenate([[pos], negs])
        scores = model.predict(uid, cands)
        order = np.argsort(-scores)
        ranked = cands[order]

        all_recs.append(ranked[:max_k])
        for it in ranked[:max_k]:
            item_counter[it] += 1

        for k in k_values:
            per_k[k]["HR"].append(_hr(ranked, pos, k))
            per_k[k]["NDCG"].append(_ndcg(ranked, pos, k))
            per_k[k]["MRR"].append(_mrr(ranked, pos, k))

    metrics = {}
    for k in k_values:
        for m in ("HR", "NDCG", "MRR"):
            metrics[f"{m}@{k}"] = float(np.mean(per_k[k][m]))

    # ---- beyond-accuracy ----
    unique_items = set()
    for r in all_recs:
        unique_items.update(r.tolist())
    metrics["Coverage"] = len(unique_items) / dataset.n_items

    pop = np.asarray(dataset.train_matrix.sum(0)).flatten()
    pop_norm = pop / (pop.sum() + 1e-12)
    novelty = []
    for r in all_recs:
        for it in r:
            p = pop_norm[it]
            if p > 0:
                novelty.append(-np.log2(p))
    metrics["Novelty"] = float(np.mean(novelty)) if novelty else 0.0

    counts = np.array(list(item_counter.values()))
    if len(counts) > 1:
        s = np.sort(counts).astype(float)
        n = len(s)
        idx = np.arange(1, n + 1)
        metrics["Gini"] = float((2 * np.sum(idx * s)) / (n * np.sum(s)) - (n + 1) / n)
    else:
        metrics["Gini"] = 0.0

    return metrics
