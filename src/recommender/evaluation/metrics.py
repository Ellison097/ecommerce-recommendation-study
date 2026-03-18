from __future__ import annotations

from collections import Counter
import math
import numpy as np

from .rerank import rerank_with_diversity_and_fairness


def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & relevant) / len(rec_k)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def average_precision_at_k(recommended, relevant, k):
    hits = 0
    score = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / max(1, min(len(relevant), k))


def intra_list_diversity(recommended, item_group):
    groups = [item_group.get(item, 'unknown') for item in recommended]
    if len(groups) <= 1:
        return 0.0
    same = 0
    total = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total += 1
            same += int(groups[i] == groups[j])
    return 1 - same / total


def evaluate_model(model, dataset, config):
    k = config['training']['top_k']
    rerank_enabled = config['reranking']['enabled']

    user_relevant = dataset.test.groupby('user_idx')['item_id'].apply(set).to_dict()
    metrics = []
    exposure = Counter()

    for user_idx, relevant in user_relevant.items():
        recs = model.recommend(user_idx, top_k=max(k * 3, 20))
        if rerank_enabled:
            recs = rerank_with_diversity_and_fairness(
                recs,
                dataset.item_group,
                lambda_diversity=config['reranking']['lambda_diversity'],
                lambda_fairness=config['reranking']['lambda_fairness'],
                top_k=k,
            )
        rec_items = [item for item, _ in recs[:k]]
        for item in rec_items:
            exposure[dataset.item_group.get(item, 'unknown')] += 1
        metrics.append({
            'precision@k': precision_at_k(rec_items, relevant, k),
            'recall@k': recall_at_k(rec_items, relevant, k),
            'ndcg@k': ndcg_at_k(rec_items, relevant, k),
            'map@k': average_precision_at_k(rec_items, relevant, k),
            'diversity': intra_list_diversity(rec_items, dataset.item_group),
        })

    coverage = len(set(item for user_idx in user_relevant for item, _ in model.recommend(user_idx, top_k=k))) / len(dataset.item_encoder)
    total_exposure = sum(exposure.values()) or 1
    fairness_gap = abs(exposure.get('head', 0) - exposure.get('long_tail', 0)) / total_exposure

    agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
    agg['coverage'] = coverage
    agg['fairness_gap'] = fairness_gap
    return agg
