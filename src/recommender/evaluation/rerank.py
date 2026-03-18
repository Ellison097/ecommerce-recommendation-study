from __future__ import annotations

from collections import Counter


def rerank_with_diversity_and_fairness(candidates, item_group, lambda_diversity=0.25, lambda_fairness=0.15, top_k=10):
    selected = []
    seen_groups = Counter()
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = None
        for idx, (item_id, score) in enumerate(remaining):
            group = item_group.get(item_id, 'unknown')
            diversity_bonus = 1.0 / (1 + seen_groups[group])
            fairness_bonus = 1.0 if seen_groups[group] == 0 else 0.0
            adjusted = score + lambda_diversity * diversity_bonus + lambda_fairness * fairness_bonus
            if best_score is None or adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        item_id, score = remaining.pop(best_idx)
        group = item_group.get(item_id, 'unknown')
        seen_groups[group] += 1
        selected.append((item_id, score))

    return selected
