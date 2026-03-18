from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .baselines import BaseRecommender, MatrixFactorizationRecommender


class HybridRecommender(BaseRecommender):
    name = 'hybrid'

    def __init__(self, n_factors=32, alpha=0.7):
        self.n_factors = n_factors
        self.alpha = alpha

    def fit(self, dataset):
        super().fit(dataset)
        self.cf = MatrixFactorizationRecommender(n_factors=self.n_factors).fit(dataset)
        self.content_sim = cosine_similarity(dataset.item_features)
        return self

    def recommend(self, user_idx, top_k=10):
        cf_scores = self.cf.user_factors[user_idx] @ self.cf.item_factors
        seen_rows = self.dataset.train[self.dataset.train['user_idx'] == user_idx]
        seen_item_idx = seen_rows['item_idx'].tolist()
        if seen_item_idx:
            content_scores = self.content_sim[seen_item_idx].mean(axis=0)
        else:
            content_scores = np.zeros(self.dataset.train_matrix.shape[1])
        scores = self.alpha * cf_scores + (1 - self.alpha) * content_scores
        item_ids = np.array([self.dataset.item_decoder[i] for i in range(self.dataset.train_matrix.shape[1])])
        return self._filter_seen(user_idx, item_ids, scores)[:top_k]
