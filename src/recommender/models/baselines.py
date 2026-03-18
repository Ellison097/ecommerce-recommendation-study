from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:
    name = 'base'

    def fit(self, dataset):
        self.dataset = dataset
        self.user_seen = dataset.train.groupby('user_idx')['item_id'].apply(set).to_dict()
        return self

    def _filter_seen(self, user_idx, item_ids, scores):
        seen = self.user_seen.get(user_idx, set())
        pairs = [(i, s) for i, s in zip(item_ids, scores) if i not in seen]
        return sorted(pairs, key=lambda x: x[1], reverse=True)


class PopularityRecommender(BaseRecommender):
    name = 'popularity'

    def fit(self, dataset):
        super().fit(dataset)
        counts = dataset.train.groupby('item_id').size().sort_values(ascending=False)
        self.item_ids = counts.index.to_numpy()
        self.scores = counts.to_numpy(dtype=float)
        return self

    def recommend(self, user_idx, top_k=10):
        return self._filter_seen(user_idx, self.item_ids, self.scores)[:top_k]


class UserCFRecommender(BaseRecommender):
    name = 'user_cf'

    def fit(self, dataset):
        super().fit(dataset)
        self.sim = cosine_similarity(dataset.train_matrix)
        return self

    def recommend(self, user_idx, top_k=10):
        scores = self.sim[user_idx] @ self.dataset.train_matrix.toarray()
        item_ids = np.array(list(self.dataset.item_encoder.keys()))
        return self._filter_seen(user_idx, item_ids, scores)[:top_k]


class ItemCFRecommender(BaseRecommender):
    name = 'item_cf'

    def fit(self, dataset):
        super().fit(dataset)
        self.item_sim = cosine_similarity(dataset.train_matrix.T)
        return self

    def recommend(self, user_idx, top_k=10):
        user_vector = self.dataset.train_matrix.getrow(user_idx).toarray().ravel()
        scores = user_vector @ self.item_sim
        item_ids = np.array(list(self.dataset.item_encoder.keys()))
        return self._filter_seen(user_idx, item_ids, scores)[:top_k]


class MatrixFactorizationRecommender(BaseRecommender):
    name = 'matrix_factorization'

    def __init__(self, n_factors=32):
        self.n_factors = n_factors

    def fit(self, dataset):
        super().fit(dataset)
        matrix = dataset.train_matrix.asfptype()
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, s, Vt = svds(matrix, k=k)
        idx = np.argsort(-s)
        self.user_factors = U[:, idx] @ np.diag(np.sqrt(s[idx]))
        self.item_factors = np.diag(np.sqrt(s[idx])) @ Vt[idx, :]
        return self

    def recommend(self, user_idx, top_k=10):
        scores = self.user_factors[user_idx] @ self.item_factors
        item_ids = np.array(list(self.dataset.item_encoder.keys()))
        return self._filter_seen(user_idx, item_ids, scores)[:top_k]
