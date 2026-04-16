"""Most-Popular baseline: recommend items by global interaction count."""
import numpy as np


class PopularityModel:
    name = "Popularity"

    def __init__(self, n_users, n_items, config=None):
        self.n_users = n_users
        self.n_items = n_items
        self.scores = np.zeros(n_items)

    def fit(self, dataset, verbose=True):
        self.scores = np.asarray(dataset.train_matrix.sum(0)).flatten().astype(float)
        self.scores /= self.scores.max() + 1e-12

    def predict(self, uid, item_indices):
        return self.scores[item_indices]

    def get_training_losses(self):
        return []

    def get_item_embeddings(self):
        return None
