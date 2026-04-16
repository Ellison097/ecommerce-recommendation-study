"""Item-based K-Nearest-Neighbours collaborative filtering."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class ItemKNNModel:
    name = "ItemKNN"

    def __init__(self, n_users, n_items, config=None):
        self.n_users = n_users
        self.n_items = n_items
        self.k = (config or {}).get("itemknn", {}).get("k", 200)
        self.sim = None
        self.train_matrix = None

    def fit(self, dataset, verbose=True):
        if verbose:
            print(f"  Computing {self.n_items}×{self.n_items} item similarity …")
        item_user = dataset.train_matrix.T.tocsr()
        self.sim = cosine_similarity(item_user, dense_output=False)
        self.train_matrix = dataset.train_matrix

    def predict(self, uid, item_indices):
        user_vec = self.train_matrix[uid].toarray().flatten()
        sub_sim = self.sim[item_indices]
        return np.asarray(sub_sim.dot(user_vec)).flatten()

    def get_training_losses(self):
        return []

    def get_item_embeddings(self):
        return None
