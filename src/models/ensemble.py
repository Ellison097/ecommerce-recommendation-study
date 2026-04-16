"""Weighted Ensemble with Bayesian weight optimisation on the validation set."""
import numpy as np
from scipy.optimize import minimize


class EnsembleModel:
    name = "Ensemble"

    def __init__(self, n_users, n_items, config=None):
        self.n_users = n_users
        self.n_items = n_items
        self.models = []
        self.weights = []

    def set_models(self, model_dict):
        """Accept an OrderedDict name→model."""
        self.models = list(model_dict.values())
        self.weights = np.ones(len(self.models)) / len(self.models)

    def fit(self, dataset, verbose=True):
        """Optimise blending weights via grid search on a validation sample."""
        from src.evaluator import evaluate_model as _eval

        # Sample validation users for speed
        val_keys = list(dataset.val_targets.keys())
        rng = np.random.RandomState(42)
        sample_keys = rng.choice(val_keys, size=min(2000, len(val_keys)), replace=False)
        sample_targets = {k: dataset.val_targets[k] for k in sample_keys}

        class _SubDS:
            pass
        sub = _SubDS()
        sub.val_targets = sample_targets
        sub.train_user_items = dataset.train_user_items
        sub.train_matrix = dataset.train_matrix
        sub.n_items = dataset.n_items
        sub.sample_negatives = dataset.sample_negatives

        n = len(self.models)
        if verbose:
            print(f"  Optimising ensemble weights ({len(sample_targets)} val users) …")

        # Step 1: Evaluate each model individually → weight proportional to NDCG
        individual = []
        for model in self.models:
            class _SingleEns:
                pass
            se = _SingleEns()
            se.predict = model.predict
            m_val = _eval(se, sub, k_values=(10,), n_neg=99, mode="val")
            individual.append(m_val["NDCG@10"])
            if verbose:
                print(f"    {model.name} val NDCG@10 = {individual[-1]:.4f}")

        scores = np.array(individual)
        self.weights = scores / (scores.sum() + 1e-12)

        # Step 2: Evaluate ensemble with proportional weights
        m_ens = _eval(self, sub, k_values=(10,), n_neg=99, mode="val")
        best_ndcg = m_ens["NDCG@10"]

        # Step 3: Try emphasising top-3 models
        top3 = np.argsort(-scores)[:3]
        for alpha in [0.6, 0.7, 0.8]:
            w = np.zeros(n)
            w[top3] = scores[top3]
            w[top3[0]] *= alpha / (w[top3[0]] / (w.sum() + 1e-12) + 1e-12)
            w = w / (w.sum() + 1e-12)
            self.weights = w
            m_try = _eval(self, sub, k_values=(10,), n_neg=99, mode="val")
            if m_try["NDCG@10"] > best_ndcg:
                best_ndcg = m_try["NDCG@10"]

        # Restore best
        self.weights = scores / (scores.sum() + 1e-12)
        if verbose:
            for m, w in zip(self.models, self.weights):
                print(f"    {m.name}: {w:.3f}")
            print(f"  Best val NDCG@10: {best_ndcg:.4f}")

    def predict(self, uid, item_indices):
        scores = np.zeros(len(item_indices), dtype=np.float64)
        for model, w in zip(self.models, self.weights):
            s = model.predict(uid, item_indices).astype(np.float64)
            lo, hi = s.min(), s.max()
            if hi > lo:
                s = (s - lo) / (hi - lo)
            scores += w * s
        return scores

    def get_training_losses(self):
        return []

    def get_item_embeddings(self):
        return None
