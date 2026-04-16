"""Implicit-feedback ratings loader with leave-one-out evaluation protocol."""
import os
import logging
import urllib.request
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict

logger = logging.getLogger(__name__)

# Standard SNAP 5-core rating files (URLs fixed by the data provider).
RATINGS_URLS = {
    "beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv",
    "digital_music": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv",
    "toys": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv",
}


def download_ratings(category="beauty", data_dir="data/ratings"):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"ratings_{category}.csv")
    if os.path.exists(filepath):
        logger.info("Found cached data at %s", filepath)
        return filepath
    url = RATINGS_URLS.get(category)
    if url is None:
        raise ValueError(f"Unknown category '{category}'. Choose from {list(RATINGS_URLS)}")
    logger.info("Downloading benchmark ratings (%s) …", category)
    try:
        urllib.request.urlretrieve(url, filepath)
        logger.info("Saved to %s", filepath)
        return filepath
    except Exception as exc:
        logger.warning("Download failed (%s). Falling back to synthetic data.", exc)
        return None


def generate_synthetic(data_dir="data/ratings", seed=42):
    """Power-law synthetic dataset with statistics similar to typical review data."""
    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "ratings_synthetic.csv")
    if os.path.exists(filepath):
        return filepath
    n_users, n_items, n_inter = 8000, 5000, 120_000
    u_prob = rng.pareto(1.5, n_users) + 1; u_prob /= u_prob.sum()
    i_prob = rng.pareto(1.2, n_items) + 1; i_prob /= i_prob.sum()
    users = rng.choice(n_users, n_inter, p=u_prob)
    items = rng.choice(n_items, n_inter, p=i_prob)
    ratings = rng.choice([1.,2.,3.,4.,5.], n_inter, p=[.04,.06,.12,.30,.48])
    ts = np.sort(rng.randint(1388534400, 1420070400, n_inter))
    df = pd.DataFrame({"user_id": [f"U{u:05d}" for u in users],
                        "item_id": [f"B{i:05d}" for i in items],
                        "rating": ratings, "timestamp": ts})
    df.drop_duplicates(["user_id","item_id"], keep="last", inplace=True)
    df.to_csv(filepath, header=False, index=False)
    logger.info("Generated synthetic data (%d interactions)", len(df))
    return filepath


def _kcore_filter(df, min_u, min_i):
    while True:
        uc = df["user_id"].value_counts()
        ic = df["item_id"].value_counts()
        mask = df["user_id"].isin(uc[uc >= min_u].index) & df["item_id"].isin(ic[ic >= min_i].index)
        if mask.all():
            return df
        df = df[mask].copy()


class InteractionDataset:
    """End-to-end pipeline: download → filter → encode → split → matrices."""

    def __init__(self, config: dict):
        self.cfg = config
        data_dir = config.get("data_dir", "data/ratings")
        category = config.get("category", "beauty")

        filepath = download_ratings(category, data_dir)
        if filepath is None:
            filepath = generate_synthetic(data_dir, config.get("seed", 42))

        raw = pd.read_csv(filepath, header=None,
                          names=["user_id", "item_id", "rating", "timestamp"])
        logger.info("Raw interactions: %d", len(raw))

        df = _kcore_filter(raw, config.get("min_user", 5), config.get("min_item", 5))

        umap = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
        imap = {it: i for i, it in enumerate(sorted(df["item_id"].unique()))}
        df = df.copy()
        df["uidx"] = df["user_id"].map(umap)
        df["iidx"] = df["item_id"].map(imap)

        self.n_users = len(umap)
        self.n_items = len(imap)
        self.user_map, self.item_map = umap, imap
        self.df = df.sort_values(["uidx", "timestamp"]).reset_index(drop=True)
        logger.info("After filtering: %d interactions, %d users, %d items (density=%.6f)",
                     len(df), self.n_users, self.n_items,
                     len(df) / (self.n_users * self.n_items))

        self._split()
        self._build_structures()

    # ------------------------------------------------------------------
    def _split(self):
        """Leave-one-out: last → test, second-to-last → val, rest → train."""
        train, val, test = [], [], []
        for _, grp in self.df.groupby("uidx"):
            if len(grp) < 3:
                train.append(grp)
                continue
            test.append(grp.iloc[[-1]])
            val.append(grp.iloc[[-2]])
            train.append(grp.iloc[:-2])
        self.train_df = pd.concat(train).reset_index(drop=True)
        self.val_df   = pd.concat(val).reset_index(drop=True)
        self.test_df  = pd.concat(test).reset_index(drop=True)
        logger.info("Split → train %d | val %d | test %d",
                     len(self.train_df), len(self.val_df), len(self.test_df))

    def _build_structures(self):
        self.train_matrix = csr_matrix(
            (np.ones(len(self.train_df)),
             (self.train_df["uidx"].values, self.train_df["iidx"].values)),
            shape=(self.n_users, self.n_items))

        self.train_user_items = defaultdict(set)
        for r in self.train_df.itertuples():
            self.train_user_items[r.uidx].add(r.iidx)

        self.val_targets = dict(zip(self.val_df["uidx"], self.val_df["iidx"]))
        self.test_targets = dict(zip(self.test_df["uidx"], self.test_df["iidx"]))

        max_len = self.cfg.get("max_seq_len", 50)
        ts = self.train_df.sort_values(["uidx", "timestamp"])
        self.user_seqs = {}
        for uid, g in ts.groupby("uidx"):
            arr = g["iidx"].values
            self.user_seqs[uid] = arr[-max_len:] if len(arr) > max_len else arr

    # ------------------------------------------------------------------
    def sample_negatives(self, uid, n=99):
        pos = self.train_user_items[uid]
        negs = []
        while len(negs) < n:
            c = np.random.randint(self.n_items)
            if c not in pos:
                negs.append(c)
        return np.array(negs)

    def get_stats(self):
        return {"n_users": self.n_users, "n_items": self.n_items,
                "n_interactions": len(self.df),
                "n_train": len(self.train_df), "n_val": len(self.val_df),
                "n_test": len(self.test_df),
                "density": len(self.df) / (self.n_users * self.n_items),
                "avg_items_per_user": len(self.df) / self.n_users,
                "avg_users_per_item": len(self.df) / self.n_items}
