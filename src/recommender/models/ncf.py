from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .baselines import BaseRecommender


class InteractionDataset(Dataset):
    def __init__(self, train_df, n_items, negative_samples=4):
        self.samples = []
        observed = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        rng = np.random.default_rng(42)
        for row in train_df.itertuples():
            self.samples.append((row.user_idx, row.item_idx, 1.0))
            for _ in range(negative_samples):
                neg = rng.integers(0, n_items)
                while neg in observed[row.user_idx]:
                    neg = rng.integers(0, n_items)
                self.samples.append((row.user_idx, int(neg), 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, y = self.samples[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(y, dtype=torch.float32)


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_idx, item_idx):
        x = torch.cat([self.user_emb(user_idx), self.item_emb(item_idx)], dim=1)
        return self.mlp(x).squeeze(-1)


class NCFRecommender(BaseRecommender):
    name = 'ncf'

    def __init__(self, n_epochs=10, batch_size=256, learning_rate=1e-3, weight_decay=1e-5, negative_samples=4, device='cpu'):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negative_samples = negative_samples
        self.device = device

    def fit(self, dataset):
        super().fit(dataset)
        n_users, n_items = dataset.train_matrix.shape
        self.model = NCF(n_users, n_items).to(self.device)
        ds = InteractionDataset(dataset.train, n_items, self.negative_samples)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.n_epochs):
            for u, i, y in loader:
                u, i, y = u.to(self.device), i.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(u, i), y)
                loss.backward()
                opt.step()
        return self

    def recommend(self, user_idx, top_k=10):
        self.model.eval()
        n_items = self.dataset.train_matrix.shape[1]
        item_idx = torch.arange(n_items, device=self.device)
        user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = torch.sigmoid(self.model(user_tensor, item_idx)).cpu().numpy()
        item_ids = np.array([self.dataset.item_decoder[i] for i in range(n_items)])
        return self._filter_seen(user_idx, item_ids, scores)[:top_k]
