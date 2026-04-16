"""Bayesian Personalised Ranking – Matrix Factorisation (Rendle et al., 2009)."""
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


class _BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

    def forward(self, u, pi, ni):
        ue = self.user_emb(u)
        pe = self.item_emb(pi)
        ne = self.item_emb(ni)
        return (ue * pe).sum(1), (ue * ne).sum(1)


class BPRMFModel:
    name = "BPR-MF"

    def __init__(self, n_users, n_items, config=None):
        cfg = config or {}
        self.n_users = n_users
        self.n_items = n_items
        self.dim = cfg.get("embed_dim", 64)
        self.lr = cfg.get("lr", 1e-3)
        self.wd = cfg.get("weight_decay", 1e-5)
        self.reg = cfg.get("bpr_mf", {}).get("reg", 1e-4)
        self.epochs = cfg.get("epochs", 30)
        self.bs = cfg.get("batch_size", 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _BPRMF(n_users, n_items, self.dim).to(self.device)
        self.losses = []

    def fit(self, dataset, verbose=True):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        rows, cols = dataset.train_matrix.nonzero()
        n = len(rows)
        rng = np.random.RandomState(42)
        bar = trange(self.epochs, desc="BPR-MF", disable=not verbose)
        for _ in bar:
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.bs):
                idx = perm[start:start + self.bs]
                u = torch.tensor(rows[idx], device=self.device)
                pi = torch.tensor(cols[idx], device=self.device)
                ni_np = np.array([dataset.sample_negatives(int(r), 1)[0] for r in rows[idx]])
                ni = torch.tensor(ni_np, device=self.device)
                pos_s, neg_s = self.net(u, pi, ni)
                loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-10).mean()
                loss += self.reg * (self.net.user_emb.weight.norm() + self.net.item_emb.weight.norm())
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item() * len(idx)
            self.losses.append(epoch_loss / n)
            bar.set_postfix(loss=f"{self.losses[-1]:.4f}")

    @torch.no_grad()
    def predict(self, uid, item_indices):
        self.net.eval()
        u = torch.tensor([uid], device=self.device)
        items = torch.tensor(item_indices, device=self.device)
        ue = self.net.user_emb(u)
        ie = self.net.item_emb(items)
        return (ue * ie).sum(1).cpu().numpy()

    def get_training_losses(self):
        return self.losses

    @torch.no_grad()
    def get_item_embeddings(self):
        return self.net.item_emb.weight.cpu().numpy()
