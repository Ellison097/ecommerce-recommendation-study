"""Multinomial Variational Auto-Encoder for CF (Liang et al., WWW 2018)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import trange


class _MultiVAE(nn.Module):
    def __init__(self, n_items, h_dims, drop):
        super().__init__()
        enc, prev = [], n_items
        for d in h_dims[:-1]:
            enc += [nn.Linear(prev, d), nn.Tanh()]
            prev = d
        enc.append(nn.Linear(prev, h_dims[-1] * 2))
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], h_dims[-1]
        for d in reversed(h_dims[:-1]):
            dec += [nn.Linear(prev, d), nn.Tanh()]
            prev = d
        dec.append(nn.Linear(prev, n_items))
        self.decoder = nn.Sequential(*dec)
        self.drop = nn.Dropout(drop)

    def encode(self, x):
        h = self.encoder(self.drop(F.normalize(x, dim=1)))
        mu, lv = torch.chunk(h, 2, 1)
        return mu, lv

    def reparameterise(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterise(mu, lv)
        return self.decoder(z), mu, lv


class MultiVAEModel:
    name = "Multi-VAE"

    def __init__(self, n_users, n_items, config=None):
        cfg = config or {}
        vc = cfg.get("multivae", {})
        self.n_users, self.n_items = n_users, n_items
        self.epochs = cfg.get("epochs", 30)
        self.bs = cfg.get("batch_size", 512)
        self.lr = cfg.get("lr", 1e-3)
        self.patience = cfg.get("patience", 5)
        self.beta_max = vc.get("beta", 0.2)
        self.anneal = vc.get("anneal_steps", 2000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _MultiVAE(n_items,
                             vc.get("hidden_dims", [600, 200]),
                             vc.get("dropout", 0.5)).to(self.device)
        self.losses = []
        self.train_mat = None

    def fit(self, dataset, verbose=True):
        self.train_mat = dataset.train_matrix.toarray().astype(np.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n_users = self.train_mat.shape[0]
        step, best, wait = 0, 1e9, 0
        bar = trange(self.epochs, desc="Multi-VAE", disable=not verbose)
        rng = np.random.RandomState(42)
        for _ in bar:
            self.net.train()
            perm = rng.permutation(n_users)
            eloss = 0.0
            for s in range(0, n_users, self.bs):
                x = torch.tensor(self.train_mat[perm[s:s+self.bs]], device=self.device)
                recon, mu, lv = self.net(x)
                kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1).mean()
                ce = -torch.sum(F.log_softmax(recon, 1) * x, 1).mean()
                beta = min(self.beta_max, self.beta_max * step / max(self.anneal, 1))
                loss = ce + beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
                eloss += loss.item() * x.size(0)
                step += 1
            avg = eloss / n_users
            self.losses.append(avg)
            bar.set_postfix(loss=f"{avg:.4f}")
            if avg < best - 1e-4:
                best, wait = avg, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

    @torch.no_grad()
    def predict(self, uid, item_indices):
        self.net.eval()
        x = torch.tensor(self.train_mat[uid:uid+1], device=self.device)
        logits = self.net(x)[0].cpu().numpy().flatten()
        return logits[item_indices]

    def get_training_losses(self):
        return self.losses

    def get_item_embeddings(self):
        w = list(self.net.decoder.parameters())[-2]
        return w.detach().cpu().numpy()
