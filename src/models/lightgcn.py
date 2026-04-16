"""LightGCN – Simplified Graph Convolution (He et al., SIGIR 2020)."""
import numpy as np, torch, torch.nn as nn
from scipy.sparse import coo_matrix
from tqdm import trange


class _LightGCN(nn.Module):
    def __init__(self, n_u, n_i, dim, n_layers):
        super().__init__()
        self.n_u, self.n_i, self.n_layers = n_u, n_i, n_layers
        self.user_emb = nn.Embedding(n_u, dim)
        self.item_emb = nn.Embedding(n_i, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.adj = None

    def build_adj(self, sp_mat, device):
        """Normalised adjacency of the user-item bipartite graph."""
        coo = coo_matrix(sp_mat)
        n = self.n_u + self.n_i
        r = np.concatenate([coo.row, coo.col + self.n_u])
        c = np.concatenate([coo.col + self.n_u, coo.row])
        deg = np.zeros(n)
        np.add.at(deg, r, 1)
        d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        vals = d_inv[r] * d_inv[c]
        idx = torch.stack([torch.from_numpy(r).long(), torch.from_numpy(c).long()])
        self.adj = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32),
                                           (n, n)).to(device)

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        out = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            out.append(x)
        final = torch.mean(torch.stack(out), 0)
        return final[:self.n_u], final[self.n_u:]


class LightGCNModel:
    name = "LightGCN"

    def __init__(self, n_users, n_items, config=None):
        cfg = config or {}
        lgcn = cfg.get("lightgcn", {})
        self.n_users, self.n_items = n_users, n_items
        self.epochs = cfg.get("epochs", 30)
        self.bs = cfg.get("batch_size", 512)
        self.lr = cfg.get("lr", 1e-3)
        self.wd = cfg.get("weight_decay", 1e-5)
        self.patience = cfg.get("patience", 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _LightGCN(n_users, n_items,
                             cfg.get("embed_dim", 64),
                             lgcn.get("n_layers", 3)).to(self.device)
        self.losses = []
        self._user_e = None
        self._item_e = None

    def fit(self, dataset, verbose=True):
        self.net.build_adj(dataset.train_matrix, self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        rows, cols = dataset.train_matrix.nonzero()
        n = len(rows)
        rng = np.random.RandomState(42)
        best, wait = 1e9, 0
        bar = trange(self.epochs, desc="LightGCN", disable=not verbose)
        for _ in bar:
            self.net.train()
            perm = rng.permutation(n)
            eloss = 0.0
            for s in range(0, n, self.bs):
                idx = perm[s:s+self.bs]
                u_e, i_e = self.net()
                u = torch.tensor(rows[idx], device=self.device)
                pi = torch.tensor(cols[idx], device=self.device)
                ni_np = np.array([dataset.sample_negatives(int(r), 1)[0] for r in rows[idx]])
                ni = torch.tensor(ni_np, device=self.device)
                pos = (u_e[u] * i_e[pi]).sum(1)
                neg = (u_e[u] * i_e[ni]).sum(1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                eloss += loss.item() * len(idx)
            avg = eloss / n
            self.losses.append(avg)
            bar.set_postfix(loss=f"{avg:.4f}")
            if avg < best - 1e-4:
                best, wait = avg, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self._cache_embeds()

    @torch.no_grad()
    def _cache_embeds(self):
        self.net.eval()
        u, i = self.net()
        self._user_e = u.cpu().numpy()
        self._item_e = i.cpu().numpy()

    def predict(self, uid, item_indices):
        return self._user_e[uid] @ self._item_e[item_indices].T

    def get_training_losses(self):
        return self.losses

    def get_item_embeddings(self):
        return self._item_e
