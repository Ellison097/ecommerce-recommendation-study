"""Neural Collaborative Filtering – NeuMF (He et al., WWW 2017)."""
import numpy as np, torch, torch.nn as nn
from tqdm import trange


class _NeuMF(nn.Module):
    def __init__(self, n_u, n_i, dim, mlp_layers, drop):
        super().__init__()
        self.gmf_u = nn.Embedding(n_u, dim)
        self.gmf_i = nn.Embedding(n_i, dim)
        self.mlp_u = nn.Embedding(n_u, dim)
        self.mlp_i = nn.Embedding(n_i, dim)
        layers = []
        in_d = dim * 2
        for d in mlp_layers:
            layers += [nn.Linear(in_d, d), nn.ReLU(), nn.Dropout(drop)]
            in_d = d
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim + mlp_layers[-1], 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, u, i):
        gmf = self.gmf_u(u) * self.gmf_i(i)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], 1))
        return self.out(torch.cat([gmf, mlp], 1)).squeeze(-1)


class NeuMFModel:
    name = "NeuMF"

    def __init__(self, n_users, n_items, config=None):
        cfg = config or {}
        ncf = cfg.get("ncf", {})
        self.n_users, self.n_items = n_users, n_items
        self.epochs = cfg.get("epochs", 30)
        self.bs = cfg.get("batch_size", 512)
        self.patience = cfg.get("patience", 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _NeuMF(n_users, n_items,
                          cfg.get("embed_dim", 64),
                          ncf.get("mlp_layers", [128, 64, 32]),
                          ncf.get("dropout", 0.2)).to(self.device)
        self.lr = cfg.get("lr", 1e-3)
        self.losses = []

    def fit(self, dataset, verbose=True):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        rows, cols = dataset.train_matrix.nonzero()
        n = len(rows)
        rng = np.random.RandomState(42)
        best_loss, wait = 1e9, 0
        bar = trange(self.epochs, desc="NeuMF", disable=not verbose)
        for _ in bar:
            self.net.train()
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for s in range(0, n, self.bs):
                idx = perm[s:s+self.bs]
                pos_u = torch.tensor(rows[idx], device=self.device)
                pos_i = torch.tensor(cols[idx], device=self.device)
                neg_i_np = np.array([dataset.sample_negatives(int(r), 1)[0] for r in rows[idx]])
                neg_i = torch.tensor(neg_i_np, device=self.device)

                u = torch.cat([pos_u, pos_u])
                i = torch.cat([pos_i, neg_i])
                labels = torch.cat([torch.ones(len(idx)), torch.zeros(len(idx))]).to(self.device)

                logits = self.net(u, i)
                loss = criterion(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item() * len(idx)
            avg = epoch_loss / n
            self.losses.append(avg)
            bar.set_postfix(loss=f"{avg:.4f}")
            if avg < best_loss - 1e-4:
                best_loss, wait = avg, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

    @torch.no_grad()
    def predict(self, uid, item_indices):
        self.net.eval()
        u = torch.full((len(item_indices),), uid, dtype=torch.long, device=self.device)
        i = torch.tensor(item_indices, device=self.device)
        return self.net(u, i).cpu().numpy()

    def get_training_losses(self):
        return self.losses

    @torch.no_grad()
    def get_item_embeddings(self):
        return self.net.gmf_i.weight.cpu().numpy()
