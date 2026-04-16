"""Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018)."""
import numpy as np, torch, torch.nn as nn
from tqdm import trange


class _Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(d * 4, d), nn.Dropout(drop))

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class _SASRec(nn.Module):
    def __init__(self, n_items, d, L, h, B, drop):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(L, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([_Block(d, h, drop) for _ in range(B)])
        self.norm = nn.LayerNorm(d)
        self.L = L

    def forward(self, seq):
        B, L = seq.size()
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(L, L, device=seq.device), 1).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)


class SASRecModel:
    name = "SASRec"

    def __init__(self, n_users, n_items, config=None):
        cfg = config or {}
        sc = cfg.get("sasrec", {})
        self.n_users, self.n_items = n_users, n_items
        self.L = cfg.get("max_seq_len", 50) if "max_seq_len" in cfg else 50
        self.epochs = cfg.get("epochs", 30)
        self.bs = cfg.get("batch_size", 256)
        self.lr = cfg.get("lr", 1e-3)
        self.patience = cfg.get("patience", 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _SASRec(n_items, cfg.get("embed_dim", 64), self.L,
                           sc.get("n_heads", 2), sc.get("n_blocks", 2),
                           sc.get("dropout", 0.2)).to(self.device)
        self.losses = []
        self.seqs = None

    @staticmethod
    def _pad(seq, L):
        if len(seq) >= L:
            return seq[-L:]
        return np.concatenate([np.zeros(L - len(seq), dtype=seq.dtype), seq])

    def fit(self, dataset, verbose=True):
        self.seqs = dataset.user_seqs
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        uids = sorted(self.seqs.keys())
        rng = np.random.RandomState(42)
        best, wait = 1e9, 0
        bar = trange(self.epochs, desc="SASRec", disable=not verbose)
        for _ in bar:
            self.net.train()
            rng.shuffle(uids)
            eloss, cnt = 0.0, 0
            for s in range(0, len(uids), self.bs):
                batch_uids = uids[s:s+self.bs]
                seqs_in, targets, negs = [], [], []
                for u in batch_uids:
                    items = self.seqs[u]
                    if len(items) < 2:
                        continue
                    seq = self._pad(items[:-1], self.L)
                    seqs_in.append(seq)
                    targets.append(items[-1])
                    neg = dataset.sample_negatives(u, 1)[0]
                    negs.append(neg)
                if not seqs_in:
                    continue
                seq_t = torch.tensor(np.array(seqs_in) + 1, device=self.device)  # shift by 1 (0=pad)
                out = self.net(seq_t)[:, -1, :]  # last position
                pos_e = self.net.item_emb(torch.tensor(targets, device=self.device) + 1)
                neg_e = self.net.item_emb(torch.tensor(negs, device=self.device) + 1)
                pos_s = (out * pos_e).sum(1)
                neg_s = (out * neg_e).sum(1)
                loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-10).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                eloss += loss.item() * len(seqs_in)
                cnt += len(seqs_in)
            avg = eloss / max(cnt, 1)
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
        seq = self.seqs.get(uid, np.array([0]))
        seq_pad = self._pad(seq, self.L)
        seq_t = torch.tensor(seq_pad[None] + 1, device=self.device)
        rep = self.net(seq_t)[:, -1, :]
        ie = self.net.item_emb(torch.tensor(item_indices + 1, device=self.device))
        return (rep * ie).sum(1).cpu().numpy()

    def get_training_losses(self):
        return self.losses

    @torch.no_grad()
    def get_item_embeddings(self):
        return self.net.item_emb.weight[1:].cpu().numpy()
