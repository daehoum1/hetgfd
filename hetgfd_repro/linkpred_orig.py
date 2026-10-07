"""Link prediction exactly as in the original HetGFD code (third_party/neurips1-master:
run_link_final.py, train_test_split_edges_custom.py, hgt_train_eval.py, models.HGTEncoder*),
plus a corrected evaluation.

Original protocol
  - split of the target edge type: val 5 %, test 10 %, train 85 % (torch.randperm with torch.manual_seed(seed))
  - val/test negatives: uniform non-edges of a (src_type × dst_type) grid sized with the node counts
    BEFORE the largest-connected-component step (IMDB 4278×2081, ACM 4019×7167, DBLP 4057×14328)
  - imputation on the full graph (val/test target edges included); HGT message passing on train edges only
  - encoder: HGT, per-node one-hot for non-attributed types, z = cat([z_src_type, z_dst_type])
  - training: GAE.recon_loss with destination ids offset by n_src; negatives = PyG negative_sampling over
    all nodes of z (any src/dst type pair)
  - evaluation (``orig``): GAE.test(z, pos, neg) WITHOUT the n_src offset → the second index points into
    the src-type block of z (e.g. IMDB scores movie·movie pairs)

Corrected evaluation (``fixed``): destination ids offset by n_src, negatives drawn from the post-LCC grid.
Both evaluations are tracked in the same training run (training does not depend on them); epoch and
(layers, lr) are selected separately per evaluation by its own validation AUC.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import HGTConv
from torch_geometric.utils import negative_sampling

from hetgfd_repro.downstream import build_heterodata
from hetgfd_repro.hetgraph import HetGraph
import dataclasses
import scipy.sparse as sp

TARGET_EDGE = {"ACM": "P-A", "DBLP": "A-P", "IMDB": "M-D"}
RAW_GRID = {"ACM": (4019, 7167), "DBLP": (4057, 14328), "IMDB": (4278, 2081)}  # pre-LCC node counts
EPS = 1e-15


@dataclass
class OrigSplit:
    et: str
    src_type: str
    dst_type: str
    n_src: int
    n_dst: int
    train_p: torch.Tensor  # (2, E) local (src, dst)
    eval: dict  # mode -> {"val": (pos, neg), "test": (pos, neg)} as *indices into z*
    g_train: HetGraph


def make_orig_split(g: HetGraph, dataset: str, seed: int) -> OrigSplit:
    et = TARGET_EDGE[dataset]
    s_t, d_t = g.edge_ends[et]
    n_src, n_dst = g.counts[g.node_types.index(s_t)], g.counts[g.node_types.index(d_t)]
    blk = g.block(et).tocoo()
    row, col = torch.from_numpy(blk.row).long(), torch.from_numpy(blk.col).long()

    # --- train_test_split_edges_custom (val_ratio 0.05, test_ratio 0.1) ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_v, n_t = int(math.floor(0.05 * row.numel())), int(math.floor(0.1 * row.numel()))
    perm = torch.randperm(row.numel())
    row, col = row[perm], col[perm]
    val_p = torch.stack([row[:n_v], col[:n_v]])
    test_p = torch.stack([row[n_v : n_v + n_t], col[n_v : n_v + n_t]])
    train_p = torch.stack([row[n_v + n_t :], col[n_v + n_t :]])
    grid = torch.ones(*RAW_GRID[dataset], dtype=torch.uint8)
    grid[row, col] = 0
    nr, nc = grid.nonzero(as_tuple=False).t()
    p = torch.randperm(nr.numel())[: n_v + n_t]
    nr, nc = nr[p], nc[p]
    val_n, test_n = torch.stack([nr[:n_v], nc[:n_v]]), torch.stack([nr[n_v:], nc[n_v:]])

    # --- corrected negatives: post-LCC grid, separate generator (does not disturb the original draw) ---
    gen = torch.Generator().manual_seed(10_000 + seed)
    pos_keys = set((row * n_dst + col).tolist())
    fixed_neg, need = [], n_v + n_t
    while len(fixed_neg) < need:
        cand = (torch.randint(n_src, (2 * need,), generator=gen) * n_dst + torch.randint(n_dst, (2 * need,), generator=gen)).tolist()
        for c in cand:
            if c not in pos_keys and c not in fixed_neg:
                fixed_neg.append(c)
                if len(fixed_neg) == need:
                    break
    fk = torch.tensor(fixed_neg)
    fval_n, ftest_n = torch.stack([fk[:n_v] // n_dst, fk[:n_v] % n_dst]), torch.stack([fk[n_v:] // n_dst, fk[n_v:] % n_dst])

    off = torch.tensor([[0], [n_src]])
    ev = {
        "orig": {"val": (val_p, val_n), "test": (test_p, test_n)},  # no offset, as in run_link_final.py
        "fixed": {"val": (val_p + off, fval_n + off), "test": (test_p + off, ftest_n + off)},
    }

    # message-passing graph: target edges restricted to the training positives (both directions)
    n = g.num_nodes
    o_s, o_d = g.offset(s_t), g.offset(d_t)
    a = sp.csr_matrix((np.ones(train_p.shape[1], np.float32), (train_p[0].numpy() + o_s, train_p[1].numpy() + o_d)), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    g_train = dataclasses.replace(g, edges={**g.edges, et: a})
    return OrigSplit(et, s_t, d_t, n_src, n_dst, train_p, ev, g_train)


class HGTEncoderOrig(nn.Module):
    """models.HGTEncoder*: Linear → ReLU per type, HGTConv stack, z = cat([z_src, z_dst])."""

    def __init__(self, data, hidden: int, layers: int, heads: int, src_type: str, dst_type: str):
        super().__init__()
        self.lin = nn.ModuleDict({t: nn.Linear(data[t].x.shape[1], hidden) for t in data.node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, data.metadata(), heads) for _ in range(layers)])
        self.src_type, self.dst_type = src_type, dst_type

    def forward(self, x_dict, ei_dict):
        h = {t: F.relu(self.lin[t](x)) for t, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, ei_dict)
        return torch.cat([h[self.src_type], h[self.dst_type]], 0)


def _decode(z, e):
    return torch.sigmoid((z[e[0]] * z[e[1]]).sum(-1))


def _auc_ap(z, pos, neg):
    s = torch.cat([_decode(z, pos), _decode(z, neg)]).cpu().numpy()
    y = np.r_[np.ones(pos.shape[1]), np.zeros(neg.shape[1])]
    return roc_auc_score(y, s), average_precision_score(y, s)


def train_eval_orig(data, sp_: OrigSplit, layers, lr, hidden=64, heads=8, wd=1e-4, epochs=1000, patience=200, seed=0, device=None):
    torch.manual_seed(seed)
    data = data.to(device)
    model = HGTEncoderOrig(data, hidden, layers, heads, sp_.src_type, sp_.dst_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos = torch.stack([sp_.train_p[0], sp_.train_p[1] + sp_.n_src]).to(device)
    n_z = sp_.n_src + sp_.n_dst
    ev = {m: {k: (p.to(device), n.to(device)) for k, (p, n) in d.items()} for m, d in sp_.eval.items()}
    hist = {m: [] for m in ev}
    best = {m: None for m in ev}
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(data.x_dict, data.edge_index_dict)
        neg = negative_sampling(pos, num_nodes=n_z)  # GAE.recon_loss default
        loss = -torch.log(_decode(z, pos) + EPS).mean() - torch.log(1 - _decode(z, neg) + EPS).mean()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            z = model(data.x_dict, data.edge_index_dict)
            for m, d in ev.items():
                v = _auc_ap(z, *d["val"])[0]
                if not hist[m] or v > max(hist[m]):
                    t_auc, t_ap = _auc_ap(z, *d["test"])
                    best[m] = {"val_auc": v, "test_auc": t_auc, "test_ap": t_ap, "epoch": epoch}
                hist[m].append(v)
        # original early-stopping rule; stop only once it holds for both evaluations
        if epoch > patience and all(max(h[-patience:]) <= max(h[:-patience]) for h in hist.values()):
            break
    del model, opt, z
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # hand cached blocks back so parallel runs on the same GPU do not starve
    return {m: {**best[m], "layers": layers, "lr": lr} for m in ev}


def evaluate_features_orig(sp_: OrigSplit, x_attr, grid_layers=(1, 2, 3), grid_lr=(0.1, 0.01, 0.001, 0.0001), seed=0, device=None, **kw):
    """Grid search; the best (layers, lr) is chosen separately for each evaluation mode by its val AUC."""
    data = build_heterodata(sp_.g_train, x_attr, "node_onehot")
    runs = [train_eval_orig(data, sp_, L, lr, seed=seed, device=device, **kw) for L, lr in itertools.product(grid_layers, grid_lr)]
    return {m: max((r[m] for r in runs), key=lambda r: r["val_auc"]) for m in ("orig", "fixed")}
