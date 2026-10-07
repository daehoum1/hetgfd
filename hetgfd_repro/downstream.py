"""Downstream evaluation: semi-supervised node classification with HGT (HetGFD App. B.3).

Imputed features go to the attributed type; non-attributed nodes get one-hot features.
HetGFD Sec. 5.1 ("one-hot node features for each node type") is ambiguous; per-node one-hot
(``node_onehot``, default) matches the paper's Zero / Full numbers on ACM better than a
per-type constant (``type_onehot``): Zero 82.3 vs 80.6 (paper 82.75), Full 93.2 vs 92.1 (92.5).
Model selection on validation Macro-F1, early stopping (patience 200, max 1000 epochs).
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from hetgfd_repro.hetgraph import HetGraph


def build_heterodata(g: HetGraph, x_attr: torch.Tensor, nonattr_feat: str = "node_onehot") -> HeteroData:
    data = HeteroData()
    T = len(g.node_types)
    for i, t in enumerate(g.node_types):
        n = g.counts[i]
        if t == g.attr_type:
            data[t].x = x_attr.float().cpu()
        elif nonattr_feat == "type_onehot":
            data[t].x = F.one_hot(torch.full((n,), i), T).float()
        elif nonattr_feat == "node_onehot":
            data[t].x = torch.eye(n)
        elif nonattr_feat == "raw":  # dataset-provided features of non-attributed types, if any
            ox = g.meta.get("other_x", {})
            data[t].x = ox[t] if t in ox else F.one_hot(torch.full((n,), i), T).float()
        elif nonattr_feat.startswith("raw:"):  # raw features only for the listed types
            ox, use = g.meta.get("other_x", {}), nonattr_feat[4:].split(",")
            data[t].x = ox[t] if t in use and t in ox else torch.eye(n)
        else:
            raise ValueError(nonattr_feat)
    for et, (s, d) in g.edge_ends.items():
        blk = g.block(et).tocoo()
        ei = torch.tensor(np.vstack([blk.row, blk.col]), dtype=torch.long)
        data[s, f"{et}", d].edge_index = ei
        if s != d:
            data[d, f"{et}_rev", s].edge_index = ei.flip(0)
    return data


class HGT(nn.Module):
    def __init__(
        self, data: HeteroData, hidden: int, out: int, layers: int, heads: int, target: str, dropout: float, act: str = "gelu"
    ):
        super().__init__()
        self.act = {"gelu": F.gelu, "relu": F.relu}[act]
        self.lin = nn.ModuleDict({t: nn.Linear(data[t].x.shape[1], hidden) for t in data.node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, data.metadata(), heads) for _ in range(layers)])
        self.out = nn.Linear(hidden, out)
        self.target, self.dropout = target, dropout

    def forward(self, x_dict, ei_dict):
        h = {t: self.act(self.lin[t](x)) for t, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, ei_dict)
            h = {t: F.dropout(v, self.dropout, self.training) for t, v in h.items()}
        return self.out(h[self.target])


def _f1(logits, y):
    p = logits.argmax(1).cpu().numpy()
    y = y.cpu().numpy()
    return f1_score(y, p, average="macro"), f1_score(y, p, average="micro")


def _val_score(logits, y, select):
    if select == "acc":  # the original code selects the epoch by validation accuracy
        return float((logits.argmax(1) == y).float().mean())
    return _f1(logits, y)[0]


def train_eval(
    data: HeteroData,
    g: HetGraph,
    layers: int,
    lr: float,
    hidden: int = 64,
    heads: int = 2,
    dropout: float = 0.2,
    epochs: int = 1000,
    patience: int = 200,
    weight_decay: float = 0.0,
    seed: int = 0,
    device=None,
    act: str = "gelu",
    select: str = "macro",
) -> dict:
    torch.manual_seed(seed)
    data = data.to(device)
    y = g.y.to(device)
    tr, va, te = (g.split[k].to(device) for k in ("train", "val", "test"))
    model = HGT(data, hidden, int(y.max()) + 1, layers, heads, g.label_type, dropout, act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        F.cross_entropy(out[tr], y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
        v = _val_score(out[va], y[va], select)
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    ma, mi = _f1(out[te], y[te])
    del model, opt, out, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # hand cached blocks back so parallel runs on the same GPU do not starve
    return {"val_macro": best, "test_macro": ma, "test_micro": mi, "layers": layers, "lr": lr}  # val_macro = selection score


def evaluate_features(
    g: HetGraph,
    x_attr: torch.Tensor,
    grid_layers=(1, 2, 3),
    grid_lr=(0.1, 0.01, 0.001, 0.0001),
    seed: int = 0,
    device=None,
    nonattr_feat: str = "node_onehot",
    **kw,
) -> dict:
    """Grid search on validation Macro-F1, report test scores of the selected config."""
    data = build_heterodata(g, x_attr, nonattr_feat)
    runs = [train_eval(data, g, L, lr, seed=seed, device=device, **kw) for L, lr in itertools.product(grid_layers, grid_lr)]
    return max(runs, key=lambda r: r["val_macro"])
