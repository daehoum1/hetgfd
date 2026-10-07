"""Full heterogeneous graph in the HetGFD setting (HGNN-AC / MAGNN preprocessing).

Nodes are indexed globally (types stacked in ``node_types`` order). Only one node
type is *attributed* (raw features); all others get virtual features during
diffusion. Relations are raw edge types, stored as symmetric N×N 0/1 matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

# name, node type names (index order of node_types.npy), attributed type, label type
HGNNAC_SPECS = {
    "ACM": {"types": ["paper", "author", "subject"], "attr": "paper", "label": "paper"},
    "DBLP": {"types": ["author", "paper", "term", "venue"], "attr": "paper", "label": "author"},
    "IMDB": {"types": ["movie", "director", "actor"], "attr": "movie", "label": "movie"},
}


@dataclass
class HetGraph:
    name: str
    node_types: list[str]
    counts: list[int]
    edges: dict[str, sp.csr_matrix]  # edge type name -> symmetric (N, N) 0/1 adjacency
    edge_ends: dict[str, tuple[str, str]]
    attr_type: str
    x: torch.Tensor  # (N_attr, F) raw features of the attributed type
    label_type: str
    y: torch.Tensor
    split: dict[str, torch.Tensor]  # indices local to label_type
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return sum(self.counts)

    def offset(self, t: str) -> int:
        i = self.node_types.index(t)
        return sum(self.counts[:i])

    def type_slice(self, t: str) -> slice:
        o = self.offset(t)
        return slice(o, o + self.counts[self.node_types.index(t)])

    def type_of_nodes(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.node_types)), self.counts)

    def block(self, et: str) -> sp.csr_matrix:
        """Bipartite block (src-type rows, dst-type cols) of an edge type."""
        s, d = self.edge_ends[et]
        return self.edges[et][self.type_slice(s)][:, self.type_slice(d)].tocsr()


def _load_feat(p: Path) -> np.ndarray:
    if p.with_suffix(".npz").exists():
        return sp.load_npz(p.with_suffix(".npz")).toarray()
    return np.load(p.with_suffix(".npy"))


def load_hgnnac(
    name: str, root: str = "data/raw/HGNNAC/preprocessed", lcc: bool = True, same_type_directed: bool = False
) -> HetGraph:
    """``same_type_directed``: keep same-type relations (ACM paper–paper citations) as the raw directed
    adjacency incl. self-loops, as the original HetGFD code does; otherwise symmetrise them."""
    spec = HGNNAC_SPECS[name]
    d = Path(root) / f"{name}_processed"
    nt = np.load(d / "node_types.npy")
    raw = sp.load_npz(d / "adjM.npz").tocsr().astype(np.float32)
    adj = ((raw + raw.T) > 0).astype(np.float32).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    types = spec["types"]
    ai, li = types.index(spec["attr"]), types.index(spec["label"])
    x = _load_feat(d / f"features_{ai}").astype(np.float32)
    y = np.load(d / "labels.npy")
    sp_idx = np.load(d / "train_val_test_idx.npz")
    split = {k.replace("_idx", ""): sp_idx[k] for k in sp_idx.files}

    keep = np.ones(adj.shape[0], dtype=bool)
    if lcc:
        # LCC over edges between *different* node types. This reproduces HetGFD Table 4 exactly
        # (ACM: 4014 / 7157 / 56 nodes; a plain LCC including P-P citations keeps 4017 / 7165 / 58).
        coo = adj.tocoo()
        cross = nt[coo.row] != nt[coo.col]
        adj_x = sp.csr_matrix((coo.data[cross], (coo.row[cross], coo.col[cross])), shape=adj.shape)
        _, comp = connected_components(adj_x, directed=False)
        keep = comp == np.bincount(comp).argmax()
    new_id = -np.ones(adj.shape[0], dtype=np.int64)
    new_id[keep] = np.arange(keep.sum())
    adj = adj[keep][:, keep].tocsr()
    nt_k = nt[keep]
    counts = [int((nt_k == i).sum()) for i in range(len(types))]

    # local re-indexing of attributed / labelled types
    def local_keep(ti):
        return keep[nt == ti]

    x = x[local_keep(ai)]
    # raw features shipped for other types (not used by HetGFD's imputation; optional downstream input)
    other_x = {}
    for ti, t in enumerate(types):
        if ti != ai and (d / f"features_{ti}.npz").exists() or (ti != ai and (d / f"features_{ti}.npy").exists()):
            other_x[t] = torch.from_numpy(_load_feat(d / f"features_{ti}").astype(np.float32)[local_keep(ti)])
    lk = local_keep(li)
    y = y[lk]
    local_map = -np.ones(lk.size, dtype=np.int64)
    local_map[lk] = np.arange(lk.sum())
    split = {k: local_map[v][local_map[v] >= 0] for k, v in split.items()}

    # split adjacency into edge types by endpoint node types
    edges, ends = {}, {}
    coo = sp.triu(adj).tocoo()
    ts, td = nt_k[coo.row], nt_k[coo.col]
    for a in range(len(types)):
        for b in range(a, len(types)):
            sel = ((ts == a) & (td == b)) | ((ts == b) & (td == a))
            if not sel.any():
                continue
            m = sp.csr_matrix((np.ones(sel.sum(), np.float32), (coo.row[sel], coo.col[sel])), shape=adj.shape)
            m = ((m + m.T) > 0).astype(np.float32).tocsr()
            et = f"{types[a][0].upper()}-{types[b][0].upper()}"
            edges[et], ends[et] = m, (types[a], types[b])
    directed = set()
    if same_type_directed:
        rk = raw[keep][:, keep].tocoo()
        for et, (s_t, d_t) in ends.items():
            if s_t != d_t:
                continue
            ti = types.index(s_t)
            sel = (nt_k[rk.row] == ti) & (nt_k[rk.col] == ti)
            edges[et] = sp.csr_matrix((np.ones(sel.sum(), np.float32), (rk.row[sel], rk.col[sel])), shape=adj.shape)
            directed.add(et)

    return HetGraph(
        name=name,
        node_types=types,
        counts=counts,
        edges=edges,
        edge_ends=ends,
        attr_type=spec["attr"],
        x=torch.from_numpy(x),
        label_type=spec["label"],
        y=torch.from_numpy(y).long(),
        split={k: torch.from_numpy(v) for k, v in split.items()},
        meta={"lcc_removed": int((~keep).sum()), "other_x": other_x, "directed": directed},
    )


def metapath_blocks(g: HetGraph) -> dict[str, list[sp.csr_matrix]]:
    """Length-2 meta-paths attr→B→attr (and direct attr–attr edges) as chains of bipartite blocks."""
    out = {}
    a = g.attr_type
    for et, (s, d) in g.edge_ends.items():
        if s == a and d == a:
            out[et] = [g.block(et)]
            continue
        if a not in (s, d):
            continue
        other = d if s == a else s
        sl_a, sl_o = g.type_slice(a), g.type_slice(other)
        ab = g.edges[et][sl_a][:, sl_o].tocsr()
        name = f"{a[0].upper()}-{other[0].upper()}-{a[0].upper()}"
        out[name] = [ab, ab.T.tocsr()]
    return out
