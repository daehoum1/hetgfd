"""Re-implementation of HetGFD (Um et al., ICLR 2025) from the paper (official code is not released).

Pipeline (Sec. 4):
  1. virtual (zero) features for non-attributed nodes
  2. preliminary diffusion with Ā = Σ_r |E_r|^{-1} A_r (row-stochastic, known rows fixed)
  3. edge-type-wise homophily H(r) on the pre-imputed matrix → ranking r*_1..r*_R
  4. pseudo-confidence ξ = α^S, S = SPD to nearest source on W = Σ_r β^{-(k_r-1)} A_r
  5. relation-aware diffusion with W̄_ij = β^{k_r-1} ξ_j / ξ_i, row-normalised, known rows fixed

After row normalisation ξ_i cancels, so T_ij ∝ β^{k-1} α^{S_j - S_i}; we evaluate it as a
grouped softmax in log space (|S_j - S_i| ≤ w_ij keeps it bounded, but α^S itself underflows).

Special cases: α = β = 1 → FP+VF;  β = 1 → PCFI+VF without PCFI's inter-channel step.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import softmax as grouped_softmax

from hetgfd_repro.hetgraph import HetGraph


def _global_init(g: HetGraph, observed: torch.Tensor, device):
    n, f = g.num_nodes, g.x.shape[1]
    sl = g.type_slice(g.attr_type)
    x0 = torch.zeros(n, f, device=device)
    known = torch.zeros(n, f, dtype=torch.bool, device=device)
    x0[sl] = torch.where(observed, g.x.to(device), torch.zeros_like(x0[sl]))
    known[sl] = observed
    return x0, known, sl


def _csr(a: sp.spmatrix, device) -> torch.Tensor:
    a = sp.csr_matrix(a, dtype=np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(a.indptr).long(), torch.from_numpy(a.indices).long(), torch.from_numpy(a.data), size=a.shape
    ).to(device)


def _row_normalize(a: sp.spmatrix) -> sp.csr_matrix:
    deg = np.asarray(a.sum(1)).ravel()
    return sp.csr_matrix(sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1e-30), 0.0)) @ a)


def _fixed_point(P: torch.Tensor, x0: torch.Tensor, known: torch.Tensor, K: int) -> torch.Tensor:
    x = x0.clone()
    for _ in range(K):
        x = torch.where(known, x0, P @ x)
    return x


def preliminary_diffusion(g: HetGraph, observed: torch.Tensor, K: int = 100, device=None) -> torch.Tensor:
    """Returns the pre-imputed matrix X̄ for *all* nodes (N, F)."""
    directed = g.meta.get("directed", set())
    # |E_r| = number of edges per direction (a symmetric matrix stores each edge twice)
    abar = sum(a / (a.nnz if et in directed else a.nnz / 2) for et, a in g.edges.items())
    x0, known, _ = _global_init(g, observed, device)
    return _fixed_point(_csr(_row_normalize(abar), device), x0, known, K)


def _mean_pair_cos(xn: torch.Tensor, i: torch.Tensor, j: torch.Tensor, chunk: int = 16384) -> float:
    tot = 0.0
    for s in range(0, len(i), chunk):
        tot += float((xn[i[s : s + chunk]] * xn[j[s : s + chunk]]).sum(1).sum())
    return tot / max(len(i), 1)


def edge_type_homophily(x: torch.Tensor, g: HetGraph, num_samples: int = 200_000, seed: int = 0) -> dict[str, float]:
    """Definition 1: mean cos over r-edges / mean cos over random node pairs in V (chunked: O(chunk·F) memory)."""
    rng = np.random.default_rng(seed)
    xn = F.normalize(x, dim=1)
    n = x.shape[0]
    a = torch.from_numpy(rng.integers(n, size=num_samples)).to(x.device)
    b = torch.from_numpy(rng.integers(n, size=num_samples)).to(x.device)
    keep = a != b
    rand = _mean_pair_cos(xn, a[keep], b[keep])
    out = {}
    directed = g.meta.get("directed", set())
    for et, adj in g.edges.items():
        coo = (adj if et in directed else sp.triu(adj)).tocoo()
        idx = rng.choice(coo.nnz, size=min(num_samples, coo.nnz), replace=False)
        i = torch.from_numpy(coo.row[idx]).long().to(x.device)
        j = torch.from_numpy(coo.col[idx]).long().to(x.device)
        out[et] = _mean_pair_cos(xn, i, j) / rand if abs(rand) > 1e-12 else float("nan")
    return out


def rank_edge_types(h: dict[str, float]) -> dict[str, int]:
    order = sorted(h, key=lambda k: -np.nan_to_num(h[k], nan=-np.inf))
    return {et: k + 1 for k, et in enumerate(order)}  # k = 1 is the most homophilic


def _source_distances(w_pc: sp.csr_matrix, sources: np.ndarray) -> np.ndarray:
    if sources.size == 0:
        return np.full(w_pc.shape[0], np.inf)
    # directed: W[i, j] is the edge i -> j (row i aggregates from j), as in the original networkx DiGraph;
    # identical to the undirected search when every relation is stored symmetrically
    return dijkstra(w_pc, directed=True, indices=sources, min_only=True)


def relation_aware_diffusion(
    g: HetGraph,
    observed: torch.Tensor,
    rank: dict[str, int],
    alpha: float,
    beta: float,
    K: int = 100,
    device=None,
    channel_chunk: int = 256,
    return_dist: bool = False,
    edge_count_norm: bool = False,
):
    """With ``return_dist`` also returns S (source distances) for all nodes: (N,) structural, (N, F) uniform.
    ``edge_count_norm``: also multiply the final transition weight of every r-type edge by 1/|E_r| (edges per
    direction), i.e. the commented-out line in the original hfp.compute_edge_weight_from_pc."""
    x0, known, sl = _global_init(g, observed, device)
    directed = g.meta.get("directed", set())
    n = g.num_nodes
    w_pc = sp.csr_matrix((n, n), dtype=np.float64)
    rows, cols, logw = [], [], []
    for et, adj in g.edges.items():
        k = rank[et]
        w_pc = w_pc + adj.astype(np.float64) * beta ** (-(k - 1))
        coo = adj.tocoo()
        rows.append(coo.row)
        cols.append(coo.col)
        lw = (k - 1) * math.log(beta)
        if edge_count_norm:
            lw -= math.log(adj.nnz if et in directed else adj.nnz / 2)
        logw.append(np.full(coo.nnz, lw))
    dst = torch.from_numpy(np.concatenate(rows)).long().to(device)  # i (receiver)
    src = torch.from_numpy(np.concatenate(cols)).long().to(device)  # j (sender)
    logw = torch.from_numpy(np.concatenate(logw)).float().to(device)
    off = sl.start
    log_alpha = math.log(alpha)

    def dist_for(source_rows: np.ndarray) -> torch.Tensor:
        s = _source_distances(w_pc, source_rows + off)
        return torch.from_numpy(np.nan_to_num(s, posinf=1e12)).float().to(device)

    obs_np = observed.cpu().numpy()
    structural = bool((obs_np == obs_np[:, :1]).all())
    if structural:
        s = dist_for(np.flatnonzero(obs_np[:, 0]))
        logits = logw + (s[src] - s[dst]) * log_alpha
        vals = grouped_softmax(logits, dst, num_nodes=n)
        P = torch.sparse_coo_tensor(torch.stack([dst, src]), vals, (n, n)).coalesce().to_sparse_csr()
        full = _fixed_point(P, x0, known, K)
        return (full, s) if return_dist else full[sl]

    # uniform missing: one SPD / transition per channel, processed in chunks
    out = torch.empty(n, x0.shape[1], device=device)
    s_all = torch.empty(n, x0.shape[1], device=device) if return_dist else None
    cache: dict[bytes, torch.Tensor] = {}
    for c0 in range(0, x0.shape[1], channel_chunk):
        cs = range(c0, min(c0 + channel_chunk, x0.shape[1]))
        dists = []
        for c in cs:
            key = np.packbits(obs_np[:, c]).tobytes()
            if key not in cache:
                cache[key] = dist_for(np.flatnonzero(obs_np[:, c]))
            dists.append(cache[key])
        s = torch.stack(dists, 1)  # (n, C)
        logits = logw[:, None] + (s[src] - s[dst]) * log_alpha
        vals = grouped_softmax(logits, dst, num_nodes=n)  # (E, C)
        xc0, kc = x0[:, c0 : cs.stop], known[:, c0 : cs.stop]
        x = xc0.clone()
        for _ in range(K):
            msg = torch.zeros_like(x).index_add_(0, dst, vals * x[src])
            x = torch.where(kc, xc0, msg)
        out[:, c0 : cs.stop] = x
        if return_dist:
            s_all[:, c0 : cs.stop] = s
        cache.clear()
    return (out, s_all) if return_dist else out[sl]


def hetgfd(
    g: HetGraph,
    observed: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    K: int = 100,
    device=None,
    seed: int = 0,
    rank: dict[str, int] | None = None,
    edge_count_norm: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Full HetGFD. Pass ``rank`` to override the homophily ranking (e.g. random / utility ranking)."""
    info = {}
    if rank is None:
        xbar = preliminary_diffusion(g, observed, K, device)
        h = edge_type_homophily(xbar, g, seed=seed)
        rank = rank_edge_types(h)
        info["H"] = h
    info["rank"] = rank
    info["edge_count_norm"] = edge_count_norm
    return relation_aware_diffusion(g, observed, rank, alpha, beta, K, device, edge_count_norm=edge_count_norm), info


def reversed_homophily_rank(g: HetGraph, observed: torch.Tensor, K: int = 100, device=None, seed: int = 0) -> dict[str, int]:
    """Edge-type ranking in the OPPOSITE order of edge-type-wise homophily (least homophilic first)."""
    h = edge_type_homophily(preliminary_diffusion(g, observed, K, device), g, seed=seed)
    r = rank_edge_types(h)
    return {et: len(r) + 1 - k for et, k in r.items()}


def fp_vf(g: HetGraph, observed: torch.Tensor, K: int = 100, device=None) -> torch.Tensor:
    """FP + virtual features: unweighted diffusion on the union graph (α = β = 1)."""
    rank = {et: 1 for et in g.edges}
    return relation_aware_diffusion(g, observed, rank, 1.0, 1.0, K, device)


def pcfi_vf(g: HetGraph, observed: torch.Tensor, alpha: float = 0.5, K: int = 100, device=None) -> torch.Tensor:
    """PCFI + virtual features, channel-wise part only (β = 1)."""
    rank = {et: 1 for et in g.edges}
    return relation_aware_diffusion(g, observed, rank, alpha, 1.0, K, device)


# ---------------------------------------------------------------------------------------------
# Baselines exactly as in the original HetGFD code (third_party/neurips1-master: fp.py, cfp.py)
# ---------------------------------------------------------------------------------------------


def fp_orig(g: HetGraph, observed: torch.Tensor, K: int = 100, device=None) -> torch.Tensor:
    """fp.py: unweighted union graph, symmetric normalisation D^-1/2 A D^-1/2 with D from column sums."""
    x0, known, sl = _global_init(g, observed, device)
    u = sp.csr_matrix(sum(a for a in g.edges.values()), dtype=np.float64)
    u.data[:] = 1.0
    deg = np.asarray(u.sum(0)).ravel()
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    P = sp.diags(dinv) @ u @ sp.diags(dinv)
    return _fixed_point(_csr(P, device), x0, known, K)[sl]


def pcfi_orig(g: HetGraph, observed: torch.Tensor, alpha: float = 0.5, gamma: float = 0.02, K: int = 100, device=None):
    """cfp.py: pseudo-confidence diffusion on the unweighted graph, then the inter-channel step
    X += γ (1 − β^S) ⊙ ((β^S ⊙ (X − mean)) · Corr),  with β = α (``self.beta = alpha`` in the original)."""
    rank = {et: 1 for et in g.edges}
    out, s = relation_aware_diffusion(g, observed, rank, alpha, 1.0, K, device, return_dist=True)
    s = s[:, None] if s.dim() == 1 else s
    conf = alpha ** s.clamp_max(1e6)
    cor = torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
    a2 = (conf * (out - out.mean(0))) @ cor
    out = out + gamma * (1 - conf) * a2
    return out[g.type_slice(g.attr_type)]
