"""Structure-agnostic imputation baselines used in HetGFD (App. B.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def knn_impute(x: torch.Tensor, observed: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """HetGFD's custom kNN: k edges per node by cosine similarity (on zero-filled features),
    missing entries filled with the mean of the neighbours' *observed* values in that channel."""
    x0 = torch.where(observed, x, torch.zeros_like(x))
    xn = F.normalize(x0, dim=1)
    m = observed.float()
    out = x0.clone()
    n = x.shape[0]
    for s in range(0, n, chunk):
        sim = xn[s : s + chunk] @ xn.T
        idx = torch.arange(s, min(s + chunk, n), device=x.device)
        sim[torch.arange(len(idx)), idx] = -float("inf")  # no self
        nb = sim.topk(k, dim=1).indices  # (c, k)
        num = (x0[nb] * m[nb]).sum(1)
        den = m[nb].sum(1)
        fill = torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num))
        out[s : s + chunk] = torch.where(observed[s : s + chunk], x0[s : s + chunk], fill)
    return out


def iterative_svd_impute(
    x: torch.Tensor, observed: torch.Tensor, rank: int, max_iters: int = 100, tol: float = 1e-5
) -> torch.Tensor:
    """fancyimpute.IterativeSVD re-implemented in torch (zero init, rank-r SVD reconstruction,
    observed entries clamped, stop when relative change < tol)."""
    x_obs = torch.where(observed, x, torch.zeros_like(x))
    cur = x_obs.clone()
    rank = max(1, min(rank, min(x.shape) - 1))
    for _ in range(max_iters):
        if rank >= min(x.shape) // 2:
            u, s, vh = torch.linalg.svd(cur, full_matrices=False)
            rec = (u[:, :rank] * s[:rank]) @ vh[:rank]
        else:
            u, s, v = torch.svd_lowrank(cur, q=rank, niter=4)
            rec = (u * s) @ v.T
        new = torch.where(observed, x_obs, rec)
        delta = (new - cur).norm() / cur.norm().clamp_min(1e-12)
        cur = new
        if delta < tol:
            break
    return cur
