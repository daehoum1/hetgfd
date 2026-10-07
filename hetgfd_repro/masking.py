"""Missing-feature masks.

Convention: a mask is a bool tensor of shape (N, d) where True means *observed*.

- ``node`` mode (structural missing): whole feature rows are missing.
- ``entry`` mode (uniform missing): individual entries are missing.
"""

from __future__ import annotations

import torch


def make_observed_mask(
    n: int, d: int, missing_rate: float, mode: str = "node", seed: int = 0, device=None
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if mode == "node":
        rows = torch.rand(n, generator=gen) >= missing_rate
        mask = rows[:, None].expand(n, d).clone()
    elif mode == "entry":
        mask = torch.rand(n, d, generator=gen) >= missing_rate
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    return mask.to(device)


def split_probe(
    observed: torch.Tensor, probe_frac: float, mode: str = "node", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split observed entries into (train, probe). Truly missing entries are never touched."""
    # salt the seed so the probe draw is independent of make_observed_mask(seed=seed)
    gen = torch.Generator().manual_seed(seed + 7_919_000)
    if mode == "node":
        obs_rows = observed.all(dim=1).cpu()
        probe_rows = obs_rows & (torch.rand(observed.shape[0], generator=gen) < probe_frac)
        probe = probe_rows[:, None].expand_as(observed).to(observed.device)
    elif mode == "entry":
        probe = observed & (torch.rand(observed.shape, generator=gen) < probe_frac).to(observed.device)
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    train = observed & ~probe
    return train, probe


def make_observed_mask_orig(n: int, d: int, missing_rate: float, mode: str = "node", seed: int = 0, device=None):
    """utils.get_missing_feature_mask of the original HetGFD code (global torch seed + Bernoulli)."""
    torch.manual_seed(seed)
    if mode == "node":
        m = torch.bernoulli(torch.Tensor([1 - missing_rate]).repeat(n)).bool().unsqueeze(1).repeat(1, d)
    elif mode == "entry":
        m = torch.bernoulli(torch.Tensor([1 - missing_rate]).repeat(n, d)).bool()
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    return m.to(device)
