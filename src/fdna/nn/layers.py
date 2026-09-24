"""FDNA layer (Garvey & Pinto formulation as given in the project report, with soft min) and a monotone value network.

FDNA node operability:  s_j = 1 - mean_i alpha_ij (1 - o_i)   (strength, cumulative degradation)
                        b_j = min(1, min_i (o_i + beta_ij))   (criticality bottleneck)
                        o_j = min(u_j, s_j, b_j)
with alpha, beta, u learnable in [0,1] (sigmoid) and min replaced by a temperature-controlled softmin."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmin(x: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    if tau <= 0:
        return x.min(dim=dim).values
    return -tau * torch.logsumexp(-x / tau, dim=dim)


class FDNALayer(nn.Module):
    """mask: (n_out, n_in) 0/1 dependency graph. Inputs o: (B, n_in) in [0,1]; returns (B, n_out)."""

    def __init__(self, mask: torch.Tensor, tau: float = 0.05, learn_intrinsic: bool = True, init_strength: float = 0.5):
        super().__init__()
        self.register_buffer("mask", mask.float())
        n_out, n_in = mask.shape
        logit = lambda p: float(torch.logit(torch.tensor(p)))
        self.alpha = nn.Parameter(torch.full((n_out, n_in), logit(init_strength)))
        self.beta = nn.Parameter(torch.full((n_out, n_in), logit(0.1)))
        self.u = nn.Parameter(torch.full((n_out,), logit(0.95)), requires_grad=learn_intrinsic)
        self.tau = tau

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        a, b, u = torch.sigmoid(self.alpha), torch.sigmoid(self.beta), torch.sigmoid(self.u)
        m = self.mask
        deg = m.sum(1).clamp(min=1)
        s = 1 - ((m * a)[None] * (1 - o)[:, None, :]).sum(2) / deg
        big = (1 - m)[None] * 10.0
        bt = softmin(o[:, None, :] + b[None] + big, dim=2, tau=self.tau)
        bt = softmin(torch.stack([torch.ones_like(bt), bt], 0), dim=0, tau=self.tau)
        us = u[None].expand_as(s)
        return softmin(torch.stack([us, s, bt], 0), dim=0, tau=self.tau).clamp(0, 1)


class MonotoneValue(nn.Module):
    """f(x, c) non-increasing in every coordinate of c (min-max lattice, Sill 1998):
    f = min_k max_j ( -softplus(w_kj) . c + h_kj(x) ), h from an MLP on the electrical features x."""

    def __init__(self, d_x: int, n_c: int, groups: int = 8, planes: int = 8, hidden: int = 128):
        super().__init__()
        self.g, self.p, self.n_c = groups, planes, n_c
        self.h = nn.Sequential(nn.Linear(d_x, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, groups * planes))
        self.w = nn.Parameter(torch.randn(groups, planes, n_c) * 0.3)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.h(x).view(-1, self.g, self.p)
        lin = -(F.softplus(self.w)[None] * c[:, None, None, :]).sum(3)
        z = lin + h
        return z.max(dim=2).values.min(dim=1).values


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
