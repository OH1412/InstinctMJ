"""Grouped Distributor Unit, Luo & Yu (2019), equations 18 and 19.

Source: https://arxiv.org/html/1906.08856#S3
delta is a GROUP SUM, not a per-neuron sigmoid probability.
"""

from __future__ import annotations

import torch
from torch import nn


class GDUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_groups: int = 16,
                 update_budget: float | tuple[float, ...] = 1.0):
        super().__init__()
        if hidden_dim % num_groups or hidden_dim // num_groups < 2:
            raise ValueError("GDU requires equal groups of at least two neurons")
        self.hidden_dim, self.num_groups = hidden_dim, num_groups
        self.group_size = hidden_dim // num_groups
        budgets = torch.as_tensor(update_budget, dtype=torch.float32).flatten()
        if budgets.numel() == 1:
            budgets = budgets.expand(num_groups).clone()
        if budgets.numel() != num_groups or not ((budgets > 0) & (budgets < self.group_size)).all():
            raise ValueError("Each delta must be strictly between zero and group size")
        self.register_buffer("budgets", budgets)
        self.distributor = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        for layer in (self.distributor, self.candidate):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def gates(self, joined: torch.Tensor) -> torch.Tensor:
        logits = self.distributor(joined).reshape(*joined.shape[:-1], self.num_groups, self.group_size)
        d = logits.softmax(dim=-1)
        delta = self.budgets.unsqueeze(-1)
        # Eq. 18c, including its >1 branch (not clipping/rescaling a GRU).
        a = torch.where(delta <= 1, delta * d,
                        (self.group_size - delta) / (self.group_size - 1) * d
                        + (delta - 1) / (self.group_size - 1))
        return a.flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        joined = torch.cat((x, hidden), dim=-1)
        gate = self.gates(joined)
        candidate = self.candidate(joined).tanh()
        return (1 - gate) * hidden + gate * candidate, gate


class GDUMemory(nn.Module):
    """Functional state API: caller owns/reset/detaches independent env states."""
    def __init__(self, input_dim: int, hidden_dim=256, num_groups=16, update_budget=1.0):
        super().__init__()
        self.cell = GDUCell(input_dim, hidden_dim, num_groups, update_budget)

    def initial_state(self, batch_size: int, device=None):
        return self.cell.budgets.new_zeros(batch_size, self.cell.hidden_dim, device=device)

    def forward(self, x, hidden, reset=None):
        if reset is not None:
            hidden = torch.where(reset.bool().unsqueeze(-1), torch.zeros_like(hidden), hidden)
        return self.cell(x, hidden)


class MemoryProjection(nn.Linear):
    def __init__(self, hidden_dim: int, visual_dim: int):
        super().__init__(hidden_dim, visual_dim)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
