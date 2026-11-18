"""Recurrent actor-critic for the cube.

  41-hot features -> linear embedding -> residual MLP blocks -> projection -> LSTM cell
  [LSTM output, projection] -> policy head (12 moves) and value head (moves to go)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubik.cube import N_FEATURES, N_MOVES, NO_MOVE

MASKED = -1e9


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class LSTMCellF(nn.Module):
    """LSTM cell with one fused gate matmul and an fp32 state update."""

    def __init__(self, inp, hid):
        super().__init__()
        self.hid = hid
        self.lin = nn.Linear(inp + hid, 4 * hid)
        with torch.no_grad():
            self.lin.bias.zero_()
            self.lin.bias[hid:2 * hid].fill_(1.0)

    def forward(self, x, h, c):
        g = self.lin(torch.cat([x, h.to(x.dtype)], -1)).float()
        i, f, gg, o = g.chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class CubePolicy(nn.Module):
    def __init__(self, d_model: int = 1024, n_res: int = 2, hidden: int = 512, head: int = 512):
        super().__init__()
        self.config = dict(d_model=d_model, n_res=n_res, hidden=hidden, head=head)
        self.hidden = hidden
        self.embed = nn.Linear(N_FEATURES, d_model)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(41))
        nn.init.zeros_(self.embed.bias)
        self.embed_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(ResBlock(d_model) for _ in range(n_res))
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.SiLU())
        self.lstm = LSTMCellF(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(2 * hidden, head), nn.SiLU(), nn.Linear(head, N_MOVES))
        self.v = nn.Sequential(nn.Linear(2 * hidden, head), nn.SiLU(), nn.Linear(head, 1))
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)
        undo = torch.zeros(N_MOVES + 1, N_MOVES, dtype=torch.bool)
        undo[torch.arange(N_MOVES), torch.arange(N_MOVES) ^ 1] = True
        self.register_buffer("undo", undo, persistent=False)

    def initial_state(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(n, self.hidden, device=device)
        return z, z.clone()

    def one_hot(self, idx: torch.Tensor) -> torch.Tensor:
        dev = idx.device.type
        dt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else self.embed.weight.dtype
        return torch.zeros(idx.shape[0], N_FEATURES, device=idx.device, dtype=dt).scatter_(1, idx.long(), 1.0)

    def trunk(self, idx: torch.Tensor, hot: torch.Tensor | None = None) -> torch.Tensor:
        """[M, 41] feature indices (or their 41-hot vectors) -> [M, hidden] state encoding."""
        x = F.silu(self.embed_norm(self.embed(self.one_hot(idx) if hot is None else hot)))
        for b in self.blocks:
            x = b(x)
        return self.proj(x)

    def heads(self, h: torch.Tensor, x: torch.Tensor, prev: torch.Tensor):
        z = torch.cat([h.to(x.dtype), x], -1)
        zp, zv = self.pi[1](self.pi[0](z)), self.v[1](self.v[0](z))
        with torch.autocast(z.device.type, enabled=False):
            logits = self.pi[2](zp.float())
            value = self.v[2](zv.float()).squeeze(-1)
        return logits.masked_fill(self.undo[prev], MASKED), value

    @staticmethod
    def prev_action(idx: torch.Tensor) -> torch.Tensor:
        return idx[..., 40].long() - 960

    def step(self, idx, h, c, start=None, hot=None):
        """One step for a batch: logits [B, 12] with the undo move masked, value [B], h', c'."""
        if start is not None:
            keep = (~start).float().unsqueeze(1)
            h, c = h * keep, c * keep
        x = self.trunk(idx, hot)
        h, c = self.lstm(x, h, c)
        logits, value = self.heads(h, x, self.prev_action(idx))
        return logits, value, h, c

    def sequence(self, idx, start, h0, c0):
        """Replay a rollout for BPTT. idx [T, B, 41], start [T, B] -> logits [T, B, 12], values [T, B]."""
        T, B = idx.shape[:2]
        x = self.trunk(idx.reshape(T * B, -1)).view(T, B, -1)
        h, c, hs = h0, c0, []
        for t in range(T):
            keep = (~start[t]).float().unsqueeze(1)
            h, c = self.lstm(x[t], h * keep, c * keep)
            hs.append(h)
        logits, values = self.heads(torch.stack(hs), x, self.prev_action(idx))
        return logits, values


H_EPS = 1e-3


def h_transform(x: torch.Tensor) -> torch.Tensor:
    """Invertible value compression (Pohlen et al. 2018): the critic predicts h(V)."""
    return torch.sign(x) * (torch.sqrt(x.abs() + 1) - 1) + H_EPS * x


def h_inverse(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (((torch.sqrt(1 + 4 * H_EPS * (y.abs() + 1 + H_EPS)) - 1) / (2 * H_EPS)) ** 2 - 1)


def categorical(logits: torch.Tensor):
    """log-probabilities and entropy of masked logits (fp32)."""
    logp = F.log_softmax(logits.float(), -1)
    ent = -(logp.exp() * logp).sum(-1)
    return logp, ent


def build_policy(config: dict | None = None) -> CubePolicy:
    return CubePolicy(**(config or {}))


def load_policy(path: str, device="cpu") -> tuple[CubePolicy, dict]:
    """Load a checkpoint or snapshot written by the trainer. Returns (policy in eval mode, metadata)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    pol = build_policy(ck["model_config"])
    pol.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    pol.to(device).eval()
    meta = {k: v for k, v in ck.items() if k not in ("model", "optimizer")}
    return pol, meta


assert NO_MOVE == N_MOVES
