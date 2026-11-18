"""Vectorised cube-solving environment on the GPU, with a scramble-depth curriculum.

Reward: -1 per move; solving ends the episode. An episode that reaches its move limit gets one extra
bootstrap step, so the critic's value of the last cube stands in for the rest of the return.
"""
from __future__ import annotations

import numpy as np
import torch

from rubik.cube import NO_MOVE, CubeOps


def time_limit(depth: torch.Tensor, knee: int = 10, slope: int = 0, cap: int = 100) -> torch.Tensor:
    """Moves allowed for a scramble of `depth` quarter turns: 2 * depth + 4, plus `slope` per depth beyond `knee`, up to `cap`."""
    classic = 2 * depth + 4
    if slope <= 0:
        return classic
    longer = (classic + slope * (depth - knee).clamp(min=0)).clamp(max=cap)
    return torch.maximum(classic, longer)


class Curriculum:
    """Scramble-depth distribution that follows the agent: most episodes near the deepest depth it solves, some on all depths."""

    def __init__(self, max_depth: int = 30, threshold: float = 0.5, decay: float = 0.9,
                 explore: float = 0.15, focus: float = 3.0, limits: np.ndarray | None = None):
        self.max_depth, self.threshold, self.decay = max_depth, threshold, decay
        self.explore, self.focus = explore, focus
        self.limits = (2.0 * np.arange(max_depth + 1) + 4) if limits is None else np.asarray(limits, dtype=float)
        self.solved = np.zeros(max_depth + 1)
        self.total = np.zeros(max_depth + 1)

    @property
    def rate(self) -> np.ndarray:
        return np.where(self.total > 0, self.solved / np.maximum(self.total, 1e-9), 0.0)

    @property
    def frontier(self) -> int:
        d = 0
        for k in range(1, self.max_depth + 1):
            if self.total[k] < 20 or self.rate[k] < self.threshold:
                break
            d = k
        return d

    def expected_length(self) -> np.ndarray:
        """Expected episode length per depth (solved ~1.3k moves, unsolved: the time limit plus the bootstrap step)."""
        k = np.arange(self.max_depth + 1, dtype=float)
        r = self.rate
        E = r * 1.3 * k + (1 - r) * (self.limits + 1)
        E[0] = 1.0
        return np.maximum(E, 1.0)

    def probs(self) -> np.ndarray:
        D, K = self.frontier, self.max_depth
        hi = min(D + 2, K)
        q = np.zeros(K + 1)
        q[1:hi + 1] = 1.0
        q[max(1, D - 2):hi + 1] = self.focus
        q /= q.sum()
        E = self.expected_length()
        u = np.zeros(K + 1)
        u[1:] = 1.0 / E[1:]
        u /= u.sum()
        e, LF, LU = self.explore, float((q * E).sum()), float((u * E).sum())
        w = e * LF / (e * LF + (1 - e) * LU) if e > 0 else 0.0
        return (1 - w) * q + w * u

    def update(self, solved: np.ndarray, total: np.ndarray):
        self.solved = self.decay * self.solved + solved
        self.total = self.decay * self.total + total

    def state_dict(self):
        return dict(solved=self.solved.tolist(), total=self.total.tolist())

    def load_state_dict(self, d):
        self.solved, self.total = np.array(d["solved"]), np.array(d["total"])


class CubeEnv:
    """n cubes stepping together, reset from a shared pool of pre-scrambled starts."""

    def __init__(self, ops: CubeOps, n: int, pool_size: int, generator: torch.Generator | None = None,
                 limit: tuple = ()):
        self.ops, self.n, self.P, self.gen = ops, n, pool_size, generator
        self.limit_args = tuple(limit)
        d = ops.device
        self.state = ops.solved.expand(n, 54).clone()
        self.prev = torch.full((n,), NO_MOVE, dtype=torch.long, device=d)
        self.depth = torch.zeros(n, dtype=torch.long, device=d)
        self.t = torch.zeros(n, dtype=torch.long, device=d)
        self.limit = torch.ones(n, dtype=torch.long, device=d)
        self.boot = torch.zeros(n, dtype=torch.bool, device=d)
        self.ptr = torch.zeros((), dtype=torch.long, device=d)
        self.pool_state = self.pool_depth = None

    def refill(self, probs: np.ndarray):
        p = torch.as_tensor(probs, dtype=torch.float, device=self.ops.device)
        self.pool_depth = torch.multinomial(p, self.P, replacement=True, generator=self.gen)
        self.pool_state = self.ops.scramble(self.pool_depth, generator=self.gen)
        self.ptr.zero_()

    def _take(self, mask: torch.Tensor):
        """Next pool entries for the envs in `mask` (consecutive, wrapping around); garbage elsewhere."""
        rank = torch.cumsum(mask.long(), 0) - 1
        i = (self.ptr + rank) % self.P
        self.ptr = self.ptr + mask.long().sum()
        return self.pool_state[i], self.pool_depth[i]

    def reset_all(self):
        self.state, self.depth = self._take(torch.ones(self.n, dtype=torch.bool, device=self.ops.device))
        self.limit = time_limit(self.depth, *self.limit_args)
        self.t.zero_()
        self.prev.fill_(NO_MOVE)
        self.boot.zero_()

    def observe(self) -> torch.Tensor:
        return self.ops.features(self.state, self.prev)

    def step(self, actions: torch.Tensor) -> dict:
        boot = self.boot
        nxt = self.ops.apply(self.state, actions)
        solved = self.ops.is_solved(nxt) & ~boot
        t1 = self.t + 1
        truncated = ~solved & ~boot & (t1 >= self.limit)
        reward = torch.where(boot, 0.0, -1.0)
        reset = boot | solved
        out = dict(reward=reward, terminal=solved, truncated=truncated, invalid=boot,
                   ended=solved | truncated, ended_depth=self.depth, ended_len=t1, start_next=reset)
        st, dp = self._take(reset)
        self.state = torch.where(reset.unsqueeze(1), st, nxt)
        self.prev = torch.where(reset, torch.full_like(actions, NO_MOVE), actions)
        self.depth = torch.where(reset, dp, self.depth)
        self.limit = torch.where(reset, time_limit(dp, *self.limit_args), self.limit)
        self.t = torch.where(reset, torch.zeros_like(t1), t1)
        self.boot = truncated
        return out
