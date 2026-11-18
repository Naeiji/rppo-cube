"""Batched evaluation of the policy on fixed test sets."""
from __future__ import annotations

import numpy as np
import torch

from rubik.cube import NO_MOVE, CubeOps, random_states
from rubik.policy import CubePolicy

EVAL_DEPTHS = list(range(1, 31))
RANDOM = "random"


def make_eval_set(per_depth: int = 256, depths=EVAL_DEPTHS, n_random: int = 256, seed: int = 1234) -> dict:
    """Fixed, device-independent test cubes: per_depth canonical scrambles per depth + random cubes."""
    ops = CubeOps("cpu")
    g = torch.Generator().manual_seed(seed)
    out = {int(k): ops.scramble(torch.full((per_depth,), int(k)), generator=g) for k in depths}
    if n_random:
        out[RANDOM] = torch.as_tensor(random_states(n_random, np.random.default_rng(seed)))
    return out


@torch.no_grad()
def rollout(policy: CubePolicy, ops: CubeOps, states: torch.Tensor, limit: int = 100, amp: bool = True,
            chunk: int = 16384):
    """Play the policy's top move from each state until solved or `limit` moves. Returns (solved [M], moves used [M])."""
    solved_all, length_all = [], []
    for s0 in states.split(chunk):
        st = s0.to(ops.device)
        m = st.shape[0]
        h, c = policy.initial_state(m, ops.device)
        prev = torch.full((m,), NO_MOVE, dtype=torch.long, device=ops.device)
        solved = ops.is_solved(st)
        length = torch.zeros(m, dtype=torch.long, device=ops.device)
        for t in range(limit):
            with torch.autocast(ops.device.type, dtype=torch.bfloat16, enabled=amp and ops.device.type == "cuda"):
                logits, _, h, c = policy.step(ops.features(st, prev), h, c)
            a = logits.argmax(-1)
            active = ~solved
            st = torch.where(active.unsqueeze(1), ops.apply(st, a), st)
            prev = torch.where(active, a, prev)
            length += active.long()
            solved |= ops.is_solved(st)
            if t % 8 == 7 and bool(solved.all()):
                break
        solved_all.append(solved.cpu())
        length_all.append(length.cpu())
    return torch.cat(solved_all), torch.cat(length_all)


def evaluate(policy: CubePolicy, ops: CubeOps, eval_set: dict, limit: int = 100) -> dict:
    """Per test group: solve rate and mean solution length (over solved cubes)."""
    keys = list(eval_set)
    sizes = [eval_set[k].shape[0] for k in keys]
    solved, length = rollout(policy, ops, torch.cat([eval_set[k] for k in keys]), limit=limit)
    out, i = {}, 0
    for k, n in zip(keys, sizes):
        s, l = solved[i:i + n], length[i:i + n]
        out[str(k)] = dict(solved=float(s.float().mean()),
                           mean_len=float(l[s].float().mean()) if bool(s.any()) else None)
        i += n
    return out
