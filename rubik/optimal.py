"""Shortest solutions for shallow cubes, by meet-in-the-middle search on the GPU.

A table holds every cube within `table_depth` quarter turns of solved. A breadth-first search from the scrambled
cube finds the first level that meets the table, which proves the shortest length for cubes within
table_depth + max_forward moves. A reference for the agent's solution lengths; the agent does not use it.
"""
from __future__ import annotations

import os
import threading

import torch

from rubik.cube import N_MOVES, CubeOps

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


class OptimalSolver:
    def __init__(self, device="cpu", table_depth: int = 6):
        self.ops = CubeOps(device)
        self.dev = self.ops.device
        self.table_depth = table_depth
        self.lock = threading.Lock()
        self._build_or_load()

    def _build_or_load(self):
        path = os.path.join(CACHE_DIR, f"optimal_table_d{self.table_depth}.pt")
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu")
            self.keys, self.dist = d["keys"].to(self.dev), d["dist"].to(self.dev)
            return
        ops = self.ops
        frontier = ops.solved.unsqueeze(0)
        keys, dists = [ops.hash(frontier)], [torch.zeros(1, dtype=torch.uint8, device=self.dev)]
        seen = keys[0]
        for d in range(1, self.table_depth + 1):
            child = self._expand(frontier)
            h = ops.hash(child)
            h, first = _unique_first(h)
            child = child[first]
            idx = torch.searchsorted(seen, h).clamp(max=seen.numel() - 1)
            new = seen[idx] != h
            frontier, h = child[new], h[new]
            keys.append(h)
            dists.append(torch.full((h.numel(),), d, dtype=torch.uint8, device=self.dev))
            seen = torch.sort(torch.cat([seen, h])).values
        k, dd = torch.cat(keys), torch.cat(dists)
        order = torch.argsort(k)
        self.keys, self.dist = k[order], dd[order]
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(dict(keys=self.keys.cpu(), dist=self.dist.cpu()), path + ".tmp")
        os.replace(path + ".tmp", path)

    def _expand(self, states):
        return self.ops.expand_all(states)

    def table_distance(self, states) -> torch.Tensor:
        """distance of each cube if it is in the table, else -1"""
        h = self.ops.hash(states)
        i = torch.searchsorted(self.keys, h).clamp(max=self.keys.numel() - 1)
        hit = self.keys[i] == h
        return torch.where(hit, self.dist[i].long(), torch.full_like(i, -1))

    @torch.no_grad()
    def distance(self, state, max_forward: int = 7) -> int | None:
        """Optimal number of quarter turns, or None if it exceeds table_depth + max_forward."""
        sol = self.solve(state, max_forward)
        return None if sol is None else len(sol)

    @torch.no_grad()
    def solve(self, state, max_forward: int = 7) -> list[int] | None:
        """An optimal move sequence, or None if the cube is deeper than table_depth + max_forward."""
        with self.lock:
            st = torch.as_tensor(state, dtype=torch.uint8, device=self.dev).view(1, 54)
            levels, parents, acts = [st], [], []
            best, best_at = None, None
            f = 0
            while True:
                td = self.table_distance(levels[f])
                ok = td >= 0
                if bool(ok.any()):
                    b = int(td[ok].min())
                    if best is None or f + b < best:
                        best = f + b
                        best_at = (f, int(torch.nonzero(ok & (td == b))[0, 0]))
                if best is not None:
                    break
                if f >= max_forward:
                    break
                child = self._expand(levels[f])
                h = self.ops.hash(child)
                h, first = _unique_first(h)
                levels.append(child[first])
                parents.append(first // N_MOVES)
                acts.append(first % N_MOVES)
                f += 1
            if best is None:
                return None
            f, i = best_at
            fwd = []
            for lv in range(f, 0, -1):
                fwd.append(int(acts[lv - 1][i]))
                i = int(parents[lv - 1][i])
            fwd.reverse()
            cur = levels[f][int(best_at[1])].unsqueeze(0)
            back = []
            d = int(self.table_distance(cur)[0])
            while d > 0:
                nb = self._expand(cur)
                nd = self.table_distance(nb)
                a = int(torch.nonzero(nd == d - 1)[0, 0])
                back.append(a)
                cur, d = nb[a:a + 1], d - 1
            return fwd + back


def _unique_first(h: torch.Tensor):
    """unique values of h (sorted) and the index of the first occurrence of each"""
    u, inv = torch.unique(h, return_inverse=True)
    first = torch.full((u.numel(),), h.numel(), dtype=torch.long, device=h.device)
    first.scatter_reduce_(0, inv, torch.arange(h.numel(), device=h.device), reduce="amin")
    return u, first
