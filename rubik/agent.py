"""Interactive use of a trained policy: traced solves, hints and the thought map for the web app."""
from __future__ import annotations

import hashlib
import math
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from rubik import cube as C
from rubik.cube import NO_MOVE, CubeOps
from rubik.policy import h_inverse, load_policy

SLOT_OF_STICKER = np.full(54, -1, np.int64)
for _j, _stickers in enumerate(list(C.CORNERS) + list(C.EDGES)):
    SLOT_OF_STICKER[np.asarray(_stickers)] = _j
SLOT_STICKERS = [list(map(int, s)) for s in list(C.CORNERS) + list(C.EDGES)]

MAP_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "maps")
MAP_CACHE_KEEP = 40


def snapshots(folder: str) -> list[int]:
    """Iteration numbers of the snapshots in an agent folder."""
    d = os.path.join(folder, "snapshots")
    names = os.listdir(d) if os.path.isdir(d) else []
    return sorted(int(n[2:8]) for n in names if len(n) == 11 and n.startswith("it") and n.endswith(".pt") and n[2:8].isdigit())


def snapshot_path(folder: str, it: int) -> str:
    return os.path.join(folder, "snapshots", f"it{int(it):06d}.pt")


def _map_cache_path(ckpt: str, version: str, n: int, seed: int) -> str:
    key = hashlib.sha1(f"{os.path.abspath(ckpt)}|{version}|{n}|{seed}".encode()).hexdigest()[:20]
    return os.path.join(MAP_CACHE, key + ".pt")


def _prune_map_cache():
    try:
        files = sorted((os.path.join(MAP_CACHE, f) for f in os.listdir(MAP_CACHE) if f.endswith(".pt")), key=os.path.getmtime)
        for f in files[:-MAP_CACHE_KEEP]:
            os.remove(f)
    except OSError:
        pass


class Agent:
    def __init__(self, path: str, device: str = "cpu"):
        self.path = path
        st = os.stat(path)
        self.version = f"{st.st_mtime_ns:x}-{st.st_size:x}"
        self.device = torch.device(device)
        self.policy, self.meta = load_policy(path, self.device)
        self.policy.requires_grad_(False)
        self.gamma = float(self.meta.get("gamma", 0.99))
        self.ops = CubeOps(self.device)
        self.lock = threading.Lock()
        self._map = None

    def value(self, v: torch.Tensor) -> float:
        """Critic output -> value: minus the discounted number of moves to go."""
        return float(h_inverse(v.detach().float().reshape(-1)[:1])[0])

    def moves_to_go(self, value: float) -> float | None:
        """Invert V = -(1 - g^n) / (1 - g): the number of moves the critic expects to need."""
        x = 1.0 + value * (1.0 - self.gamma)
        if x <= 1e-6:
            return None
        return max(0.0, math.log(x) / math.log(self.gamma))

    def _amp(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def _attention(self, idx, hot_grad, state_np) -> list[float]:
        """|gradient x input| of the chosen move's log-probability, per sticker."""
        g = hot_grad[0, idx[0, :40]].abs().float().cpu().numpy()
        slot = g[:20].copy()
        for j in range(20):
            ref = int(C.REF_STICKERS[j])
            where = int(np.nonzero(state_np == ref)[0][0])
            slot[SLOT_OF_STICKER[where]] += g[20 + j]
        slot /= max(slot.max(), 1e-12)
        out = np.zeros(54)
        for j, st in enumerate(SLOT_STICKERS):
            out[st] = slot[j]
        return [round(float(x), 3) for x in out]

    def trace(self, state, limit: int = 100, attention: bool = True, project: bool = True) -> dict:
        """Play the policy's top move from `state` until solved or `limit` moves, recording every step."""
        with self.lock:
            return self._trace(state, limit, attention, project)

    def _trace(self, state, limit, attention, project):
        pol, ops, dev = self.policy, self.ops, self.device
        st = torch.as_tensor(np.asarray(state, dtype=np.uint8), device=dev).view(1, 54)
        h, c = pol.initial_state(1, dev)
        prev = torch.full((1,), NO_MOVE, dtype=torch.long, device=dev)
        IDX, HIN, CIN, LOG, VAL, HS, CS, STATES, moves = [], [], [], [], [], [], [], [], []
        with torch.no_grad():
            for t in range(limit + 1):
                idx = ops.features(st, prev)
                logits, v, h2, c2 = pol.step(idx, h, c)
                if t == limit or bool(ops.is_solved(st)[0]):
                    final_v, final_h, final_c = v, h2, c2
                    break
                lp = F.log_softmax(logits.float(), -1)
                a = int(lp[0].argmax())
                IDX.append(idx); HIN.append(h); CIN.append(c); LOG.append(lp); VAL.append(v)
                HS.append(h2); CS.append(c2); STATES.append(st)
                moves.append(a)
                at = torch.tensor([a], device=dev)
                st, prev, h, c = ops.apply(st, at), at, h2, c2
        T = len(moves)
        steps = []
        if T:
            logp = torch.cat(LOG).float()
            probs = logp.exp()
            ent = -(probs * logp.clamp(min=-60)).sum(-1)
            H, Cc = torch.cat(HS).float(), torch.cat(CS).float()
            vals = [self.value(v) for v in VAL]
            xy = self.project_many(H) if project else None
            attn = attn_prev = None
            if attention:
                I = torch.cat(IDX)
                hot = pol.one_hot(I).float().requires_grad_(True)
                with torch.enable_grad():
                    lg, _, _, _ = pol.step(I, torch.cat(HIN), torch.cat(CIN), hot=hot)
                    lp = F.log_softmax(lg.float(), -1).gather(1, torch.tensor(moves, device=dev).unsqueeze(1)).sum()
                    lp.backward()
                G, I_cpu = hot.grad.detach().cpu(), I.cpu()
                states_np = torch.cat(STATES).cpu().numpy()
                attn = [self._attention(I_cpu[t:t + 1], G[t:t + 1], states_np[t]) for t in range(T)]
                gmax = G.gather(1, I_cpu[:, :40].long()).abs().amax(1).clamp(min=1e-12)
                attn_prev = (G.gather(1, I_cpu[:, 40:41].long()).abs()[:, 0] / gmax).tolist()
            probs_l, ent_l = probs.cpu().tolist(), ent.cpu().tolist()
            H_l, C_l = H.cpu().numpy().round(3).tolist(), Cc.clamp(-3, 3).cpu().numpy().round(3).tolist()
            for t in range(T):
                info = dict(move=moves[t], probs=[round(p, 4) for p in probs_l[t]], value=round(vals[t], 3),
                            togo=self.moves_to_go(vals[t]), entropy=round(ent_l[t], 4), h=H_l[t], c=C_l[t])
                if xy is not None:
                    info["xy"] = xy[t]
                if attn is not None:
                    info["attn"] = attn[t]
                    info["attn_prev"] = round(attn_prev[t], 3)
                steps.append(info)
        value = self.value(final_v)
        final = dict(value=round(value, 3), togo=self.moves_to_go(value),
                     h=final_h[0].float().cpu().numpy().round(3).tolist(),
                     c=final_c[0].float().clamp(-3, 3).cpu().numpy().round(3).tolist())
        if project:
            final["xy"] = self.project_many(final_h.float())[0]
        return dict(solved=bool(ops.is_solved(st)[0]), moves=moves, notation=C.moves_to_str(moves), steps=steps,
                    final=final, final_state=st[0].cpu().numpy().tolist(), version=self.version)

    @torch.no_grad()
    def thought_map(self, n: int = 384, seed: int = 7) -> dict:
        """PCA of the LSTM state over solves of random scrambles, with the moves still needed at each point."""
        if self._map is not None:
            return self._map
        cache = _map_cache_path(self.path, self.version, n, seed)
        if os.path.exists(cache):
            try:
                m = torch.load(cache, map_location="cpu", weights_only=True)
                m["mean"], m["comps"] = m["mean"].to(self.device), m["comps"].to(self.device)
                self._map = m
                return m
            except Exception:
                pass
        pol, ops, dev = self.policy, self.ops, self.device
        g = torch.Generator(device=dev).manual_seed(seed)
        depths = torch.randint(1, 27, (n,), device=dev, generator=g)
        st = ops.scramble(depths, generator=g)
        h, c = pol.initial_state(n, dev)
        prev = torch.full((n,), NO_MOVE, dtype=torch.long, device=dev)
        solved = ops.is_solved(st)
        H, alive, t_solve = [], [], torch.full((n,), -1, dtype=torch.long, device=dev)
        for t in range(80):
            with self._amp():
                logits, _, h, c = pol.step(ops.features(st, prev), h, c)
            H.append(h.float().clone())
            alive.append(~solved)
            a = logits.argmax(-1)
            st = torch.where(solved.unsqueeze(1), st, ops.apply(st, a))
            prev = a
            newly = ops.is_solved(st) & ~solved
            t_solve = torch.where(newly, torch.full_like(t_solve, t + 1), t_solve)
            solved |= newly
            if bool(solved.all()):
                break
        T = len(H)
        H = torch.stack(H)
        alive = torch.stack(alive)
        tt = torch.arange(T, device=dev).unsqueeze(1).expand(T, n)
        togo = torch.where(t_solve >= 0, t_solve.unsqueeze(0) - tt, torch.full_like(tt, -1))
        X, Y = H[alive], togo[alive]
        mean = X.mean(0)
        _, _, V = torch.linalg.svd(X[torch.randperm(X.shape[0], device=dev, generator=g)[:8000]] - mean, full_matrices=False)
        comps = V[:2]
        xy = (X - mean) @ comps.T
        if float(torch.corrcoef(torch.stack([xy[:, 0], Y.float()]))[0, 1]) < 0:
            comps[0] = -comps[0]
            xy[:, 0] = -xy[:, 0]
        keep = torch.randperm(X.shape[0], device=dev, generator=g)[:3500]
        pts = torch.cat([xy[keep], Y[keep].float().unsqueeze(1)], 1).cpu().numpy()
        lo, hi = xy.min(0).values.cpu().numpy(), xy.max(0).values.cpu().numpy()
        self._map = dict(mean=mean, comps=comps,
                         points=[[round(float(a), 3), round(float(b), 3), int(d)] for a, b, d in pts],
                         lo=lo.tolist(), hi=hi.tolist(), solved_frac=float(solved.float().mean()))
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            torch.save({**self._map, "mean": mean.cpu(), "comps": comps.cpu()}, cache + ".tmp")
            os.replace(cache + ".tmp", cache)
            _prune_map_cache()
        except OSError:
            pass
        return self._map

    def thought_map_public(self) -> dict:
        m = self.thought_map()
        return {**{k: v for k, v in m.items() if k not in ("mean", "comps")}, "version": self.version}

    def project_many(self, H: torch.Tensor) -> list[list[float]]:
        m = self.thought_map()
        xy = (H.to(m["mean"].device).float() - m["mean"]) @ m["comps"].T
        return xy.cpu().numpy().round(3).tolist()
