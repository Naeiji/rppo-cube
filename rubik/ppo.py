"""Recurrent PPO (LSTM actor-critic, BPTT) for the cube, entirely on the GPU.

One iteration: a rollout of `rollout` moves on `n_envs` cubes, GAE, a few PPO epochs over minibatches of whole
env sequences (the LSTM is replayed from its stored state), then the curriculum moves the scramble depth.

Files in the output folder: policy.pt (the agent), checkpoint.pt (to resume), snapshots/, config.json,
metrics.jsonl, eval.jsonl.
"""
from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass

import torch

from rubik.cube import CubeOps
from rubik.env import CubeEnv, Curriculum, time_limit
from rubik.evaluate import RANDOM, evaluate, make_eval_set
from rubik.policy import CubePolicy, categorical, h_inverse, h_transform

NOT_SETTINGS = ("out", "eval_every", "snapshot_every", "eval_per_depth", "seed", "device", "amp")


@dataclass
class PPOConfig:
    out: str = "my_agent"
    n_envs: int = 16384
    rollout: int = 16
    pool_factor: int = 4
    total_samples: float = 2e9
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    lr_end: float = 3e-5
    epochs: int = 2
    minibatches: int = 8
    clip: float = 0.2
    ent_coef: float = 0.01
    ent_coef_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.05
    max_depth: int = 30
    cur_threshold: float = 0.5
    cur_explore: float = 0.15
    limit_slope: int = 0
    limit_knee: int = 10
    limit_cap: int = 100
    d_model: int = 1024
    n_res: int = 2
    hidden: int = 512
    head: int = 512
    eval_every: float = 2.5e7
    snapshot_every: float = 5e7
    eval_per_depth: int = 256
    seed: int = 0
    device: str = ""
    amp: bool = True


def _lin(a, b, frac):
    return a + (b - a) * min(max(frac, 0.0), 1.0)


class Trainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        if os.path.exists(os.path.join(cfg.out, "policy.pt")) and not os.path.exists(os.path.join(cfg.out, "checkpoint.pt")):
            raise SystemExit(f"{cfg.out} holds a trained agent and no checkpoint to resume from: choose another --out")
        os.makedirs(os.path.join(cfg.out, "snapshots"), exist_ok=True)
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.ops = CubeOps(self.device)
        self.policy = CubePolicy(cfg.d_model, cfg.n_res, cfg.hidden, cfg.head).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        limit = (cfg.limit_knee, cfg.limit_slope, cfg.limit_cap)
        self.cur = Curriculum(cfg.max_depth, cfg.cur_threshold, explore=cfg.cur_explore,
                              limits=time_limit(torch.arange(cfg.max_depth + 1), *limit).numpy())
        self.env = CubeEnv(self.ops, cfg.n_envs, cfg.n_envs * cfg.pool_factor, self.gen, limit=limit)
        self.it, self.samples = 0, 0
        self.eval_set = make_eval_set(cfg.eval_per_depth)
        if os.path.exists(self.path("checkpoint.pt")):
            self._resume()
            print(f"resumed at iteration {self.it} ({self.samples / 1e6:.1f} M moves)", flush=True)
        self.h, self.c = self.policy.initial_state(cfg.n_envs, self.device)
        self.start = torch.ones(cfg.n_envs, dtype=torch.bool, device=self.device)
        self._write_config()

    def path(self, name):
        return os.path.join(self.cfg.out, name)

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16,
                              enabled=self.cfg.amp and self.device.type == "cuda")

    def _write_config(self):
        """config.json lists the stages of the training: a new entry whenever a resume changes the settings."""
        settings = {k: v for k, v in asdict(self.cfg).items() if k not in NOT_SETTINGS}
        stages = []
        if os.path.exists(self.path("config.json")):
            with open(self.path("config.json")) as f:
                stages = [s for s in json.load(f)["stages"] if s["from_samples"] < self.samples]
        if not stages or stages[-1]["settings"] != settings:
            stages.append(dict(from_samples=self.samples, settings=settings))
        with open(self.path("config.json"), "w") as f:
            json.dump(dict(stages=stages), f, indent=1)

    def _resume(self):
        ck = torch.load(self.path("checkpoint.pt"), map_location="cpu", weights_only=True)
        if ck["model_config"] != self.policy.config:
            raise ValueError(f"model shape {ck['model_config']} in the checkpoint != settings {self.policy.config}")
        self.policy.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.cur.load_state_dict(ck["curriculum"])
        self.it, self.samples = ck["it"], ck["samples"]
        if ck["rng_device"] == self.device.type:
            self.gen.set_state(ck["rng"])
        for name in ("metrics.jsonl", "eval.jsonl"):
            p = self.path(name)
            if os.path.exists(p):
                keep = [line for line in open(p) if line.strip() and json.loads(line)["it"] <= self.it]
                with open(p + ".tmp", "w") as f:
                    f.writelines(keep)
                os.replace(p + ".tmp", p)
        for f in os.listdir(self.path("snapshots")):
            if f.startswith("it") and f.endswith(".pt") and int(f[2:8]) > self.it:
                os.remove(os.path.join(self.path("snapshots"), f))

    def _save_policy(self, path, half=False):
        weights = {k: (v.half() if half and v.is_floating_point() else v).detach().cpu() for k, v in self.policy.state_dict().items()}
        torch.save(dict(model=weights, model_config=self.policy.config, gamma=self.cfg.gamma, it=self.it, samples=self.samples),
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    def save(self):
        """checkpoint.pt (everything needed to resume) and policy.pt (the agent as it is now)."""
        ck = dict(model={k: v.detach().cpu() for k, v in self.policy.state_dict().items()}, model_config=self.policy.config,
                  optimizer=self.opt.state_dict(), curriculum=self.cur.state_dict(), it=self.it, samples=self.samples,
                  rng=self.gen.get_state(), rng_device=self.device.type)
        torch.save(ck, self.path("checkpoint.pt") + ".tmp")
        os.replace(self.path("checkpoint.pt") + ".tmp", self.path("checkpoint.pt"))
        self._save_policy(self.path("policy.pt"))

    def snapshot(self):
        self._save_policy(self.path(f"snapshots/it{self.it:06d}.pt"), half=True)

    @torch.no_grad()
    def rollout(self):
        cfg, env, pol = self.cfg, self.env, self.policy
        T, N, dev = cfg.rollout, cfg.n_envs, self.device
        b = dict(obs=torch.empty(T, N, 41, dtype=torch.int16, device=dev),
                 act=torch.empty(T, N, dtype=torch.long, device=dev),
                 logp=torch.empty(T, N, device=dev), val=torch.empty(T, N, device=dev),
                 rew=torch.empty(T, N, device=dev),
                 term=torch.empty(T, N, dtype=torch.bool, device=dev),
                 trunc=torch.empty(T, N, dtype=torch.bool, device=dev),
                 invalid=torch.empty(T, N, dtype=torch.bool, device=dev),
                 start=torch.empty(T, N, dtype=torch.bool, device=dev),
                 h0=self.h.clone(), c0=self.c.clone())
        K = cfg.max_depth + 1
        ep_total = torch.zeros(K, device=dev)
        ep_solved = torch.zeros(K, device=dev)
        ep_len = torch.zeros((), device=dev)
        h, c, start = self.h, self.c, self.start
        for t in range(T):
            idx = env.observe()
            with self.autocast():
                logits, v, h, c = pol.step(idx, h, c, start)
            u = torch.rand(logits.shape, device=dev, generator=self.gen).clamp_(1e-9, 1 - 1e-7)
            a = (logits - torch.log(-torch.log(u))).argmax(-1)
            logp, _ = categorical(logits)
            b["obs"][t] = idx
            b["act"][t] = a
            b["logp"][t] = logp.gather(1, a.unsqueeze(1)).squeeze(1)
            b["val"][t] = h_inverse(v)
            b["start"][t] = start
            out = env.step(a)
            b["rew"][t], b["term"][t] = out["reward"], out["terminal"]
            b["trunc"][t], b["invalid"][t] = out["truncated"], out["invalid"]
            ep_total.index_add_(0, out["ended_depth"], out["ended"].float())
            ep_solved.index_add_(0, out["ended_depth"], out["terminal"].float())
            ep_len += (out["ended_len"] * out["terminal"]).sum()
            start = out["start_next"]
        with self.autocast():
            _, v_last, _, _ = pol.step(env.observe(), h, c, start)
        b["v_last"] = h_inverse(v_last)
        self.h, self.c, self.start = h, c, start
        return b, dict(total=ep_total.cpu().numpy(), solved=ep_solved.cpu().numpy(), len_sum=float(ep_len))

    @torch.no_grad()
    def gae(self, b):
        cfg = self.cfg
        T = cfg.rollout
        adv = torch.zeros_like(b["rew"])
        last = torch.zeros_like(b["v_last"])
        v_next = torch.cat([b["val"][1:], b["v_last"].unsqueeze(0)])
        for t in reversed(range(T)):
            nonterm = (~b["term"][t]).float()
            cont = (~(b["term"][t] | b["trunc"][t] | b["invalid"][t])).float()
            delta = b["rew"][t] + cfg.gamma * v_next[t] * nonterm - b["val"][t]
            last = delta + cfg.gamma * cfg.lam * cont * last
            adv[t] = last
        return adv, adv + b["val"]

    def update(self, b, adv, ret, ent_coef):
        cfg, pol = self.cfg, self.policy
        N, MB = cfg.n_envs, cfg.n_envs // cfg.minibatches
        valid = ~b["invalid"]
        a_valid = adv[valid]
        adv = (adv - a_valid.mean()) / (a_valid.std() + 1e-8)
        zero = torch.zeros((), device=self.device)
        sums = dict(pg=zero.clone(), v=zero.clone(), ent=zero.clone(), kl=zero.clone(), clip=zero.clone())
        n_upd, epochs_done = 0, 0
        for epoch in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device, generator=self.gen)
            kl_epoch = []
            for k in range(cfg.minibatches):
                ids = perm[k * MB:(k + 1) * MB]
                m = valid[:, ids].float()
                msum = m.sum().clamp(min=1.0)
                with self.autocast():
                    logits, values = pol.sequence(b["obs"][:, ids].long(), b["start"][:, ids], b["h0"][ids], b["c0"][ids])
                logp_all, ent = categorical(logits)
                lp = logp_all.gather(-1, b["act"][:, ids].unsqueeze(-1)).squeeze(-1)
                log_ratio = lp - b["logp"][:, ids]
                ratio = log_ratio.exp()
                A = adv[:, ids]
                pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
                pg_loss = (pg * m).sum() / msum
                v_loss = (((values - h_transform(ret[:, ids])) ** 2) * m).sum() / msum
                ent_mean = (ent * m).sum() / msum
                loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * ent_mean
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (((ratio - 1) - log_ratio) * m).sum() / msum
                    clipped = (((ratio - 1).abs() > cfg.clip).float() * m).sum() / msum
                kl_epoch.append(kl)
                if n_upd == 0:
                    kl0 = kl
                sums["pg"] += pg_loss.detach()
                sums["v"] += v_loss.detach()
                sums["ent"] += ent_mean.detach()
                sums["kl"] += kl
                sums["clip"] += clipped
                n_upd += 1
            epochs_done += 1
            if float(torch.stack(kl_epoch).mean()) > cfg.target_kl:
                break
        out = {k: float(v) / max(n_upd, 1) for k, v in sums.items()}
        out["kl0"] = float(kl0)
        with torch.no_grad():
            r, v = ret[valid], b["val"][valid]
            out["ev"] = float(1 - (r - v).var() / (r.var() + 1e-8))
        out["epochs"] = epochs_done
        return out

    def run_eval(self):
        res = evaluate(self.policy, self.ops, self.eval_set, limit=100)
        with open(self.path("eval.jsonl"), "a") as f:
            f.write(json.dumps(dict(it=self.it, samples=self.samples, results=res)) + "\n")
        deepest = 0
        for k in range(1, self.cfg.max_depth + 1):
            if res[str(k)]["solved"] < 0.5:
                break
            deepest = k
        print(f"  eval at {self.samples / 1e6:.0f} M moves: at least half solved up to depth {deepest} | "
              f"random cubes {100 * res[RANDOM]['solved']:.1f} %", flush=True)

    def train(self):
        cfg = self.cfg
        stop = {"flag": False}

        def _stop(signum, frame):
            stop["flag"] = True
            print(f"signal {signum}: stopping after this iteration", flush=True)

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        if self.it == 0:
            self.run_eval()
            self.snapshot()
        evaluated = self.it
        next_eval = (self.samples // cfg.eval_every + 1) * cfg.eval_every
        next_snap = (self.samples // cfg.snapshot_every + 1) * cfg.snapshot_every
        self.env.refill(self.cur.probs())
        self.env.reset_all()
        while self.samples < cfg.total_samples and not stop["flag"]:
            t0 = time.time()
            frac = self.samples / cfg.total_samples
            lr = _lin(cfg.lr, cfg.lr_end, frac)
            for g in self.opt.param_groups:
                g["lr"] = lr
            ent_coef = _lin(cfg.ent_coef, cfg.ent_coef_end, frac)
            self.env.refill(self.cur.probs())
            b, ep = self.rollout()
            adv, ret = self.gae(b)
            st = self.update(b, adv, ret, ent_coef)
            del b, adv, ret
            self.cur.update(ep["solved"], ep["total"])
            n_new = cfg.n_envs * cfg.rollout
            self.it += 1
            self.samples += n_new
            n_ep, n_sol = float(ep["total"].sum()), float(ep["solved"].sum())
            rec = dict(it=self.it, samples=self.samples, lr=lr, ent_coef=ent_coef, episodes=n_ep,
                       solved_frac=n_sol / max(n_ep, 1), solve_len=ep["len_sum"] / max(n_sol, 1), frontier=self.cur.frontier, **st)
            with open(self.path("metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.it % 10 == 1:
                print(f"it {self.it} | {self.samples / 1e6:.0f} M moves | {n_new / (time.time() - t0) / 1e3:.0f} k/s | "
                      f"frontier {self.cur.frontier} | solved {rec['solved_frac']:.2f} in {rec['solve_len']:.1f} | "
                      f"ent {st['ent']:.2f} kl {st['kl']:.4f} clip {st['clip']:.2f} ev {st['ev']:.2f} v {st['v']:.3f}", flush=True)
            if self.samples >= next_eval:
                self.run_eval()
                self.save()
                evaluated = self.it
                next_eval += cfg.eval_every
            if self.samples >= next_snap:
                self.snapshot()
                next_snap += cfg.snapshot_every
        if evaluated != self.it:
            self.run_eval()
        self.snapshot()
        self.save()
        print(f"{'stopped' if stop['flag'] else 'finished'} at iteration {self.it}, {self.samples / 1e6:.0f} M moves", flush=True)
