"""Measure an agent on fixed test cubes: the policy alone, no search.

    python -m rubik.benchmark [--agent agent] [--quick] [--curve] [--papers]

On held-out cubes (seed 4321; training evaluates on seed 1234):
  * solve rate and solution length for scramble depths 1..30 and for uniformly random cubes;
  * the agent's own estimate of the moves left, against the moves it then needs;
  * solution length against proven shortest solutions (rubik/optimal.py) for depths <= 12;
  * --curve: every snapshot of the training on random cubes (the learning curve);
  * --papers: the 1,000 test cubes of the DeepCubeA paper (downloaded once).
Writes <agent>/benchmark.json, which the app's Results view shows. Parts that are not measured again are kept.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from rubik.agent import Agent, snapshot_path, snapshots
from rubik.cube import NO_MOVE, CubeOps
from rubik.evaluate import RANDOM, evaluate, make_eval_set, rollout
from rubik.optimal import OptimalSolver
from rubik.policy import h_inverse, load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SEED = 4321
DEPTHS = list(range(1, 31))
PUBLISHED = [
    dict(name="DeepCubeA", moves=21.50, note="value network + A* search"),
    dict(name="EfficientCube", moves=21.26, note="beam search, width 262,144"),
    dict(name="CayleyPy", moves=20.669, note="26 networks + beam search"),
]


@torch.no_grad()
def estimated_moves(ag: Agent, states: torch.Tensor) -> np.ndarray:
    """The agent's own estimate of the moves left, before the first move (NaN where it expects more than ~100)."""
    st = states.to(ag.device)
    n = st.shape[0]
    h, c = ag.policy.initial_state(n, ag.device)
    prev = torch.full((n,), NO_MOVE, dtype=torch.long, device=ag.device)
    with ag._amp():
        _, v, _, _ = ag.policy.step(ag.ops.features(st, prev), h, c)
    values = h_inverse(v.float().reshape(-1)).cpu().numpy()
    return np.array([np.nan if t is None else t for t in (ag.moves_to_go(float(x)) for x in values)])


def measure_depths(ag: Agent, quick: bool) -> dict:
    per_depth = 64 if quick else 256
    es = make_eval_set(per_depth=per_depth, n_random=per_depth, seed=TEST_SEED)
    keys = list(es)
    cubes = torch.cat([es[k] for k in keys])
    solved, length = rollout(ag.policy, ag.ops, cubes, limit=100)
    solved, length, guess = solved.numpy(), length.numpy(), estimated_moves(ag, cubes)
    depths, estimate, i = {}, {}, 0
    for k in keys:
        s, ln, g = solved[i:i + per_depth], length[i:i + per_depth], guess[i:i + per_depth]
        i += per_depth
        depths[str(k)] = dict(solved=float(s.mean()), mean_len=float(ln[s].mean()) if s.any() else None)
        ok = s & ~np.isnan(g)
        if ok.any():
            estimate[str(k)] = dict(estimate=float(g[ok].mean()), needed=float(ln[ok].mean()))
    solver = OptimalSolver(ag.device)
    shortest = {}
    for k in range(1, 13):
        cubes = es[k][:16 if quick else 48]
        solved, length = rollout(ag.policy, ag.ops, cubes, limit=100)
        pairs = []
        for i in range(cubes.shape[0]):
            best = solver.solve(cubes[i].numpy(), max_forward=6)
            if best is not None and bool(solved[i]):
                pairs.append((len(best), int(length[i])))
        if pairs:
            shortest[str(k)] = dict(shortest=float(np.mean([o for o, _ in pairs])), agent=float(np.mean([g for _, g in pairs])),
                                    found=float(np.mean([g == o for o, g in pairs])))
    return dict(per_depth=per_depth, depths=depths, estimate=estimate, shortest=shortest)


def measure_curve(folder: str, device, cubes: int) -> dict:
    ops = CubeOps(device)
    es = make_eval_set(per_depth=1, depths=[1], n_random=cubes, seed=TEST_SEED)
    points = []
    for it in snapshots(folder):
        policy, meta = load_policy(snapshot_path(folder, it), ops.device)
        res = evaluate(policy, ops, {RANDOM: es[RANDOM]}, limit=100)[RANDOM]
        points.append(dict(samples=int(meta.get("samples") or 0), solved=res["solved"], mean_len=res["mean_len"]))
        print(f"{points[-1]['samples'] / 1e6:7.0f} M moves: {100 * res['solved']:5.1f} % of random cubes", flush=True)
    return dict(cubes=cubes, snapshots=points)


def measure_papers(ag: Agent) -> dict:
    from rubik.deepcubea import load_test_set
    ts = load_test_set()
    states, shortest = torch.as_tensor(ts["states"]), torch.as_tensor(ts["optimal_len"])
    solved, length = rollout(ag.policy, ag.ops, states, limit=100)
    assert not bool((solved & (length < shortest)).any()), "a solution shorter than the shortest one: the conversion is off"
    return dict(cubes=len(states), shortest=float(shortest.double().mean()), solved=float(solved.double().mean()),
                mean_len=float(length[solved].double().mean()) if bool(solved.any()) else None, published=PUBLISHED)


def page(o: dict, stages: list) -> dict:
    """The Results view: a few big numbers, four curves and one bar chart."""
    depths, curve, papers = o["depths"], o.get("curve") or {}, o.get("papers") or {}
    rnd = depths[RANDOM]
    cubes, solved, mean_len = o["per_depth"], rnd["solved"], rnd["mean_len"]
    points = curve.get("snapshots") or []
    if points and points[-1]["samples"] == o.get("samples"):
        cubes, solved, mean_len = curve["cubes"], points[-1]["solved"], points[-1]["mean_len"]
    tiles = [
        dict(value=f"{100 * solved:.0f} %", label=f"of {cubes:,} random cubes solved", hint="fully scrambled test cubes, at most 100 moves each"),
        dict(value="—" if mean_len is None else f"{mean_len:.0f}", label="moves per cube", hint="quarter turns, mean over the cubes it solved"),
        dict(value=f"{sum(depths[str(k)]['solved'] >= 0.9 for k in DEPTHS)} / 30", label="scramble depths solved",
             hint=f"depths 1 to 30 with at least 90 % of {o['per_depth']} test cubes solved"),
    ]
    if papers:
        tiles.append(dict(value=f"{100 * papers['solved']:.0f} %", label=f"of the {papers['cubes']:,} DeepCubeA test cubes",
                          hint="the public test set of the DeepCubeA paper"))
    tiles.append(dict(value="0", label="search", hint="one pass through the network per move, no lookahead"))
    curves = []
    if points:
        marks = [dict(x=s["from_samples"] / 1e6, label=f"Stage {i + 1}") for i, s in enumerate(stages) if s["from_samples"] > 0]
        curves.append(dict(title="Learning curve", sub=f"{curve['cubes']:,} random test cubes solved, as training goes on",
                           xName="after", xMoves=True, pct=True, yMin=0, yMax=100, marks=marks,
                           series=[dict(name="solved", pts=[[p["samples"] / 1e6, 100 * p["solved"]] for p in points])]))
    curves.append(dict(title="Solution length", sub="mean moves by scramble depth, against the shortest possible", xName="depth", yMin=0,
                       series=[dict(name="scramble length", pts=[[k, k] for k in DEPTHS], dash=[2, 4]),
                               dict(name="this agent", pts=[[k, depths[str(k)]["mean_len"]] for k in DEPTHS if depths[str(k)]["mean_len"]]),
                               dict(name="shortest possible", pts=[[int(k), v["shortest"]] for k, v in o["shortest"].items()])]))
    if any("found" in v for v in o["shortest"].values()):
        curves.append(dict(title="Shortest solutions", sub="how often it finds a shortest solution, by scramble depth", xName="depth",
                           pct=True, yMin=0, yMax=100,
                           series=[dict(name="found a shortest one", pts=[[int(k), 100 * v["found"]] for k, v in o["shortest"].items() if "found" in v])]))
    est = o.get("estimate") or {}
    dk = [k for k in DEPTHS if str(k) in est]
    if dk:
        curves.append(dict(title="Its own estimate", sub="moves left as it guesses before the first move, against the moves it needs",
                           xName="depth", yMin=0,
                           series=[dict(name="moves needed", pts=[[k, est[str(k)]["needed"]] for k in dk]),
                                   dict(name="its estimate", pts=[[k, est[str(k)]["estimate"]] for k in dk])]))
    bars = None
    if papers and papers.get("mean_len"):
        rows = [dict(name="shortest possible", value=papers["shortest"], note="")]
        rows += [dict(name=p["name"], value=p["moves"], note=p["note"]) for p in sorted(papers["published"], key=lambda p: p["moves"])]
        rows.append(dict(name="this agent", value=papers["mean_len"], note="one network, no search", own=True))
        bars = dict(title="On the papers' test set", rows=rows,
                    sub=f"mean moves per cube on the {papers['cubes']:,} DeepCubeA test cubes; the others add a search, this agent does not")
    return dict(tiles=tiles, curves=curves, bars=bars)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default=os.path.join(ROOT, "agent"), help="agent folder")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="fewer cubes")
    p.add_argument("--curve", action="store_true", help="also measure every snapshot (the learning curve)")
    p.add_argument("--curve-cubes", type=int, default=1000)
    p.add_argument("--papers", action="store_true", help="also measure the DeepCubeA test set")
    p.add_argument("--rerender", action="store_true", help="only rebuild the page from the saved numbers")
    a = p.parse_args()
    path = os.path.join(a.agent, "benchmark.json")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
    if not a.rerender:
        torch.backends.cuda.matmul.allow_tf32 = True
        ag = Agent(os.path.join(a.agent, "policy.pt"), a.device)
        if out.get("samples") != ag.meta.get("samples"):
            out = {}
        out.update(samples=ag.meta.get("samples"), it=ag.meta.get("it"), **measure_depths(ag, a.quick))
        if a.curve:
            out["curve"] = measure_curve(a.agent, a.device, a.curve_cubes)
        if a.papers:
            out["papers"] = measure_papers(ag)
    stages = []
    if os.path.exists(os.path.join(a.agent, "config.json")):
        with open(os.path.join(a.agent, "config.json")) as f:
            stages = json.load(f)["stages"]
    out["page"] = page(out, stages)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for t in out["page"]["tiles"]:
        print(f"{t['value']:>14}  {t['label']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
