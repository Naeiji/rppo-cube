"""Flask app for the cube agent.

    python -m rubik.web [--agent agent] [--host 127.0.0.1] [--port 5000] [--device cuda]

One page with four views (Solve, Training, Results, How it works) and a JSON API under /api/.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from rubik import cube as C
from rubik.agent import Agent, snapshot_path, snapshots
from rubik.cube import CubeOps
from rubik.optimal import OptimalSolver

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_HEX = ["#f4f5f7", "#e5333b", "#14b35a", "#ffd23f", "#ff8a1f", "#1f6feb"]
STEPS_MAX = 200
_jsonl = {}


class ApiError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def read_jsonl(path: str) -> list:
    """The records of a .jsonl file, cached until the file changes; a half-written last line is skipped."""
    if not os.path.exists(path):
        return []
    key = (os.path.getmtime(path), os.path.getsize(path))
    if _jsonl.get(path, (None, None))[0] != key:
        records = []
        for line in open(path):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        _jsonl[path] = (key, records)
    return _jsonl[path][1]


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(dict(DEVICE="cuda" if torch.cuda.is_available() else "cpu", AGENT_DIR=os.path.join(ROOT, "agent"), MAX_AGENTS=10))
    app.config.update(config or {})
    device, folder = app.config["DEVICE"], app.config["AGENT_DIR"]
    agents: OrderedDict[str, Agent] = OrderedDict()
    lock = threading.Lock()
    cpu_ops = CubeOps("cpu")
    rng = np.random.default_rng()
    solver = {}

    def get_agent(snapshot=None) -> Agent:
        """The agent, or the agent as it was at a snapshot of its training (an iteration number)."""
        if snapshot is None:
            path = os.path.join(folder, "policy.pt")
        elif isinstance(snapshot, bool) or not isinstance(snapshot, int):
            raise ApiError("snapshot: an iteration number")
        else:
            path = snapshot_path(folder, snapshot)
        if not os.path.exists(path):
            raise ApiError("no trained agent in this folder" if snapshot is None else "unknown snapshot", 404)
        with lock:
            key = f"{path}@{os.path.getmtime(path)}"
            if key not in agents:
                agents[key] = Agent(path, device)
                while len(agents) > app.config["MAX_AGENTS"]:
                    agents.popitem(last=False)
            agents.move_to_end(key)
            return agents[key]

    def body() -> dict:
        d = request.get_json(silent=True)
        if not isinstance(d, dict):
            raise ApiError("expected a JSON object")
        return d

    def get_state(d) -> np.ndarray:
        s = d.get("state")
        if not isinstance(s, list) or len(s) != 54:
            raise ApiError("state: list of 54 sticker ids")
        try:
            arr = np.array([int(x) for x in s], dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            raise ApiError("state: list of 54 sticker ids")
        reason = C.check_solvable(arr) if sorted(arr.tolist()) == list(range(54)) else "not a permutation"
        if reason:
            raise ApiError(f"invalid cube: {reason}")
        return arr.astype(np.uint8)

    def get_int(d, key, default, lo, hi) -> int:
        v = d.get(key, default)
        if isinstance(v, bool):
            raise ApiError(f"{key}: integer expected")
        try:
            v = int(v)
        except (TypeError, ValueError, OverflowError):
            raise ApiError(f"{key}: integer expected")
        if not lo <= v <= hi:
            raise ApiError(f"{key}: must be in [{lo}, {hi}]")
        return v

    def scrambled(depth: int, seed: int) -> list[int]:
        g = torch.Generator().manual_seed(seed)
        _, m = cpu_ops.scramble(torch.tensor([depth]), generator=g, return_moves=True)
        return [int(a) for a in m[0] if a >= 0][:depth]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(move_names=C.MOVE_NAMES, faces=C.FACES, face_hex=FACE_HEX, color_names=C.COLOR_NAMES,
                       sticker_pos=C.STICKER_POS.tolist(), sticker_normal=C.STICKER_NORMAL.tolist(),
                       move_src=C.MOVE_SRC.tolist(), corners=C.CORNERS.tolist(), edges=C.EDGES.tolist(), device=device)

    @app.post("/api/scramble")
    def scramble():
        d = body()
        if d.get("random"):
            st = C.random_states(1, rng)[0]
            return jsonify(state=st.tolist(), moves=None)
        if "moves" in d:
            try:
                mv = C.parse_moves(str(d["moves"]))
            except ValueError as e:
                raise ApiError(str(e))
        else:
            mv = scrambled(get_int(d, "depth", 12, 0, 100), int(rng.integers(2 ** 62)))
        return jsonify(state=C.apply_moves(C.solved_state(), mv).tolist(), moves=mv)

    @app.post("/api/facelets")
    def facelets():
        try:
            st = C.from_facelets(str(body().get("facelets", "")))
        except ValueError as e:
            raise ApiError(str(e))
        return jsonify(state=st.tolist())

    @app.post("/api/solve")
    def solve():
        d = body()
        st = get_state(d)
        ag = get_agent(d.get("snapshot"))
        t0 = time.time()
        tr = ag.trace(st, limit=get_int(d, "limit", 100, 1, STEPS_MAX), attention=bool(d.get("attention", True)))
        tr["secs"] = round(time.time() - t0, 3)
        return jsonify(tr)

    @app.post("/api/hint")
    def hint():
        d = body()
        st = get_state(d)
        ag = get_agent(d.get("snapshot"))
        tr = ag.trace(st, limit=1, attention=True)
        return jsonify(step=tr["steps"][0] if tr["steps"] else None, version=ag.version)

    @app.post("/api/optimal")
    def optimal():
        st = get_state(body())
        with lock:
            if "solver" not in solver:
                solver["solver"] = OptimalSolver(device)
        s = solver["solver"]
        sol = s.solve(st, max_forward=6)
        return jsonify(distance=None if sol is None else len(sol), limit=s.table_depth + 6,
                       notation=None if sol is None else C.moves_to_str(sol))

    @app.get("/api/thoughtmap")
    def thoughtmap():
        ag = get_agent(request.args.get("snapshot", type=int))
        with ag.lock:
            return jsonify(ag.thought_map_public())

    def training_log():
        stages = []
        if os.path.exists(os.path.join(folder, "config.json")):
            with open(os.path.join(folder, "config.json")) as f:
                stages = json.load(f)["stages"]
        return read_jsonl(os.path.join(folder, "metrics.jsonl")), read_jsonl(os.path.join(folder, "eval.jsonl")), stages

    def moves_at(metrics) -> dict:
        at = {m["it"]: m["samples"] for m in metrics}
        at[0] = 0
        return at

    @app.get("/api/training")
    def training():
        """The training of the agent, in training moves: curriculum, evaluations, stages and snapshots."""
        metrics, evals, stages = training_log()
        if not metrics or not stages:
            return jsonify(available=False)
        last, settings = metrics[-1], stages[-1]["settings"]
        step = max(1, len(metrics) // 600)
        thin = metrics[::step] + ([last] if (len(metrics) - 1) % step else [])
        at = moves_at(metrics)

        def stage(s):
            st = s["settings"]
            if st.get("limit_slope", 0) > 0:
                return dict(from_samples=s["from_samples"], label=f"up to {st['limit_cap']} moves per cube",
                            hint=f"{st['limit_slope']} more moves for every scramble move beyond {st['limit_knee']}")
            return dict(from_samples=s["from_samples"], label="tight move limit", hint="at most 2k+4 moves for a cube scrambled with k moves")

        return jsonify(available=True, live=time.time() - os.path.getmtime(os.path.join(folder, "metrics.jsonl")) < 90,
                       samples=last["samples"], it=last["it"], depth=last["frontier"], max_depth=settings["max_depth"],
                       n_envs=settings["n_envs"],
                       stages=[stage(s) for s in stages],
                       log=[[m["samples"], m["frontier"], m.get("ent"), m.get("ev")] for m in thin],
                       evals=[dict(samples=e["samples"], results=e["results"]) for e in evals],
                       snapshots=[dict(it=it, samples=at[it]) for it in snapshots(folder) if it in at])

    @app.post("/api/replay")
    def replay():
        """One cube, tried by the agent of every snapshot."""
        d = body()
        if d.get("state") is not None:
            st, moves = get_state(d), None
        else:
            moves = scrambled(get_int(d, "depth", 10, 1, 30), get_int(d, "seed", 1, 0, 2 ** 31))
            st = C.apply_moves(C.solved_state(), moves)
        limit = get_int(d, "limit", 100, 1, STEPS_MAX)
        at = moves_at(training_log()[0])
        results = []
        for it in snapshots(folder):
            tr = get_agent(it).trace(st, limit=limit, attention=False, project=False)
            results.append(dict(it=it, samples=at.get(it), solved=tr["solved"], length=len(tr["moves"])))
        return jsonify(state=np.asarray(st).tolist(), scramble=moves, results=results)

    @app.get("/api/benchmark")
    def benchmark():
        try:
            with open(os.path.join(folder, "benchmark.json")) as f:
                return jsonify(available=True, **json.load(f)["page"])
        except (OSError, ValueError, KeyError):
            return jsonify(available=False)

    @app.after_request
    def release_gpu_memory(resp):
        if device.startswith("cuda") and request.path in ("/api/solve", "/api/hint", "/api/thoughtmap", "/api/optimal", "/api/replay"):
            torch.cuda.empty_cache()
        return resp

    @app.errorhandler(ApiError)
    def api_error(e):
        return jsonify(error=str(e)), e.status

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api/"):
            return jsonify(error=e.description), e.code
        return e

    @app.errorhandler(Exception)
    def any_error(e):
        log.exception("unhandled error")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

    return app
