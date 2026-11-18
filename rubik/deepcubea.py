"""The public test set of DeepCubeA (1,000 scrambled cubes with shortest solutions), converted to this project's cubes.

Source: data/cube3/test/data_0.pkl of https://github.com/forestagostinelli/DeepCubeA (Agostinelli et al. 2019).
Their cube uses another sticker numbering, so the relabelling is searched (every one of their moves, relabelled,
must be exactly one of our moves) and every converted cube is checked with its own solution.
"""
from __future__ import annotations

import io
import itertools
import json
import os
import pickle
import urllib.request

import numpy as np

from rubik import cube as C

URL = "https://github.com/forestagostinelli/DeepCubeA/raw/master/data/cube3/test/data_0.pkl"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, ".cache", "deepcubea_test.json")

THEIR_FACES = ["U", "D", "L", "R", "B", "F"]
THEIR_MOVES = [f"{f}{n}" for f in THEIR_FACES for n in (-1, 1)]
_ADJ = {0: [2, 5, 3, 4], 1: [2, 4, 3, 5], 2: [0, 4, 1, 5], 3: [0, 5, 1, 4], 4: [0, 3, 1, 2], 5: [0, 2, 1, 3]}


def _cells(spec) -> list[tuple[int, int]]:
    """their [rows, cols] spec (an int or a range each) -> list of (row, col), rows outer"""
    rows, cols = [[x] if isinstance(x, int) else list(x) for x in spec]
    return [(r, c) for r in rows for c in cols]


def their_move_tables() -> dict[str, np.ndarray]:
    """Their moves as gather indices (``next = state[src]``), rebuilt from their ``_compute_rotation_idxs``."""
    n = 3
    adj_idxs = {
        0: {2: [range(n), n - 1], 3: [range(n), n - 1], 4: [range(n), n - 1], 5: [range(n), n - 1]},
        1: {2: [range(n), 0], 3: [range(n), 0], 4: [range(n), 0], 5: [range(n), 0]},
        2: {0: [0, range(n)], 1: [0, range(n)], 4: [n - 1, range(n - 1, -1, -1)], 5: [0, range(n)]},
        3: {0: [n - 1, range(n)], 1: [n - 1, range(n)], 4: [0, range(n - 1, -1, -1)], 5: [n - 1, range(n)]},
        4: {0: [range(n), n - 1], 1: [range(n - 1, -1, -1), 0], 2: [0, range(n)], 3: [n - 1, range(n - 1, -1, -1)]},
        5: {0: [range(n), 0], 1: [range(n - 1, -1, -1), n - 1], 2: [n - 1, range(n)], 3: [0, range(n - 1, -1, -1)]},
    }
    ring = [[0, range(n)], [range(n), n - 1], [n - 1, range(n - 1, -1, -1)], [range(n - 1, -1, -1), 0]]
    flat = lambda f, rc: f * n * n + rc[0] * n + rc[1]
    out = {}
    for move in THEIR_MOVES:
        face, sign = THEIR_FACES.index(move[0]), int(move[1:])
        src = np.arange(6 * n * n)
        faces_to = _ADJ[face]
        faces_from = [faces_to[(i + 1) % 4] for i in range(4)] if sign == 1 else [faces_to[(i + 3) % 4] for i in range(4)]
        cubes_from = [(i + 3) % 4 for i in range(4)] if sign == 1 else [(i + 1) % 4 for i in range(4)]
        for i in range(4):
            for new, old in zip(_cells(ring[i]), _cells(ring[cubes_from[i]])):
                src[flat(face, new)] = flat(face, old)
        for f_to, f_from in zip(faces_to, faces_from):
            for new, old in zip(_cells(adj_idxs[face][f_to]), _cells(adj_idxs[face][f_from])):
                src[flat(f_to, new)] = flat(f_from, old)
        out[move] = src
    return out


def _extend(theirs: list[np.ndarray], ours: list[np.ndarray], x: int, y: int, P: np.ndarray) -> bool:
    """Grow P from P[x] = y along the moves (P[their[i]] = our[P[i]]); False on a contradiction."""
    P[x] = y
    stack = [x]
    while stack:
        i = stack.pop()
        for t, o in zip(theirs, ours):
            j, pj = int(t[i]), int(o[P[i]])
            if P[j] == -1:
                P[j] = pj
                stack.append(j)
            elif P[j] != pj:
                return False
    return True


def find_mapping(keep_faces: bool = True):
    """(P, mu): P[their sticker position] = our position, mu[their move name] = our action."""
    T = their_move_tables()
    their_axes = [("U", "D"), ("L", "R"), ("B", "F")]
    our_axes = [("U", "D"), ("R", "L"), ("F", "B")]
    solutions = []
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            sigma = {}
            for k, (a, b) in enumerate(their_axes):
                oa, ob = our_axes[perm[k]]
                sigma[a], sigma[b] = (ob, oa) if flips[k] else (oa, ob)
            for direction in (0, 1):
                mu = {m: 2 * C.FACES.index(sigma[m[0]]) + ((0 if int(m[1:]) == 1 else 1) ^ direction) for m in THEIR_MOVES}
                theirs = [T[m] for m in THEIR_MOVES]
                ours = [C.MOVE_SRC[mu[m]] for m in THEIR_MOVES]
                P = np.full(54, -1)
                for f in THEIR_FACES:
                    P[9 * THEIR_FACES.index(f) + 4] = 9 * C.FACES.index(sigma[f]) + 4
                ok = True
                for x in range(54):
                    if P[x] != -1:
                        continue
                    target_face = C.FACES.index(sigma[THEIR_FACES[x // 9]])
                    for y in range(9 * target_face, 9 * target_face + 9):
                        if y in P:
                            continue
                        Q = P.copy()
                        if _extend(theirs, ours, x, y, Q) and all(
                                Q[i] == -1 or Q[i] // 9 == C.FACES.index(sigma[THEIR_FACES[i // 9]]) for i in range(54)):
                            P = Q
                            break
                    else:
                        ok = False
                        break
                if not ok or sorted(P.tolist()) != list(range(54)):
                    continue
                if all(np.array_equal(P[t], o[P]) for t, o in zip(theirs, ours)):
                    solutions.append((P, mu, sigma, direction))
    if not keep_faces:
        return solutions
    for P, mu, sigma, direction in solutions:
        if all(sigma[f] == f for f in THEIR_FACES):
            return P, mu
    raise RuntimeError("no relabelling keeps the face names")


def convert_state(their_state, P: np.ndarray) -> np.ndarray:
    """their sticker array -> ours: the sticker at their position i sits at our position P[i]"""
    s = np.asarray(their_state)
    out = np.empty(54, dtype=np.uint8)
    out[P] = P[s]
    return out


class _State:
    __slots__ = ["colors", "hash"]


class _SafeUnpickler(pickle.Unpickler):
    """Builds numpy arrays and the state stub only: a pickle from the internet must not run anything else."""

    def find_class(self, module, name):
        try:
            import numpy._core.multiarray as ncm
        except ImportError:
            import numpy.core.multiarray as ncm
        allowed = {("environments.cube3", "Cube3State"): _State,
                   ("numpy.core.multiarray", "_reconstruct"): ncm._reconstruct,
                   ("numpy.core.multiarray", "scalar"): ncm.scalar,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"blocked global in the downloaded file: {module}.{name}")
        return allowed[(module, name)]


def fetch(path: str = DEFAULT_PATH, url: str = URL) -> str:
    """Download their pickle, read it with the restricted unpickler, and keep plain JSON at `path`."""
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    d = _SafeUnpickler(io.BytesIO(raw)).load()
    states = np.stack([np.asarray(s.colors) for s in d["states"]]).astype(int)
    sols = [[f"{m[0]}{int(m[1])}" for m in sol] for sol in d["solutions"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(source=url, note="DeepCubeA test set (Agostinelli et al. 2019): their sticker arrays and "
                                        "optimal solutions in their move names; converted on load by rubik.deepcubea",
                       states=states.tolist(), solutions=sols), f)
    return path


def load_test_set(path: str = DEFAULT_PATH, download: bool = True) -> dict:
    """states [N, 54] uint8 (our cubes), solutions (our actions, optimal), optimal_len [N]. Every cube is verified."""
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        fetch(path)
    with open(path) as f:
        d = json.load(f)
    T = their_move_tables()
    P, mu = find_mapping()
    states, solutions = [], []
    for k, (s, sol) in enumerate(zip(d["states"], d["solutions"])):
        s = np.asarray(s)
        t = s.copy()
        for m in sol:
            t = t[T[m]]
        if not np.array_equal(t, np.arange(54)):
            raise ValueError(f"test cube {k}: its own solution does not solve it (move tables differ from theirs)")
        ours = convert_state(s, P)
        moves = [mu[m] for m in sol]
        if C.check_solvable(ours) is not None or not C.is_solved(C.apply_moves(ours, moves)):
            raise ValueError(f"test cube {k}: the converted solution does not solve the converted cube")
        states.append(ours)
        solutions.append(moves)
    return dict(states=np.stack(states), solutions=solutions, optimal_len=np.array([len(m) for m in solutions]),
                source=d.get("source", ""))
