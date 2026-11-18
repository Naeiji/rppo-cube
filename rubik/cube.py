"""3x3x3 Rubik's cube in the quarter-turn metric, batched on the GPU.

A cube is a uint8 vector of 54 sticker ids: state[pos] is the sticker now at position pos, in the facelet order
U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9. A sticker's colour is id // 9 and the solved cube is arange(54).
The 12 actions are U U' R R' F F' D D' L L' B B'; action a ^ 1 undoes action a.
"""
from __future__ import annotations

import numpy as np
import torch

FACES = "URFDLB"
COLOR_NAMES = ["white", "red", "green", "yellow", "orange", "blue"]
MOVE_NAMES = ["U", "U'", "R", "R'", "F", "F'", "D", "D'", "L", "L'", "B", "B'"]
N_MOVES = 12
NO_MOVE = 12
GODS_NUMBER_QTM = 26
CENTERS = [9 * f + 4 for f in range(6)]
NON_CENTERS = [i for i in range(54) if i % 9 != 4]

_NORMAL = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}
_RIGHT = {"U": (1, 0, 0), "R": (0, 0, -1), "F": (1, 0, 0), "D": (1, 0, 0), "L": (0, 0, 1), "B": (-1, 0, 0)}
_DOWN = {"U": (0, 0, 1), "R": (0, -1, 0), "F": (0, -1, 0), "D": (0, 0, -1), "L": (0, -1, 0), "B": (0, -1, 0)}


def _geometry():
    pos = np.zeros((54, 3), np.int64)
    nrm = np.zeros((54, 3), np.int64)
    for f, face in enumerate(FACES):
        n, right, down = (np.array(v[face]) for v in (_NORMAL, _RIGHT, _DOWN))
        for r in range(3):
            for c in range(3):
                pos[9 * f + 3 * r + c] = n + (c - 1) * right + (r - 1) * down
                nrm[9 * f + 3 * r + c] = n
    return pos, nrm


STICKER_POS, STICKER_NORMAL = _geometry()
_INDEX = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(STICKER_POS, STICKER_NORMAL))}


def _rotate(v, n, clockwise):
    """Rotate integer vector v by 90 degrees about axis n: clockwise as seen from the tip of n."""
    cross = np.cross(n, v)
    return n * int(np.dot(n, v)) + (-cross if clockwise else cross)


def _move_src(face, clockwise):
    n = np.array(_NORMAL[face])
    src = np.arange(54)
    for i in range(54):
        if np.dot(STICKER_POS[i], n) == 1:
            j = _INDEX[(tuple(_rotate(STICKER_POS[i], n, clockwise)), tuple(_rotate(STICKER_NORMAL[i], n, clockwise)))]
            src[j] = i
    return src


MOVE_SRC = np.stack([_move_src(face, cw) for face in FACES for cw in (True, False)])
MOVE_FACE = np.arange(N_MOVES) // 2
MOVE_AXIS = MOVE_FACE % 3
MOVE_SLOT = MOVE_FACE // 3
MOVE_QUARTERS = np.where(np.arange(N_MOVES) % 2 == 0, 1, 3)
INVERSE = np.arange(N_MOVES) ^ 1


def _cubies():
    corners, edges = [], []
    for p in sorted({tuple(p) for p in STICKER_POS}):
        stickers = [i for i in range(54) if tuple(STICKER_POS[i]) == p and i % 9 != 4]
        if len(stickers) == 3:
            ud = [i for i in stickers if STICKER_NORMAL[i][1] != 0][0]
            a, b = [i for i in stickers if i != ud]
            n0, na, nb = STICKER_NORMAL[ud], STICKER_NORMAL[a], STICKER_NORMAL[b]
            corners.append([ud, a, b] if np.dot(np.cross(n0, na), nb) > 0 else [ud, b, a])
        elif len(stickers) == 2:
            a, b = stickers
            if STICKER_NORMAL[b][1] != 0 or (STICKER_NORMAL[a][1] == 0 and STICKER_NORMAL[b][2] != 0):
                a, b = b, a
            edges.append([a, b])
    return np.array(corners), np.array(edges)


CORNERS, EDGES = _cubies()
STICKER_CLASS_INDEX = np.zeros(54, np.int64)
STICKER_CLASS_INDEX[CORNERS.reshape(-1)] = np.arange(24)
STICKER_CLASS_INDEX[EDGES.reshape(-1)] = np.arange(24)
REF_STICKERS = np.concatenate([CORNERS[:, 0], EDGES[:, 0]])
N_FEATURES = 40 * 24 + 13


def solved_state() -> np.ndarray:
    return np.arange(54, dtype=np.uint8)


def apply_moves(state: np.ndarray, moves) -> np.ndarray:
    s = np.asarray(state, dtype=np.uint8).copy()
    for a in moves:
        s = s[MOVE_SRC[int(a)]]
    return s


def is_solved(state) -> bool:
    return bool(np.array_equal(np.asarray(state), np.arange(54)))


_TOKEN_SUFFIX = {"": [0], "'": [1], "2": [0, 0], "2'": [1, 1], "'2": [1, 1], "3": [1], "3'": [0]}


def parse_moves(text: str) -> list[int]:
    """'R U R\\' U2' -> action list (quarter turns; X2 becomes two X)."""
    out = []
    for tok in text.replace(",", " ").replace("’", "'").split():
        face, suffix = tok[0].upper(), tok[1:]
        if face not in FACES or suffix not in _TOKEN_SUFFIX:
            raise ValueError(f"unknown move {tok!r} (use U R F D L B with optional ' or 2)")
        out += [2 * FACES.index(face) + s for s in _TOKEN_SUFFIX[suffix]]
    return out


def moves_to_str(moves, merge_halves: bool = False) -> str:
    names, i, moves = [], 0, [int(a) for a in moves]
    while i < len(moves):
        if merge_halves and i + 1 < len(moves) and moves[i] == moves[i + 1]:
            names.append(FACES[moves[i] // 2] + "2")
            i += 2
        else:
            names.append(MOVE_NAMES[moves[i]])
            i += 1
    return " ".join(names)


def to_facelets(state) -> str:
    return "".join(FACES[int(i) // 9] for i in np.asarray(state))


def _parity(perm) -> int:
    perm, seen, parity = list(perm), [False] * len(perm), 0
    for i in range(len(perm)):
        if not seen[i]:
            j, n = i, 0
            while not seen[j]:
                seen[j], j, n = True, perm[j], n + 1
            parity ^= (n - 1) & 1
    return parity


def cubie_state(state):
    """sticker state -> (corner perm, corner twist, edge perm, edge flip); raises ValueError on a broken cube."""
    s = np.asarray(state)
    cp, co, ep, eo = np.zeros(8, int), np.zeros(8, int), np.zeros(12, int), np.zeros(12, int)
    corner_of = {frozenset(int(x) // 9 for x in c): k for k, c in enumerate(CORNERS)}
    edge_of = {frozenset(int(x) // 9 for x in e): k for k, e in enumerate(EDGES)}
    for i, pos in enumerate(CORNERS):
        k = corner_of.get(frozenset(int(s[p]) // 9 for p in pos))
        if k is None:
            raise ValueError("a corner has an impossible colour combination")
        cp[i] = k
        twist = [t for t in range(3) if int(s[pos[t]]) == CORNERS[k][0]]
        if not twist:
            raise ValueError("inconsistent corner stickers")
        co[i] = twist[0]
    for i, pos in enumerate(EDGES):
        k = edge_of.get(frozenset(int(s[p]) // 9 for p in pos))
        if k is None:
            raise ValueError("an edge has an impossible colour combination")
        ep[i] = k
        eo[i] = 0 if int(s[pos[0]]) == EDGES[k][0] else 1
    return cp, co, ep, eo


def from_cubies(cp, co, ep, eo) -> np.ndarray:
    s = solved_state()
    for i in range(8):
        for k in range(3):
            s[CORNERS[i][(co[i] + k) % 3]] = CORNERS[cp[i]][k]
    for i in range(12):
        for k in range(2):
            s[EDGES[i][(eo[i] + k) % 2]] = EDGES[ep[i]][k]
    return s


def check_solvable(state) -> str | None:
    """None if the state is a reachable cube, else a human-readable reason."""
    s = np.asarray(state)
    if s.shape != (54,) or sorted(s.tolist()) != list(range(54)):
        return "not a permutation of the 54 stickers"
    try:
        cp, co, ep, eo = cubie_state(s)
    except ValueError as e:
        return str(e)
    if len(set(cp)) != 8 or len(set(ep)) != 12:
        return "a piece appears twice"
    if co.sum() % 3:
        return "a corner is twisted (corner twist sum is not 0 mod 3)"
    if eo.sum() % 2:
        return "an edge is flipped (edge flip sum is odd)"
    if _parity(cp) != _parity(ep):
        return "two pieces are swapped (corner and edge permutation parities differ)"
    if not np.array_equal(from_cubies(cp, co, ep, eo), s):
        return "a corner's stickers are in mirror order (or a centre is out of place): no real cube looks like this"
    return None


def from_facelets(colors: str) -> np.ndarray:
    """54 face letters (URFDLB, Kociemba order) -> sticker state. Raises ValueError if unsolvable."""
    colors = colors.strip().upper()
    if len(colors) != 54 or any(c not in FACES for c in colors):
        raise ValueError("need 54 letters from URFDLB")
    for f, face in enumerate(FACES):
        if colors[9 * f + 4] != face:
            raise ValueError("centres must be U R F D L B in Kociemba order")
        if colors.count(face) != 9:
            raise ValueError(f"colour {face} appears {colors.count(face)} times, not 9")
    col = np.array([FACES.index(c) for c in colors])
    s = solved_state()
    for group in (CORNERS, EDGES):
        for pos in group:
            here = [int(col[p]) for p in pos]
            match = [home for home in group if sorted(int(h) // 9 for h in home) == sorted(here)]
            if not match:
                raise ValueError("a piece has an impossible colour combination")
            home = match[0]
            for p in pos:
                s[p] = [h for h in home if int(h) // 9 == int(col[p])][0]
    reason = check_solvable(s)
    if reason:
        raise ValueError(reason)
    return s


def random_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniformly random reachable cubes (exact: random pieces + the three invariants)."""
    out = np.zeros((n, 54), np.uint8)
    for k in range(n):
        cp, ep = rng.permutation(8), rng.permutation(12)
        if _parity(cp) != _parity(ep):
            ep[[0, 1]] = ep[[1, 0]]
        co = rng.integers(0, 3, 8)
        co[7] = (-co[:7].sum()) % 3
        eo = rng.integers(0, 2, 12)
        eo[11] = eo[:11].sum() % 2
        out[k] = from_cubies(cp, co, ep, eo)
    return out


class CubeOps:
    """Device-resident tables and batched operations on [N, 54] uint8 states."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        d = self.device
        self.src = torch.as_tensor(MOVE_SRC, device=d)
        self.solved = torch.arange(54, device=d, dtype=torch.uint8)
        self.arange54 = torch.arange(54, device=d, dtype=torch.uint8)
        self.move_axis = torch.as_tensor(MOVE_AXIS, device=d)
        self.move_slot = torch.as_tensor(MOVE_SLOT, device=d)
        self.move_quarters = torch.as_tensor(MOVE_QUARTERS, device=d)
        self.move_cw = torch.as_tensor(np.arange(N_MOVES) % 2 == 0, device=d)
        self.class_index = torch.as_tensor(STICKER_CLASS_INDEX, device=d)
        self.ref = torch.as_tensor(REF_STICKERS, device=d)
        self.feat_offset = torch.arange(40, device=d) * 24
        undo = torch.zeros(N_MOVES + 1, N_MOVES, dtype=torch.bool, device=d)
        undo[torch.arange(N_MOVES), torch.as_tensor(INVERSE)] = True
        self.undo_mask = undo
        g = torch.Generator().manual_seed(12345)
        self.hash_w = torch.randint(-(2 ** 62), 2 ** 62, (54,), generator=g, dtype=torch.int64).to(d)

    def apply(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return state.gather(1, self.src[actions])

    def is_solved(self, state: torch.Tensor) -> torch.Tensor:
        return (state == self.solved).all(1)

    def inverse_perm(self, state: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(state)
        return inv.scatter_(1, state.long(), self.arange54.expand_as(state).contiguous())

    def features(self, state: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """Feature indices [N, 41]: the piece in each of the 20 slots, the slot of each of the 20 pieces, the previous move."""
        here = self.class_index[state[:, self.ref].long()]
        where = self.class_index[self.inverse_perm(state)[:, self.ref].long()]
        idx = torch.cat([here, where], 1) + self.feat_offset
        return torch.cat([idx, 960 + prev_action.long().unsqueeze(1)], 1)

    def expand_all(self, state: torch.Tensor) -> torch.Tensor:
        """All 12 children of every cube: [M, 54] -> [M*12, 54]; child m*12 + a is move a applied to cube m."""
        return torch.stack([state[:, self.src[a]] for a in range(N_MOVES)], 1).reshape(-1, 54)

    def hash(self, state: torch.Tensor, chunk: int = 131072) -> torch.Tensor:
        if state.shape[0] <= chunk:
            return (state.long() * self.hash_w).sum(1)
        return torch.cat([(s.long() * self.hash_w).sum(1) for s in state.split(chunk)])

    def scramble(self, depths: torch.Tensor, generator: torch.Generator | None = None,
                 start: torch.Tensor | None = None, return_moves: bool = False):
        """Random canonical scrambles of the given lengths: no move is cancelled or merged by a later one."""
        n = depths.shape[0]
        d = self.device
        state = self.solved.expand(n, 54).clone() if start is None else start.clone()
        k_max = int(depths.max()) if n else 0
        axis = torch.full((n,), -1, dtype=torch.long, device=d)
        cnt = torch.zeros(n, 2, dtype=torch.long, device=d)
        rows = torch.arange(n, device=d)
        moves = torch.full((n, max(k_max, 1)), -1, dtype=torch.long, device=d)
        for t in range(k_max):
            same = self.move_axis.unsqueeze(0) == axis.unsqueeze(1)
            c = cnt.gather(1, self.move_slot.unsqueeze(0).expand(n, N_MOVES))
            ok_same = (c == 0) | ((c == 1) & self.move_cw) | ((c == 3) & ~self.move_cw)
            allowed = ~same | ok_same
            score = torch.rand(n, N_MOVES, device=d, generator=generator).masked_fill(~allowed, -1.0)
            a = score.argmax(1)
            active = t < depths
            state = torch.where(active.unsqueeze(1), self.apply(state, a), state)
            ax, sl = self.move_axis[a], self.move_slot[a]
            new_run = (ax != axis) & active
            cnt = torch.where(new_run.unsqueeze(1), torch.zeros_like(cnt), cnt)
            upd = (cnt[rows, sl] + self.move_quarters[a]) % 4
            cnt[rows, sl] = torch.where(active, upd, cnt[rows, sl])
            axis = torch.where(active, ax, axis)
            moves[:, t] = torch.where(active, a, moves[:, t])
        return (state, moves) if return_moves else state
