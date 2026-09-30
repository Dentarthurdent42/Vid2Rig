"""Skeleton topology, forward kinematics and skinning weights."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

from .joints import Joint, estimate_joint
from .parts import PartFit
from .tracks import Tracks


@dataclass
class Rig:
    """A skeleton fitted to tracks.

    Bones are the rigid parts of `fit`, ordered so that parents precede children.
    Bone k's pivot is the joint to its parent (the part centroid for the root).
    The rest pose is the pose at the reference frame, so rest rotations are identity.
    """

    fit: PartFit
    order: list[int]  # part indices, parents first; bone i = part order[i]
    parent: np.ndarray  # (B,) parent bone index, -1 for the root
    joints: list[Joint | None]  # (B,) joint to the parent, None for the root
    pivots: np.ndarray  # (B, D) rest pivot positions
    tails: np.ndarray  # (B, D) rest end points (for display / BVH end sites)
    names: list[str]
    weights: np.ndarray  # (N, B) skinning weights
    fps: float = 30.0
    edge_cost: np.ndarray = field(default=None, repr=False)

    @property
    def B(self) -> int:
        return len(self.order)

    @property
    def D(self) -> int:
        return self.pivots.shape[1]

    @property
    def T(self) -> int:
        return self.fit.R.shape[1]

    def world_rotations(self) -> np.ndarray:
        """(B, T, D, D) rotation of each bone relative to the rest pose."""
        return self.fit.R[self.order]

    def world_translations(self) -> np.ndarray:
        return self.fit.t[self.order]

    def local_rotations(self) -> np.ndarray:
        """(B, T, D, D) rotation of each bone relative to its parent."""
        Rw = self.world_rotations()
        L = Rw.copy()
        for i, p in enumerate(self.parent):
            if p >= 0:
                L[i] = np.swapaxes(Rw[p], 1, 2) @ Rw[i]
        return L

    def joint_positions(self) -> np.ndarray:
        """(B, T, D) pivot positions over time, by forward kinematics from the root."""
        Rw = self.world_rotations()
        tw = self.world_translations()
        P = np.zeros((self.B, self.T, self.D))
        P[0] = np.einsum("tde,e->td", Rw[0], self.pivots[0]) + tw[0]
        for i in range(1, self.B):
            p = self.parent[i]
            P[i] = P[p] + np.einsum("tde,e->td", Rw[p], self.pivots[i] - self.pivots[p])
        return P

    def tail_positions(self) -> np.ndarray:
        Rw = self.world_rotations()
        P = self.joint_positions()
        return P + np.einsum("btde,be->btd", Rw, self.tails - self.pivots)

    def skin(self, rest_points: np.ndarray | None = None) -> np.ndarray:
        """Linear blend skinning of the rest points -> (T, N, D)."""
        X0 = self.fit.canonical if rest_points is None else rest_points
        Rw, tw = self.world_rotations(), self.world_translations()
        Y = np.einsum("btde,ne->btnd", Rw, X0) + tw[:, :, None]
        return np.einsum("nb,btnd->tnd", self.weights, Y)

    def summary(self) -> dict:
        bones = []
        for i in range(self.B):
            j = self.joints[i]
            bones.append({
                "name": self.names[i],
                "parent": self.names[self.parent[i]] if self.parent[i] >= 0 else None,
                "part": int(self.order[i]),
                "points": int((self.fit.labels == self.order[i]).sum()),
                "pivot": np.round(self.pivots[i], 5).tolist(),
                "joint": None if j is None else {
                    "type": j.kind,
                    "axis": None if j.axis is None else np.round(j.axis, 4).tolist(),
                    "rms": round(j.rms, 6),
                    "range_deg": round(j.range_deg, 2),
                    "hinge_score": round(j.hinge_score, 4),
                },
            })
        return {"frames": self.T, "fps": self.fps, "dims": self.D, "reference_frame": self.fit.ref,
                "bones": bones, "outlier_points": int((self.fit.labels < 0).sum())}


def _closest_pair(Pa, Pb, ta, tb):
    if len(Pa) > len(Pb):
        d, j = ta.query(Pb)
        i = int(np.argmin(d))
        return float(d[i]), 0.5 * (Pb[i] + Pa[j[i]])
    d, j = tb.query(Pa)
    i = int(np.argmin(d))
    return float(d[i]), 0.5 * (Pa[i] + Pb[j[i]])


def build_rig(tracks: Tracks, fit: PartFit, root: int | None = None,
              gap_weight: float = 1.0, max_influences: int = 4) -> Rig:
    """Connect parts into a tree (minimum spanning tree over joint-fit costs)."""
    K, D = fit.K, fit.canonical.shape[1]
    labels = fit.labels
    pts = [fit.canonical[labels == k] for k in range(K)]
    trees = [cKDTree(p) for p in pts]
    ext = tracks.extent()

    cost = np.zeros((K, K))
    p0s = {}
    for a in range(K):
        for b in range(a + 1, K):
            gap, p0 = _closest_pair(pts[a], pts[b], trees[a], trees[b])
            valid = fit.valid[a] & fit.valid[b]
            j = estimate_joint(fit.R[a], fit.t[a], fit.R[b], fit.t[b], valid, p0)
            prox = trees[a].query(j.position)[0] + trees[b].query(j.position)[0]
            cost[a, b] = cost[b, a] = j.rms + 0.5 * prox + gap_weight * gap + 1e-9 * ext
            p0s[a, b] = p0s[b, a] = p0

    if root is None:
        root = int(np.argmax([len(p) for p in pts]))
    mst = minimum_spanning_tree(cost).toarray() if K > 1 else np.zeros((1, 1))
    adj = (mst + mst.T) > 0

    order, parent_part = [root], {root: -1}
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b in np.flatnonzero(adj[a]):
            if b not in parent_part:
                parent_part[b] = a
                order.append(int(b))
                queue.append(int(b))
    bone_of = {p: i for i, p in enumerate(order)}
    parent = np.array([bone_of[parent_part[p]] if parent_part[p] >= 0 else -1 for p in order])

    joints: list[Joint | None] = []
    pivots = np.zeros((len(order), D))
    for i, p in enumerate(order):
        if parent[i] < 0:
            joints.append(None)
            pivots[i] = pts[p].mean(0)
            continue
        q = order[parent[i]]
        valid = fit.valid[q] & fit.valid[p]
        j = estimate_joint(fit.R[q], fit.t[q], fit.R[p], fit.t[p], valid, p0s[q, p], parent=q, child=p)
        joints.append(j)
        pivots[i] = j.pos_parent

    tails = np.zeros_like(pivots)
    for i, p in enumerate(order):
        children = np.flatnonzero(parent == i)
        if len(children):
            tails[i] = pivots[children].mean(0)
            if np.linalg.norm(tails[i] - pivots[i]) < 1e-6 * ext:
                children = []
        if not len(children):
            d = np.linalg.norm(pts[p] - pivots[i], axis=1)
            far = d >= np.percentile(d, 90)
            tails[i] = pts[p][far].mean(0)

    weights = skinning_weights(fit, order, max_influences)
    names = ["root"] + [f"joint_{i:02d}" for i in range(1, len(order))]
    return Rig(fit, order, parent, joints, pivots, tails, names, weights, tracks.fps, cost)


def skinning_weights(fit: PartFit, order: list[int], max_influences: int = 4) -> np.ndarray:
    """Soft weights from how well each part's motion explains each point."""
    r = fit.residual[order].T  # (N, B)
    ok = fit.labels >= 0
    own = r[ok, [order.index(l) for l in fit.labels[ok]]] if ok.any() else np.array([fit.noise_var])
    s2 = 2.0 * max(np.median(own[np.isfinite(own)]), fit.noise_var)
    # Points that no motion explains (never visible enough) follow the root.
    none = ~np.isfinite(r).any(1)
    r = np.where(np.isfinite(r), r, np.inf)
    r[none, 0] = 0.0
    logw = -(r - r.min(1, keepdims=True)) / s2
    w = np.exp(logw)
    if max_influences and w.shape[1] > max_influences:
        cut = -np.sort(-w, axis=1)[:, max_influences - 1 : max_influences]
        w[w < cut] = 0.0
    w[w < 1e-3] = 0.0
    return w / w.sum(1, keepdims=True)
