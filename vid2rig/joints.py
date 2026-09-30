"""Joint estimation between two rigid parts.

For parts A and B with motions T_A(t), T_B(t), a joint is a point that is
fixed in both local frames:  R_A p_A + t_A = R_B p_B + t_B  for all t.
Stacking every frame gives a linear least-squares problem in (p_A, p_B).
A hinge leaves the solution free along its axis; that unobservable direction
is taken from a prior (where the two parts touch).

The relative rotation R_A^T R_B then tells the joint type: a hinge rotates
about one fixed axis, a ball joint about varying axes, a "fixed" joint
barely moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import rotation_log


@dataclass
class Joint:
    parent: int
    child: int
    pos_parent: np.ndarray  # joint centre in the parent's local frame (= world at ref frame)
    pos_child: np.ndarray  # joint centre in the child's local frame
    kind: str  # "ball" | "hinge" | "fixed"
    axis: Optional[np.ndarray]  # hinge axis in the parent's local frame (3D only)
    rms: float  # RMS distance between the two estimates of the joint centre over time
    range_deg: float  # range of motion of the relative rotation
    hinge_score: float  # fraction of relative rotation about the dominant axis

    @property
    def position(self) -> np.ndarray:
        return 0.5 * (self.pos_parent + self.pos_child)


def estimate_joint(
    RA: np.ndarray, tA: np.ndarray, RB: np.ndarray, tB: np.ndarray,
    valid: np.ndarray, p0: np.ndarray, parent: int = 0, child: int = 1,
    weak: float = 0.05, hinge_threshold: float = 0.95, min_range_deg: float = 4.0,
) -> Joint:
    """Fit a joint between two motions, both in the ref-frame gauge.

    valid: frames where both motions are observed. p0: prior joint location
    (world at the reference frame), e.g. the midpoint of the closest pair of
    points between the parts. weak: directions whose singular value is below
    weak x the largest are treated as unobservable and taken from p0.
    """
    D = tA.shape[-1]
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        idx = np.arange(len(tA))
    M = np.concatenate([RA[idx], -RB[idx]], axis=2).reshape(-1, 2 * D)
    b = (tB[idx] - tA[idx]).reshape(-1)
    # Truncated SVD around the prior: well-observed directions come from the data,
    # near-singular ones (e.g. along a hinge axis) stay at the prior.
    x0 = np.concatenate([p0, p0])
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > weak * s[0]
    sol = x0 + Vt[keep].T @ ((U[:, keep].T @ (b - M @ x0)) / s[keep])
    pA, pB = sol[:D], sol[D:]

    wa = np.einsum("tde,e->td", RA, pA) + tA
    wb = np.einsum("tde,e->td", RB, pB) + tB
    rms = float(np.sqrt(np.mean(np.sum((wa - wb)[idx] ** 2, -1))))

    rel = np.einsum("ted,tef->tdf", RA[idx], RB[idx])  # R_A^T R_B
    rv = rotation_log(rel)
    if D == 2:
        ang = rv[:, 0]
        rng_deg = float(np.rad2deg(ang.max() - ang.min()))
        kind = "hinge" if rng_deg >= min_range_deg else "fixed"
        return Joint(parent, child, pA, pB, kind, None, rms, rng_deg, 1.0)

    C = rv.T @ rv
    w, U = np.linalg.eigh(C)
    axis = U[:, -1]
    score = float(w[-1] / max(w.sum(), 1e-12))
    ang = rv @ axis
    rng_deg = float(np.rad2deg(max(ang.max() - ang.min(), np.linalg.norm(rv, axis=1).max())))
    if rng_deg < min_range_deg:
        kind = "fixed"
    elif score >= hinge_threshold:
        kind = "hinge"
    else:
        kind = "ball"
    return Joint(parent, child, pA, pB, kind, axis if kind == "hinge" else None, rms, rng_deg, score)
