"""Rigid-transform utilities that work in 2D and 3D."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rot2d(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def kabsch_batch(q: np.ndarray, X: np.ndarray, W: np.ndarray, min_points: int = 3):
    """Per-frame weighted rigid fit  R_t q_i + t_t ~= X[t, i].

    q: (N, D) canonical positions, X: (T, N, D) observations, W: (T, N) weights
    (zero = unobserved). Returns R (T, D, D), t (T, D) and valid (T,), where
    valid marks frames with enough support; invalid frames get the identity.
    """
    T, N, D = X.shape
    W = np.asarray(W, dtype=np.float64)
    ws = W.sum(1)
    valid = ((W > 0).sum(1) >= max(min_points, D)) & (ws > 0)
    wn = W / np.where(ws > 0, ws, 1.0)[:, None]
    mp = wn @ q
    mq = np.einsum("tn,tnd->td", wn, X)
    H = np.einsum("tn,nd,tne->tde", wn, q, X, optimize=True) - mp[:, :, None] * mq[:, None, :]
    U, _, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, 1, 2)
    Ut = np.swapaxes(U, 1, 2)
    d = np.sign(np.linalg.det(V @ Ut))
    d[d == 0] = 1.0
    Dm = np.tile(np.eye(D), (T, 1, 1))
    Dm[:, -1, -1] = d
    R = V @ Dm @ Ut
    t = mq - np.einsum("tde,te->td", R, mp)
    R[~valid] = np.eye(D)
    t[~valid] = 0.0
    return R, t, valid


def fill_invalid(R: np.ndarray, t: np.ndarray, valid: np.ndarray):
    """Interpolate transforms on frames without enough support (slerp in 3D)."""
    if valid.all() or not valid.any():
        return R, t
    T, D = t.shape
    idx = np.flatnonzero(valid)
    frames = np.arange(T)
    t = np.stack([np.interp(frames, idx, t[idx, d]) for d in range(D)], -1)
    if D == 3:
        if len(idx) == 1:
            R = np.repeat(R[idx[:1]], T, 0)
        else:
            R = Slerp(idx, Rotation.from_matrix(R[idx]))(np.clip(frames, idx[0], idx[-1])).as_matrix()
    else:
        ang = np.unwrap(np.arctan2(R[idx, 1, 0], R[idx, 0, 0]))
        R = rot2d(np.interp(frames, idx, ang))
    return R, t


def apply(R: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply per-frame transforms (T,D,D),(T,D) to points p (N,D) or (D,) -> (T,N,D) / (T,D)."""
    if p.ndim == 1:
        return np.einsum("tde,e->td", R, p) + t
    return np.matmul(p[None], np.swapaxes(R, 1, 2)) + t[:, None]


def apply_inverse(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """R_t^T (X[t] - t_t) for X (T, N, D)."""
    return np.matmul(X - t[:, None], R)


def regauge(R: np.ndarray, t: np.ndarray, ref: int):
    """Re-express a motion so that the transform at frame `ref` is the identity."""
    Rr, tr = R[ref], t[ref]
    R2 = R @ Rr.T
    t2 = t - np.einsum("tde,e->td", R2, tr)
    return R2, t2


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle of rotations. 3D -> (T, 3) rotation vectors; 2D -> (T, 1) angles."""
    if R.shape[-1] == 3:
        return Rotation.from_matrix(R).as_rotvec()
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])[..., None]


def to3d(R: np.ndarray, t: np.ndarray):
    """Embed planar transforms in 3D (rotation about +Z)."""
    if R.shape[-1] == 3:
        return R, t
    R3 = np.zeros(R.shape[:-2] + (3, 3))
    R3[..., :2, :2] = R
    R3[..., 2, 2] = 1.0
    t3 = np.concatenate([t, np.zeros(t.shape[:-1] + (1,))], -1)
    return R3, t3


def pad3(p: np.ndarray) -> np.ndarray:
    if p.shape[-1] == 3:
        return p
    return np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], -1)
