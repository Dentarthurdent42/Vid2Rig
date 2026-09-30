"""Synthetic articulated objects with ground truth, for testing and demos.

A small "creature" (torso, head, tail, jointed legs) is animated with smooth
random joint motion, and surface points are tracked through time with
optional noise and occlusion, mimicking the output of a point tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .tracks import Tracks


@dataclass
class BoneSpec:
    name: str
    parent: int
    offset: tuple  # pivot, in the parent's local frame
    tip: tuple  # segment end, in the bone's local frame
    radius: float
    kind: str = "ball"  # "ball" | "hinge"
    axis: tuple = (0.0, 0.0, 1.0)
    amp_deg: float = 35.0
    base: tuple = (0.0, 0.0, 0.0)  # segment start, in the bone's local frame


def creature_spec(n_legs: int = 4) -> list[BoneSpec]:
    bones = [
        BoneSpec("torso", -1, (0, 0, 0), (0.5, 0, 0), 0.16, base=(-0.5, 0, 0), amp_deg=20),
        BoneSpec("head", 0, (0.55, 0.05, 0), (0.3, 0.18, 0), 0.1, amp_deg=35),
        BoneSpec("tail", 0, (-0.55, 0.02, 0), (-0.45, 0.12, 0), 0.05, amp_deg=40),
    ]
    sides = [0.12, -0.12] if n_legs == 4 else [0.12]
    for x, tag in ((0.35, "front"), (-0.35, "hind")):
        for z in sides:
            side = "L" if z > 0 else "R"
            up = len(bones)
            bones.append(BoneSpec(f"{tag}_upper_{side}", 0, (x, -0.1, z), (0, -0.35, 0), 0.06, amp_deg=35))
            bones.append(
                BoneSpec(f"{tag}_lower_{side}", up, (0, -0.35, 0), (0, -0.3, 0), 0.05, kind="hinge", amp_deg=45)
            )
    return bones


@dataclass
class SyntheticScene:
    tracks: Tracks
    labels: np.ndarray  # (N,) ground-truth bone per point
    parent: np.ndarray  # (K,) ground-truth parent, -1 for root
    pivots: np.ndarray  # (K, T, D) ground-truth joint centres over time
    kinds: list[str]
    names: list[str]
    specs: list[BoneSpec] = field(repr=False, default_factory=list)


def _smooth_signal(rng, T, fps, amp, n=3):
    t = np.arange(T) / fps
    s = np.zeros(T)
    for _ in range(n):
        f = rng.uniform(0.3, 1.2)
        s += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return amp * s / n * 1.6


def make_scene(
    T: int = 120,
    n_points: int = 1500,
    noise: float = 0.0,
    dropout: float = 0.0,
    planar: bool = False,
    n_legs: int = 4,
    fps: float = 30.0,
    seed: int = 0,
) -> SyntheticScene:
    """Animate the creature and return tracks plus ground truth.

    noise: std of Gaussian noise added to observations (scene units, the creature is ~1.5 long).
    dropout: fraction of observations hidden, as random gaps in each track.
    planar: 2D scene (all joints rotate about Z), like tracks taken straight from an image.
    """
    rng = np.random.default_rng(seed)
    specs = creature_spec(2 if planar else n_legs)
    K = len(specs)

    # Local joint rotations over time.
    local = np.zeros((K, T, 3, 3))
    for k, b in enumerate(specs):
        amp = np.deg2rad(b.amp_deg)
        if planar or b.kind == "hinge":
            axis = np.array([0, 0, 1.0]) if planar else np.asarray(b.axis, float)
            ang = _smooth_signal(rng, T, fps, amp)
            rv = ang[:, None] * axis
        else:
            rv = np.stack([_smooth_signal(rng, T, fps, amp) for _ in range(3)], -1)
        local[k] = Rotation.from_rotvec(rv).as_matrix()

    # Root motion: drift and turn.
    tt = np.arange(T) / fps
    root_pos = np.stack([0.3 * tt, 0.05 * np.sin(2 * np.pi * 0.7 * tt), 0.1 * np.sin(0.5 * tt)], -1)

    # Forward kinematics.
    Rw = np.zeros((K, T, 3, 3))
    pw = np.zeros((K, T, 3))
    for k, b in enumerate(specs):
        if b.parent < 0:
            Rw[k] = local[k]
            pw[k] = root_pos
        else:
            Rw[k] = Rw[b.parent] @ local[k]
            pw[k] = pw[b.parent] + np.einsum("tde,e->td", Rw[b.parent], np.asarray(b.offset, float))

    # Sample points on each bone, proportional to its size.
    sizes = np.array([np.linalg.norm(np.subtract(b.tip, b.base)) * b.radius + b.radius**2 for b in specs])
    counts = np.maximum(12, np.round(n_points * sizes / sizes.sum()).astype(int))
    local_pts, labels = [], []
    for k, (b, c) in enumerate(zip(specs, counts)):
        s = rng.uniform(0, 1, c)[:, None]
        base, tip = np.asarray(b.base, float), np.asarray(b.tip, float)
        if planar:
            ang = rng.uniform(0, 2 * np.pi, c)
            r = b.radius * np.sqrt(rng.uniform(0, 1, c))
            n = np.stack([np.cos(ang), np.sin(ang), np.zeros(c)], -1) * r[:, None]
        else:
            n = rng.normal(size=(c, 3))
            n *= b.radius / np.linalg.norm(n, axis=1, keepdims=True)
        local_pts.append(base + s * (tip - base) + n)
        labels.append(np.full(c, k))
    local_pts = np.concatenate(local_pts)
    labels = np.concatenate(labels)
    X = np.einsum("ktde,ne->ktnd", Rw, local_pts)[labels, :, np.arange(len(labels))] + pw[labels]
    X = np.swapaxes(X, 0, 1)  # (T, N, 3)

    # Occlusion: random gaps of random length.
    N = len(labels)
    visible = np.ones((T, N), bool)
    if dropout > 0:
        n_gaps = max(1, int(dropout * T / 12))
        for i in range(N):
            for _ in range(n_gaps):
                if rng.uniform() < 0.9:
                    L = rng.integers(4, 24)
                    s0 = rng.integers(0, T)
                    visible[s0 : s0 + L, i] = False
    X = X + rng.normal(scale=noise, size=X.shape) if noise > 0 else X

    palette = (np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
         [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128]]
    ))
    colors = palette[labels % len(palette)].astype(np.uint8)
    if planar:
        X = X[..., :2]
        pw = pw[..., :2]
    tracks = Tracks(X, visible, fps, colors)
    return SyntheticScene(
        tracks=tracks,
        labels=labels,
        parent=np.array([b.parent for b in specs]),
        pivots=pw,
        kinds=["hinge" if (planar or b.kind == "hinge") else "ball" for b in specs],
        names=[b.name for b in specs],
        specs=specs,
    )


__all__ = ["BoneSpec", "SyntheticScene", "creature_spec", "make_scene"]
