"""Dense long-range point tracking inside the object mask with CoTracker 3."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from ..tracks import Tracks


def sample_queries(masks: np.ndarray, n_points: int, n_query_frames: int = 4, erode: int = 3,
                   seed: int = 0) -> np.ndarray:
    """Query points (t, x, y) sampled inside the mask on a few evenly spaced frames.

    Querying several frames lets parts that appear later in the video get tracks too.
    """
    rng = np.random.default_rng(seed)
    T = len(masks)
    qf = np.unique(np.linspace(0, T - 1, n_query_frames).round().astype(int))
    per = int(np.ceil(n_points / len(qf)))
    out = []
    for t in qf:
        m = binary_erosion(masks[t], iterations=erode) if erode else masks[t]
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        pick = rng.choice(len(xs), size=min(per, len(xs)), replace=False)
        out.append(np.stack([np.full(len(pick), t), xs[pick], ys[pick]], -1))
    if not out:
        raise ValueError("mask is empty in every query frame")
    return np.concatenate(out).astype(np.float32)


def track_points(
    frames: np.ndarray,
    masks: np.ndarray,
    n_points: int = 2000,
    n_query_frames: int = 4,
    fps: float = 30.0,
    device: str | None = None,
    model: str = "cotracker3_offline",
    mask_margin: int = 4,
) -> Tracks:
    """Track points on the object. Returns 2D pixel tracks (T, N, 2).

    A tracked point is marked invisible when CoTracker says so or when it falls
    outside the (slightly dilated) object mask.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tracker = torch.hub.load("facebookresearch/co-tracker", model).to(device)
    queries = sample_queries(masks, n_points, n_query_frames)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.inference_mode():
        tr, vis = tracker(video, queries=torch.from_numpy(queries)[None].to(device))
    pts = tr[0].cpu().numpy().astype(np.float64)  # (T, N, 2)
    vis = vis[0].cpu().numpy() > 0.5  # (T, N)

    T, H, W = masks.shape
    grown = np.stack([binary_dilation(m, iterations=mask_margin) for m in masks]) if mask_margin else masks
    xi = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
    inside = grown[np.arange(T)[:, None], yi, xi]
    in_frame = (pts[..., 0] >= 0) & (pts[..., 0] < W) & (pts[..., 1] >= 0) & (pts[..., 1] < H)
    vis &= inside & in_frame
    qt = queries[:, 0].astype(int)
    colors = frames[qt, queries[:, 2].astype(int), queries[:, 1].astype(int)]
    return Tracks(pts, vis, fps, colors)
