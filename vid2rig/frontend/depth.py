"""Monocular depth (Depth Anything V2) and lifting 2D tracks to 3D."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from ..tracks import Tracks

DEFAULT_DEPTH = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def estimate_depth(frames: np.ndarray, model_id: str = DEFAULT_DEPTH, device: str | None = None,
                   batch: int = 8) -> np.ndarray:
    """Per-frame depth maps (T, H, W). Metric models give metres; relative models give disparity."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    T, H, W, _ = frames.shape
    out = np.zeros((T, H, W), np.float32)
    with torch.inference_mode():
        for s in range(0, T, batch):
            inputs = proc(images=list(frames[s : s + batch]), return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth  # (b, h, w)
            pred = F.interpolate(pred[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out[s : s + batch] = pred.float().cpu().numpy()
    return out


def image_to_world_2d(tracks: Tracks, height: int, width: int) -> Tracks:
    """Pixel tracks -> planar, y-up coordinates in units of image height, centred on the image."""
    p = tracks.points.copy()
    p[..., 0] = (p[..., 0] - width / 2) / height
    p[..., 1] = -(p[..., 1] - height / 2) / height
    return Tracks(p, tracks.visible, tracks.fps, tracks.colors)


def lift_tracks(
    tracks: Tracks,
    depth: np.ndarray,
    masks: np.ndarray | None = None,
    fov_deg: float = 60.0,
    disparity: bool = False,
    window: int = 2,
    temporal: int = 5,
    stabilize_scale: bool = True,
) -> Tracks:
    """Unproject 2D pixel tracks with per-frame depth into 3D (y-up, camera at origin looking down -Z).

    window: half-size of the median window used to sample depth (only mask pixels count).
    temporal: median filter length applied to each track's depth over time.
    stabilize_scale: rescale each frame's depth so the object's median depth varies smoothly,
        which suppresses the per-frame scale flicker of single-image depth models.
    """
    T, H, W = depth.shape
    z = 1.0 / np.maximum(depth, 1e-6) if disparity else depth.astype(np.float64)
    if stabilize_scale and masks is not None:
        med = np.array([np.median(z[t][masks[t]]) if masks[t].any() else np.nan for t in range(T)])
        ok = np.isfinite(med)
        if ok.sum() > 2:
            med = np.interp(np.arange(T), np.flatnonzero(ok), med[ok])
            smooth = median_filter(med, size=min(9, T), mode="nearest")
            z = z * (smooth / med)[:, None, None]

    u, v = tracks.points[..., 0], tracks.points[..., 1]
    vis = tracks.visible.copy()
    zs = np.full(u.shape, np.nan)
    for t in range(T):
        for i in np.flatnonzero(vis[t]):
            x0, y0 = int(round(u[t, i])), int(round(v[t, i]))
            ys = slice(max(0, y0 - window), min(H, y0 + window + 1))
            xs = slice(max(0, x0 - window), min(W, x0 + window + 1))
            patch = z[t, ys, xs]
            if masks is not None:
                patch = patch[masks[t, ys, xs]]
            if patch.size:
                zs[t, i] = np.median(patch)
    vis &= np.isfinite(zs)

    if temporal > 1:
        for i in range(tracks.N):
            idx = np.flatnonzero(vis[:, i])
            if len(idx) >= temporal:
                zs[idx, i] = median_filter(zs[idx, i], size=temporal, mode="nearest")

    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2)
    X = (u - W / 2) / f * zs
    Y = (v - H / 2) / f * zs
    P = np.stack([X, -Y, -zs], -1)  # OpenCV camera -> y-up, right-handed
    return Tracks(np.where(vis[..., None], P, 0.0), vis, tracks.fps, tracks.colors)
