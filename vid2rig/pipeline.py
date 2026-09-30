"""End-to-end entry points: video -> tracks -> rig -> files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .export.bvh import write_bvh
from .export.gltf import write_gltf
from .export.viewer import write_viewer
from .parts import segment_parts
from .rig import Rig, build_rig
from .tracks import Tracks


def fit_rig(tracks: Tracks, n_parts: int | None = None, root: int | None = None, **seg_kwargs) -> Rig:
    """Segment tracks into rigid parts and connect them into a skeleton."""
    fit = segment_parts(tracks, n_parts=n_parts, **seg_kwargs)
    return build_rig(tracks, fit, root=root)


def export_rig(rig: Rig, tracks: Tracks, out_dir: str | Path, name: str = "rig",
               bvh_scale: float = 100.0) -> dict[str, Path]:
    """Write BVH, glTF, an HTML viewer, a JSON summary and per-point labels."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "bvh": out / f"{name}.bvh",
        "gltf": out / f"{name}.gltf",
        "viewer": out / f"{name}_viewer.html",
        "summary": out / f"{name}.json",
        "labels": out / f"{name}_labels.npz",
    }
    write_bvh(rig, paths["bvh"], scale=bvh_scale)
    write_gltf(rig, paths["gltf"], colors=tracks.colors)
    write_viewer(rig, tracks, paths["viewer"], title=name)
    paths["summary"].write_text(json.dumps(rig.summary(), indent=2))
    bone_of_part = {p: i for i, p in enumerate(rig.order)}
    bone_labels = np.array([bone_of_part.get(int(l), -1) for l in rig.fit.labels])
    np.savez_compressed(paths["labels"], bone=bone_labels, weights=rig.weights, rest=rig.fit.canonical,
                        names=np.array(rig.names))
    return paths


def video_to_tracks(
    video: str | Path,
    points: Sequence[tuple[float, float]] = (),
    box: tuple[float, float, float, float] | None = None,
    frame_idx: int = 0,  # frame the prompt refers to (after striding)
    mode: str = "3d",
    n_points: int = 2000,
    max_frames: int | None = 200,
    stride: int = 1,
    size: int = 512,
    fov_deg: float = 60.0,
    depth_model: str | None = None,
    disparity: bool = False,
    sam2_model: str | None = None,
    device: str | None = None,
    save_masks: str | Path | None = None,
) -> Tracks:
    """Video -> object mask (SAM 2) -> point tracks (CoTracker) -> optional 3D lift (Depth Anything).

    mode: "3d" lifts tracks with monocular depth; "2d" keeps image-plane tracks.
    """
    from .frontend.depth import image_to_world_2d
    from .frontend.segment import DEFAULT_SAM2, segment_video
    from .frontend.track import track_points
    from .frontend.video import read_video

    frames, fps, scale = read_video(video, max_frames=max_frames, stride=stride, size=size)
    T, H, W, _ = frames.shape
    # Prompts are given in original-video pixels.
    points = [(x * scale, y * scale) for x, y in points]
    box = None if box is None else tuple(c * scale for c in box)
    masks = segment_video(frames, points, box=box, frame_idx=frame_idx, model_id=sam2_model or DEFAULT_SAM2,
                          device=device)
    if save_masks:
        np.savez_compressed(save_masks, masks=masks)
    tracks2d = track_points(frames, masks, n_points=n_points, fps=fps, device=device)
    if mode == "2d":
        return image_to_world_2d(tracks2d, H, W)
    from .frontend.depth import DEFAULT_DEPTH, estimate_depth, lift_tracks

    depth = estimate_depth(frames, model_id=depth_model or DEFAULT_DEPTH, device=device)
    return lift_tracks(tracks2d, depth, masks, fov_deg=fov_deg, disparity=disparity)
