"""Object segmentation through a video with SAM 2."""
from __future__ import annotations

import tempfile
from typing import Sequence

import numpy as np

from .video import write_frames

DEFAULT_SAM2 = "facebook/sam2.1-hiera-small"


def segment_video(
    frames: np.ndarray,
    points: Sequence[tuple[float, float]] = (),
    point_labels: Sequence[int] | None = None,
    box: tuple[float, float, float, float] | None = None,
    frame_idx: int = 0,
    model_id: str = DEFAULT_SAM2,
    device: str | None = None,
) -> np.ndarray:
    """Masks (T, H, W) bool for the object picked by clicks and/or a box on `frame_idx`.

    points are (x, y) pixels; point_labels are 1 (object) / 0 (background), default all 1.
    """
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    if not len(points) and box is None:
        raise ValueError("give at least one point or a box to select the object")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    T, H, W, _ = frames.shape
    masks = np.zeros((T, H, W), bool)

    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        write_frames(frames, tmp)
        state = predictor.init_state(video_path=tmp)
        kw = {}
        if len(points):
            kw["points"] = np.asarray(points, np.float32)
            kw["labels"] = np.asarray(point_labels if point_labels is not None else [1] * len(points), np.int32)
        if box is not None:
            kw["box"] = np.asarray(box, np.float32)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=frame_idx, obj_id=1, **kw)
        for reverse in (False, True):
            if reverse and frame_idx == 0:
                continue
            for fi, _, logits in predictor.propagate_in_video(state, start_frame_idx=frame_idx, reverse=reverse):
                masks[fi] = (logits[0, 0] > 0).float().cpu().numpy() > 0.5
    return masks
