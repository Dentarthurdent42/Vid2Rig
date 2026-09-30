"""Video decoding helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_video(path: str | Path, max_frames: int | None = None, stride: int = 1,
               size: int | None = 512) -> tuple[np.ndarray, float, float]:
    """Decode a video to (T, H, W, 3) uint8 RGB, downscaled so the long side is at most `size`.

    Returns frames, the effective frame rate (after striding) and the resize
    factor (multiply original-video pixel coordinates by it).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, scale = [], 0, 1.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if size:
                h, w = rgb.shape[:2]
                scale = min(1.0, size / max(h, w))
                if scale < 1:
                    rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps / stride, scale


def write_frames(frames: np.ndarray, directory: str | Path) -> Path:
    """Write frames as 00000.jpg, 00001.jpg, ... (the layout SAM 2 expects)."""
    import cv2

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return d
