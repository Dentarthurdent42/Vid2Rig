"""Point-track container shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Tracks:
    """Long-range point tracks of one object.

    points:  (T, N, D) positions, D = 2 (image plane) or 3 (lifted). Values at
             invisible entries are ignored.
    visible: (T, N) bool visibility / reliability mask.
    fps:     frame rate, used for exported animation timing.
    colors:  optional (N, 3) uint8 RGB per track, for visualisation.
    """

    points: np.ndarray
    visible: np.ndarray
    fps: float = 30.0
    colors: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.visible = np.asarray(self.visible, dtype=bool)
        if self.points.ndim != 3 or self.points.shape[2] not in (2, 3):
            raise ValueError(f"points must be (T, N, 2|3), got {self.points.shape}")
        if self.visible.shape != self.points.shape[:2]:
            raise ValueError("visible must be (T, N) matching points")
        self.visible &= np.isfinite(self.points).all(-1)
        self.points = np.where(self.visible[..., None], self.points, 0.0)

    @property
    def T(self) -> int:
        return self.points.shape[0]

    @property
    def N(self) -> int:
        return self.points.shape[1]

    @property
    def D(self) -> int:
        return self.points.shape[2]

    def subset(self, idx: np.ndarray) -> "Tracks":
        colors = None if self.colors is None else self.colors[idx]
        return Tracks(self.points[:, idx], self.visible[:, idx], self.fps, colors)

    def extent(self) -> float:
        """Rough object diameter, used to make tolerances scale-free."""
        p = self.points[self.visible]
        if len(p) == 0:
            return 1.0
        lo, hi = np.percentile(p, [2, 98], axis=0)
        return float(max(np.linalg.norm(hi - lo), 1e-9))

    def save(self, path: str | Path) -> None:
        extra = {} if self.colors is None else {"colors": self.colors}
        np.savez_compressed(path, points=self.points, visible=self.visible, fps=self.fps, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "Tracks":
        z = np.load(path)
        colors = z["colors"] if "colors" in z.files else None
        return cls(z["points"], z["visible"], float(z["fps"]) if "fps" in z.files else 30.0, colors)
