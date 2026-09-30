"""vid2rig: infer a skeleton (joints, bones, skinning) from the motion of an object in a video."""
from .joints import Joint, estimate_joint
from .parts import PartFit, segment_parts
from .pipeline import export_rig, fit_rig, video_to_tracks
from .rig import Rig, build_rig
from .tracks import Tracks

__all__ = [
    "Joint", "PartFit", "Rig", "Tracks", "build_rig", "estimate_joint", "export_rig", "fit_rig",
    "segment_parts", "video_to_tracks",
]
__version__ = "0.1.0"
