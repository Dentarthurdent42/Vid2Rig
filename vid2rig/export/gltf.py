"""glTF 2.0 export: skinned point cloud + joint hierarchy + animation (single .gltf file)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import pad3, to3d
from ..rig import Rig


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr: np.ndarray, gltf_type: str, component: int, target=None, minmax=False) -> int:
        arr = np.ascontiguousarray(arr)
        while len(self.data) % 4:
            self.data.append(0)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.data += arr.tobytes()
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": component,
               "count": int(arr.shape[0]), "type": gltf_type}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = flat.min(0).tolist()
            acc["max"] = flat.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


FLOAT, USHORT = 5126, 5123


def write_gltf(rig: Rig, path: str | Path, rest_points: np.ndarray | None = None,
               colors: np.ndarray | None = None) -> None:
    buf = _Buffer()
    B, T = rig.B, rig.T
    pivots = pad3(rig.pivots).astype(np.float32)
    pts = pad3(rig.fit.canonical if rest_points is None else rest_points).astype(np.float32)

    # Skinning attributes: top-4 influences.
    w = rig.weights
    idx = np.argsort(-w, axis=1)[:, :4]
    ww = np.take_along_axis(w, idx, 1)
    if B < 4:
        idx = np.pad(idx, ((0, 0), (0, 4 - idx.shape[1])))
        ww = np.pad(ww, ((0, 0), (0, 4 - ww.shape[1])))
    ww = (ww / ww.sum(1, keepdims=True)).astype(np.float32)
    idx[ww == 0] = 0  # unused influence slots must point at joint 0

    attrs = {
        "POSITION": buf.add(pts, "VEC3", FLOAT, 34962, minmax=True),
        "JOINTS_0": buf.add(idx.astype(np.uint16), "VEC4", USHORT, 34962),
        "WEIGHTS_0": buf.add(ww, "VEC4", FLOAT, 34962),
    }
    if colors is not None:
        attrs["COLOR_0"] = buf.add((np.asarray(colors, np.float32) / 255.0), "VEC3", FLOAT, 34962)

    # Inverse bind matrices: rest pose has identity rotations, so just -pivot (column-major).
    ibm = np.tile(np.eye(4, dtype=np.float32), (B, 1, 1))
    ibm[:, :3, 3] = -pivots
    ibm_acc = buf.add(np.swapaxes(ibm, 1, 2).reshape(B, 16), "MAT4", FLOAT)

    nodes = []
    children = [np.flatnonzero(rig.parent == i).tolist() for i in range(B)]
    for i in range(B):
        tr = pivots[i] if rig.parent[i] < 0 else pivots[i] - pivots[rig.parent[i]]
        node = {"name": rig.names[i], "translation": tr.tolist()}
        if children[i]:
            node["children"] = children[i]
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": "points", "mesh": 0, "skin": 0})

    # Animation.
    L, _ = to3d(rig.local_rotations(), np.zeros((B, T, rig.D)))
    times = (np.arange(T) / rig.fps).astype(np.float32)
    t_acc = buf.add(times, "SCALAR", FLOAT, minmax=True)
    samplers, channels = [], []
    for i in range(B):
        q = Rotation.from_matrix(L[i]).as_quat().astype(np.float32)  # xyzw, as glTF expects
        samplers.append({"input": t_acc, "output": buf.add(_hemisphere(q), "VEC4", FLOAT), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": "rotation"}})
    root_t = pad3(rig.joint_positions()[0]).astype(np.float32)
    samplers.append({"input": t_acc, "output": buf.add(root_t, "VEC3", FLOAT), "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": 0, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "vid2rig"},
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": "tracked_points", "primitives": [{"attributes": attrs, "mode": 0}]}],
        "skins": [{"joints": list(range(B)), "inverseBindMatrices": ibm_acc, "skeleton": 0}],
        "animations": [{"name": "tracked_motion", "samplers": samplers, "channels": channels}],
        "buffers": [{"byteLength": len(buf.data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(buf.data)).decode()}],
        "bufferViews": buf.views,
        "accessors": buf.accessors,
    }
    Path(path).write_text(json.dumps(gltf))


def _hemisphere(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive keys interpolate the short way."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q
