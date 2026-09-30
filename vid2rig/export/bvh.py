"""BVH export (skeleton + per-frame local rotations)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import pad3, to3d
from ..rig import Rig


def write_bvh(rig: Rig, path: str | Path, scale: float = 100.0) -> None:
    """Write the rig as BVH. Channels use ZXY Euler order; units are scene units x scale."""
    L, _ = to3d(rig.local_rotations(), np.zeros((rig.B, rig.T, rig.D)))
    pivots = pad3(rig.pivots) * scale
    tails = pad3(rig.tails) * scale
    root_pos = pad3(rig.joint_positions()[0]) * scale
    children = [np.flatnonzero(rig.parent == i) for i in range(rig.B)]

    lines = ["HIERARCHY"]

    def emit(i, depth):
        ind = "\t" * depth
        if rig.parent[i] < 0:
            lines.append(f"{ind}ROOT {rig.names[i]}")
            off = np.zeros(3)
        else:
            lines.append(f"{ind}JOINT {rig.names[i]}")
            off = pivots[i] - pivots[rig.parent[i]]
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if rig.parent[i] < 0:
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Xrotation Yrotation")
        for c in children[i]:
            emit(c, depth + 1)
        if not len(children[i]):
            e = tails[i] - pivots[i]
            lines.extend([f"{ind}\tEnd Site", f"{ind}\t{{", f"{ind}\t\tOFFSET {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}", f"{ind}\t}}"])
        lines.append(f"{ind}}}")

    order = []

    def dfs(i):
        order.append(i)
        for c in children[i]:
            dfs(c)

    emit(0, 0)
    dfs(0)

    # Intrinsic ZXY (R = Rz Rx Ry), which is how BVH channel order is read.
    eul = np.stack([Rotation.from_matrix(L[i]).as_euler("ZXY", degrees=True) for i in order], 1)  # (T, B, 3)
    lines += ["MOTION", f"Frames: {rig.T}", f"Frame Time: {1.0 / rig.fps:.6f}"]
    for t in range(rig.T):
        vals = list(root_pos[t]) + list(eul[t].reshape(-1))
        lines.append(" ".join(f"{v:.5f}" for v in vals))
    Path(path).write_text("\n".join(lines) + "\n")


def read_bvh_motion(path: str | Path):
    """Minimal BVH reader, used by tests: returns (names, parents, offsets, frames)."""
    names, parents, offsets, stack = [], [], [], []
    lines = Path(path).read_text().splitlines()
    i = 0
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
        elif tok and tok[0] == "End":
            names.append(None)
            parents.append(stack[-1])
        elif tok and tok[0] == "{":
            stack.append(len(names) - 1)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET":
            offsets.append([float(x) for x in tok[1:4]])
        i += 1
    frames = np.array([[float(x) for x in l.split()] for l in lines[i + 3 :] if l.strip()])
    return names, parents, np.array(offsets), frames
