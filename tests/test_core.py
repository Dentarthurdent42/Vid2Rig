import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vid2rig import Tracks, build_rig, estimate_joint, export_rig, segment_parts
from vid2rig.cli import compare_to_ground_truth
from vid2rig.export.bvh import read_bvh_motion
from vid2rig.geometry import apply, kabsch_batch, rot2d
from vid2rig.synthetic import make_scene


@pytest.fixture(scope="module")
def scene3d():
    sc = make_scene(T=90, n_points=900, noise=0.005, dropout=0.3, n_legs=2, seed=1)
    fit = segment_parts(sc.tracks)
    return sc, build_rig(sc.tracks, fit)


@pytest.fixture(scope="module")
def scene2d():
    sc = make_scene(T=90, n_points=900, noise=0.004, dropout=0.2, planar=True, seed=2)
    fit = segment_parts(sc.tracks)
    return sc, build_rig(sc.tracks, fit)


@pytest.mark.parametrize("D", [2, 3])
def test_kabsch_recovers_motion(D):
    rng = np.random.default_rng(0)
    T, N = 10, 50
    q = rng.normal(size=(N, D))
    R = Rotation.random(T, random_state=1).as_matrix() if D == 3 else rot2d(rng.uniform(-3, 3, T))
    t = rng.normal(size=(T, D))
    X = apply(R, t, q)
    R2, t2, valid = kabsch_batch(q, X, np.ones((T, N)))
    assert valid.all()
    assert np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-8)


def _two_part_motion(kind, T=60, seed=0):
    rng = np.random.default_rng(seed)
    RA = Rotation.from_rotvec(rng.normal(scale=0.3, size=(T, 3))).as_matrix()
    tA = rng.normal(size=(T, 3))
    joint = np.array([0.4, -0.2, 0.1])
    ang = np.sin(np.linspace(0, 6, T))
    if kind == "hinge":
        rel = Rotation.from_rotvec(ang[:, None] * np.array([0, 0, 1.0])).as_matrix()
    else:
        rel = Rotation.from_rotvec(np.stack([ang, np.cos(np.linspace(0, 9, T)) * 0.8, 0.5 * ang**2], -1)).as_matrix()
    RB = RA @ rel
    # Joint fixed in both frames: RA j + tA = RB j + tB.
    tB = np.einsum("tde,e->td", RA, joint) + tA - np.einsum("tde,e->td", RB, joint)
    return RA, tA, RB, tB, joint


@pytest.mark.parametrize("kind", ["ball", "hinge"])
def test_estimate_joint(kind):
    RA, tA, RB, tB, joint = _two_part_motion(kind)
    p0 = joint + np.array([0, 0, 0.05])  # prior only fixes the hinge's free axis direction
    j = estimate_joint(RA, tA, RB, tB, np.ones(len(tA), bool), p0)
    assert j.kind == kind
    assert j.rms < 1e-6
    if kind == "ball":
        assert np.allclose(j.pos_parent, joint, atol=1e-4)
    else:
        assert abs(abs(j.axis[2]) - 1) < 1e-6
        assert np.allclose(j.pos_parent[:2], joint[:2], atol=1e-4)


def test_segmentation_and_rig_3d(scene3d):
    sc, rig = scene3d
    score = compare_to_ground_truth(rig, sc)
    assert score["bones_found"] == score["bones_true"] == 7
    assert score["label_purity"] > 0.98
    assert score["parents_correct"] == "7/7"
    assert score["joint_type_accuracy"] == 1.0
    assert score["pivot_error_mean"] < 0.02
    assert score["skinning_rms"] < 0.02


def test_segmentation_and_rig_2d(scene2d):
    sc, rig = scene2d
    score = compare_to_ground_truth(rig, sc)
    assert score["bones_found"] == score["bones_true"] == 7
    assert score["label_purity"] > 0.98
    assert score["parents_correct"] == "7/7"
    assert score["pivot_error_max"] < 0.02


def test_forced_part_count():
    sc = make_scene(T=60, n_points=600, planar=True, seed=3)
    fit = segment_parts(sc.tracks, n_parts=7)
    assert fit.K == 7


def _bvh_fk(path):
    names, parents, offsets, frames = read_bvh_motion(path)
    J = [i for i, n in enumerate(names) if n is not None]
    out = np.zeros((len(frames), len(J), 3))
    for f, row in enumerate(frames):
        c = 0
        world_R, world_p = {}, {}
        for i in J:
            if parents[i] < 0:
                p, rot = row[0:3], row[3:6]
                c = 6
                world_p[i] = offsets[i] + p
                world_R[i] = Rotation.from_euler("ZXY", rot, degrees=True).as_matrix()
            else:
                rot = row[c : c + 3]
                c += 3
                par = parents[i]
                world_p[i] = world_p[par] + world_R[par] @ offsets[i]
                world_R[i] = world_R[par] @ Rotation.from_euler("ZXY", rot, degrees=True).as_matrix()
        out[f] = [world_p[i] for i in J]
    return [names[i] for i in J], out


@pytest.mark.parametrize("which", ["scene3d", "scene2d"])
def test_exports_round_trip(which, request, tmp_path):
    sc, rig = request.getfixturevalue(which)
    paths = export_rig(rig, sc.tracks, tmp_path, bvh_scale=1.0)

    # BVH: forward kinematics of the file reproduces the rig's joint trajectories.
    names, P = _bvh_fk(paths["bvh"])
    ref = rig.joint_positions()
    if rig.D == 2:
        ref = np.concatenate([ref, np.zeros(ref.shape[:-1] + (1,))], -1)
    order = [rig.names.index(n) for n in names]
    assert np.allclose(P, ref[order].transpose(1, 0, 2), atol=1e-3)

    # glTF: well-formed, and node animation reproduces the same joint positions.
    g = json.loads(paths["gltf"].read_text())
    assert g["asset"]["version"] == "2.0"
    assert len(g["skins"][0]["joints"]) == rig.B
    for acc in g["accessors"]:
        assert acc["bufferView"] < len(g["bufferViews"])
    summary = json.loads(paths["summary"].read_text())
    assert len(summary["bones"]) == rig.B

    html = paths["viewer"].read_text()
    blob = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(blob)
    assert len(data["bones"]) == rig.B and data["T"] == rig.T
    assert "clipboard" in html


def test_tracks_io(tmp_path):
    sc = make_scene(T=10, n_points=100, dropout=0.3)
    p = tmp_path / "t.npz"
    sc.tracks.save(p)
    t2 = Tracks.load(p)
    assert np.array_equal(t2.visible, sc.tracks.visible)
    assert np.allclose(t2.points, sc.tracks.points)
    assert np.array_equal(t2.colors, sc.tracks.colors)
