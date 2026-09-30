"""Front-end pieces that don't need the neural models."""
import numpy as np

from vid2rig import Tracks
from vid2rig.frontend.depth import image_to_world_2d, lift_tracks
from vid2rig.frontend.track import sample_queries


def test_sample_queries_inside_mask():
    masks = np.zeros((20, 64, 80), bool)
    masks[:, 10:40, 20:60] = True
    q = sample_queries(masks, 200, n_query_frames=4, erode=2)
    assert len(q) >= 200
    t, x, y = q.T
    assert set(np.unique(t)) <= {0, 6, 13, 19}
    assert masks[t.astype(int), y.astype(int), x.astype(int)].all()


def test_lift_tracks_unprojects_with_depth():
    T, H, W = 3, 60, 80
    uv = np.array([[40.0, 30.0], [70.0, 10.0]])  # image centre, and an off-centre pixel
    pts = np.repeat(uv[None], T, 0)
    depth = np.full((T, H, W), 2.0)
    tr = lift_tracks(Tracks(pts, np.ones((T, 2), bool)), depth, fov_deg=90, temporal=1, stabilize_scale=False)
    f = 0.5 * W / np.tan(np.deg2rad(45))
    assert np.allclose(tr.points[0, 0], [0, 0, -2])
    assert np.allclose(tr.points[0, 1], [(70 - 40) / f * 2, -(10 - 30) / f * 2, -2])


def test_image_to_world_2d_is_y_up():
    pts = np.array([[[50.0, 0.0], [50.0, 100.0]]])
    tr = image_to_world_2d(Tracks(pts, np.ones((1, 2), bool)), height=100, width=100)
    assert tr.points[0, 0, 1] > tr.points[0, 1, 1]


def _render(scene, H=320, W=400, fov_deg=60.0, cam_dist=4.0, splat=2):
    """Project a synthetic scene through a pinhole camera: 2D tracks, z-buffer depth, masks."""
    X = scene.tracks.points.copy()
    X[..., 2] -= cam_dist  # object in front of a camera looking down -Z
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2)
    z = -X[..., 2]
    u = f * X[..., 0] / z + W / 2
    v = -f * X[..., 1] / z + H / 2
    T, N = z.shape
    depth = np.full((T, H, W), np.inf)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            yy, xx = np.clip(vi + dy, 0, H - 1), np.clip(ui + dx, 0, W - 1)
            for t in range(T):
                np.minimum.at(depth[t], (yy[t], xx[t]), z[t])
    masks = np.isfinite(depth)
    # A point is occluded when something is clearly in front of it.
    front = depth[np.arange(T)[:, None], np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)]
    visible = scene.tracks.visible & (z <= front + 0.03)
    return Tracks(np.stack([u, v], -1), visible, scene.tracks.fps), np.where(masks, depth, 0.0), masks


def test_depth_lift_then_fit():
    """2D tracks + rendered depth -> lift -> rig recovers the synthetic skeleton."""
    from vid2rig import fit_rig
    from vid2rig.cli import compare_to_ground_truth
    from vid2rig.synthetic import make_scene

    sc = make_scene(T=90, n_points=1200, n_legs=2, seed=1)
    tr2d, depth, masks = _render(sc)
    tr3d = lift_tracks(tr2d, depth, masks, fov_deg=60.0, temporal=1)
    assert 0.25 < tr3d.visible.mean() <= sc.tracks.visible.mean()  # back faces are hidden
    rig = fit_rig(tr3d)

    # Score against ground truth expressed in the same (camera) frame.
    sc.tracks = tr3d
    sc.pivots = sc.pivots - np.array([0, 0, 4.0])
    score = compare_to_ground_truth(rig, sc)
    assert score["bones_found"] == 7
    assert score["parents_correct"] == "7/7"
    assert score["label_purity"] > 0.95
    assert score["pivot_error_mean"] < 0.08
