"""Command line interface: `vid2rig demo | fit | track | run`."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _pair(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def _box(s: str) -> tuple[float, float, float, float]:
    v = [float(c) for c in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("box is x0,y0,x1,y1")
    return tuple(v)


def _add_fit_args(p):
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--name", default="rig", help="base name of output files")
    p.add_argument("--parts", type=int, default=None, help="force the number of rigid parts")
    p.add_argument("--max-parts", type=int, default=16)
    p.add_argument("--root", type=int, default=None, help="part index to use as root (default: largest)")


def _add_video_args(p):
    p.add_argument("video")
    p.add_argument("--point", type=_pair, action="append", default=[], help="x,y click on the object (repeatable)")
    p.add_argument("--box", type=_box, default=None, help="x0,y0,x1,y1 box around the object")
    p.add_argument("--frame", type=int, default=0, help="frame index the prompt refers to")
    p.add_argument("--2d", dest="planar", action="store_true", help="skip depth; fit a planar rig in the image plane")
    p.add_argument("--points", type=int, default=2000, help="number of tracked points")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--size", type=int, default=512, help="resize long side to this many pixels")
    p.add_argument("--fov", type=float, default=60.0, help="horizontal field of view in degrees (3D lift)")
    p.add_argument("--depth-model", default=None)
    p.add_argument("--disparity", action="store_true", help="depth model outputs inverse depth")
    p.add_argument("--sam2-model", default=None)
    p.add_argument("--device", default=None)


def _fit_and_export(tracks, args):
    from .pipeline import export_rig, fit_rig

    t0 = time.time()
    rig = fit_rig(tracks, n_parts=args.parts, root=args.root, max_parts=args.max_parts)
    paths = export_rig(rig, tracks, args.out, name=args.name)
    print(f"fitted {rig.B} bones from {tracks.N} tracks x {tracks.T} frames in {time.time() - t0:.1f}s")
    for b in rig.summary()["bones"]:
        j = b["joint"]
        desc = "root" if j is None else f"{j['type']:<5} range {j['range_deg']:6.1f} deg  rms {j['rms']:.4f}"
        print(f"  {b['name']:<10} parent={str(b['parent']):<10} points={b['points']:<5} {desc}")
    for k, p in paths.items():
        print(f"{k:>8}: {p}")
    return rig


def _video_tracks(args):
    from .pipeline import video_to_tracks

    if not args.point and args.box is None:
        raise SystemExit("select the object with --point x,y and/or --box x0,y0,x1,y1")
    return video_to_tracks(
        args.video, points=args.point, box=args.box, frame_idx=args.frame, mode="2d" if args.planar else "3d",
        n_points=args.points, max_frames=args.max_frames, stride=args.stride, size=args.size, fov_deg=args.fov,
        depth_model=args.depth_model, disparity=args.disparity, sam2_model=args.sam2_model, device=args.device,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vid2rig", description="Infer a skeleton rig from the motion of an object in a video.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="fit a rig to a synthetic articulated creature and compare with ground truth")
    _add_fit_args(p)
    p.add_argument("--noise", type=float, default=0.005)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--n-points", type=int, default=1500)
    p.add_argument("--planar", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("fit", help="fit a rig to saved tracks (.npz with points (T,N,D) and visible (T,N))")
    p.add_argument("tracks")
    _add_fit_args(p)

    p = sub.add_parser("track", help="video -> tracks .npz (SAM 2 + CoTracker [+ Depth Anything])")
    _add_video_args(p)
    p.add_argument("--save", default="tracks.npz")

    p = sub.add_parser("run", help="video -> rig in one go")
    _add_video_args(p)
    _add_fit_args(p)

    args = ap.parse_args(argv)
    if args.cmd == "demo":
        from .synthetic import make_scene

        sc = make_scene(T=args.frames, n_points=args.n_points, noise=args.noise, dropout=args.dropout,
                        planar=args.planar, seed=args.seed)
        rig = _fit_and_export(sc.tracks, args)
        print(json.dumps(compare_to_ground_truth(rig, sc), indent=2))
    elif args.cmd == "fit":
        from .tracks import Tracks

        _fit_and_export(Tracks.load(args.tracks), args)
    elif args.cmd == "track":
        tracks = _video_tracks(args)
        tracks.save(args.save)
        print(f"saved {tracks.N} tracks x {tracks.T} frames ({tracks.D}D) to {args.save}")
    elif args.cmd == "run":
        tracks = _video_tracks(args)
        from pathlib import Path

        Path(args.out).mkdir(parents=True, exist_ok=True)
        tracks.save(Path(args.out) / "tracks.npz")
        _fit_and_export(tracks, args)


def compare_to_ground_truth(rig, scene) -> dict:
    """Score a fitted rig against a synthetic scene's ground truth."""
    fit = rig.fit
    ok = fit.labels >= 0
    gt_of = [int(np.bincount(scene.labels[fit.labels == p]).argmax()) for p in rig.order]
    purity = sum(np.bincount(scene.labels[ok & (fit.labels == p)]).max() for p in rig.order) / max(ok.sum(), 1)
    topo = sum(
        (rig.parent[i] < 0 and scene.parent[g] < 0) or (rig.parent[i] >= 0 and gt_of[rig.parent[i]] == scene.parent[g])
        for i, g in enumerate(gt_of)
    )
    err = [float(np.linalg.norm(rig.pivots[i] - scene.pivots[g, fit.ref])) for i, g in enumerate(gt_of) if rig.parent[i] >= 0]
    kinds = [scene.kinds[g] == rig.joints[i].kind for i, g in enumerate(gt_of) if rig.parent[i] >= 0]
    X = rig.skin()
    V = scene.tracks.visible
    return {
        "bones_found": rig.B,
        "bones_true": len(scene.names),
        "label_purity": round(float(purity), 4),
        "parents_correct": f"{topo}/{rig.B}",
        "joint_type_accuracy": round(float(np.mean(kinds)), 4) if kinds else None,
        "pivot_error_mean": round(float(np.mean(err)), 4) if err else None,
        "pivot_error_max": round(float(np.max(err)), 4) if err else None,
        "skinning_rms": round(float(np.sqrt((((X - scene.tracks.points) ** 2).sum(-1))[V].mean())), 5),
        "body_length": round(scene.tracks.extent(), 3),
        "matched_names": {rig.names[i]: scene.names[g] for i, g in enumerate(gt_of)},
    }


if __name__ == "__main__":
    main()
