# Vid2Rig

Infer a skeleton rig (joints, bones, joint types and skinning weights) from how an
arbitrary object moves in a video. No template or category prior is needed: parts
that move rigidly together become bones, and the points they rotate about become joints.

```
video ─▶ SAM 2 mask ─▶ CoTracker point tracks ─▶ Depth Anything 3D lift
      ─▶ rigid-part motion segmentation ─▶ joint estimation ─▶ skeleton tree ─▶ skinning
      ─▶ BVH · glTF · interactive HTML viewer
```

The core (everything after the tracks) is only NumPy and SciPy. It works on 2D
(image-plane) or 3D tracks, so you can use any tracker or mocap source.

## Quick start

```bash
pip install -e .            # core only
vid2rig demo --out demo     # synthetic creature -> rig, scored against ground truth
open demo/rig_viewer.html
```

`demo` animates a synthetic 11-bone creature (torso, head, tail, four two-segment legs
with ball hips and hinge knees). It adds tracking noise and 30% occlusion, fits a rig
and prints how the result compares with the ground truth:

```
bones_found 11/11, label_purity 1.0, parents_correct 11/11, joint_type_accuracy 1.0,
pivot_error_mean 0.008 (body length 2.45), ~13 s on a laptop CPU
```

### From a video

```bash
pip install -e ".[video]"   # torch, opencv, transformers, sam2 (CoTracker loads through torch.hub)
# if the sam2 wheel fails to build: pip install git+https://github.com/facebookresearch/sam2.git

# click on the object in frame 0 (original video pixel coordinates)
vid2rig run clip.mp4 --point 420,310 --out out

# or a box, several clicks, planar mode (no depth), fewer frames
vid2rig run clip.mp4 --box 300,200,640,520 --2d --max-frames 120 --out out

# two steps: cache the tracks, then iterate on the fit
vid2rig track clip.mp4 --point 420,310 --save tracks.npz
vid2rig fit tracks.npz --out out --max-parts 12
```

Outputs in `--out`:

| file | contents |
|---|---|
| `rig.bvh` | skeleton + per-frame joint rotations (ZXY), root translation |
| `rig.gltf` | skinned point cloud, joint hierarchy, animation (Blender: File → Import → glTF) |
| `rig_viewer.html` | standalone viewer: orbit, scrub, **click any joint, point or legend row to copy its label** |
| `rig.json` | bones, parents, joint types, hinge axes, ranges of motion, fit residuals |
| `rig_labels.npz` | per-track bone label, skinning weights, rest positions |
| `tracks.npz` | (`run` only) the tracks, for re-fitting with `vid2rig fit` |

## How it works

**1. Tracks.** SAM 2 turns one click into a mask for every frame. CoTracker 3 then
tracks about 2000 points sampled inside the mask. Queries come from several frames, so
parts that show up later still get tracks. A point counts as invisible where CoTracker
says so or where it leaves the mask. In 3D mode each track is lifted with Depth Anything V2
(median depth sampled inside the mask, temporal median filter, per-frame scale
stabilisation) and unprojected with a pinhole camera (`--fov`). Camera motion is fine: it
is absorbed into the root's motion, because the rig only uses motion between parts.

**2. Rigid parts** (`vid2rig/parts.py`). Two points on the same rigid part keep a constant
distance, so the variation of pairwise distances over time gives an affinity matrix.
Spectral clustering of that matrix gives the initial parts. A *rigid k-means* then
alternates between fitting one SE(3) motion per part (weighted Kabsch with Huber weights,
handling occlusion) and moving each point to the motion that best explains its
trajectory. Finally a split/merge loop fixes the part count: a part is split when one
shared motion explains its halves much worse than two motions would, and touching parts
are merged when one motion explains them both. "Parts" that are really just collections of
badly tracked points (residual far above that of a typical part) are dissolved at the end.
Every threshold is relative to a noise level estimated from the data, so no tolerances
depend on scene units.

**3. Joints** (`vid2rig/joints.py`). A joint between parts A and B is a point fixed in
both local frames: `R_A(t) p_A + t_A(t) = R_B(t) p_B + t_B(t)` for every frame. Stacking
all frames gives a linear least-squares problem for the joint centre. A hinge leaves the
centre free along its axis, so a small ridge term pulls it towards where the parts touch.
The relative rotation `R_Aᵀ R_B` decides the joint type: **hinge** (one dominant axis,
which is reported), **ball**, or **fixed**.

**4. Topology** (`vid2rig/rig.py`). A joint is fitted for every pair of parts, and the
cost of each pair is joint residual + distance of the joint from both parts + gap
between the parts. The skeleton is the minimum spanning tree of those costs, rooted at the
largest part. The rest pose is the reference frame (the one with the most visible tracks),
and local rotations come from the part motions.

**5. Skinning.** Each point is weighted by how well each bone's motion explains its
trajectory, using a soft-min over residuals with at most 4 influences. Points near a joint
blend between the two bones automatically.

## Python API

```python
from vid2rig import Tracks, fit_rig, export_rig

tracks = Tracks(points, visible, fps=30)   # points (T, N, 2|3), visible (T, N) bool
rig = fit_rig(tracks)                       # or fit_rig(tracks, n_parts=8)
rig.joint_positions()                       # (bones, T, D) joint trajectories
rig.local_rotations()                       # (bones, T, D, D)
rig.weights                                 # (N, bones) skinning weights
export_rig(rig, tracks, "out")
```

## Limitations

- **A joint can only be found if it moves.** If a joint never bends in the clip, the
  two parts on either side merge into one bone. Use clips with varied motion.
- **Monocular depth is the weakest link.** Per-frame depth models flicker and
  flatten thin parts. A video-consistent depth model, or a 4D reconstruction method, will
  give better 3D tracks. `--2d` avoids depth entirely and gives a planar rig in the image plane.
- **Non-rigid surfaces** (cloth, fur, muscle bulge) show up as extra noise. A strongly
  deforming region can end up as its own small bone.
- Hinge axes need real 3D rotation to be observed. In `--2d` every joint is a hinge
  about the view axis.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests check segmentation, topology, joint types and pivots against synthetic
ground truth in 2D and 3D. One test renders the creature through a pinhole camera
(2D tracks plus a z-buffer depth map with self-occlusion), lifts it back to 3D and fits it.
The tests also check that forward kinematics of the exported BVH reproduces the fitted
joint trajectories, and that glTF and viewer output are well formed. The exported glTF
passes the Khronos glTF validator with no errors or warnings.

The neural front-end (`vid2rig/frontend/segment.py`, `track.py`, `depth.estimate_depth`)
calls SAM 2, CoTracker 3 and Depth Anything V2. It is not covered by the tests because it
needs the model weights.

## References

- Yan & Pollefeys, *Automatic kinematic chain building from feature trajectories of articulated objects*, CVPR 2006
- Ross, Tarlow & Zemel, *Learning articulated structure and motion*, IJCV 2010
- Le & Deng, *Smooth skinning decomposition with rigid bones*, SIGGRAPH Asia 2012
- Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024
- Karaev et al., *CoTracker3: Simpler and better point tracking by pseudo-labelling real videos*, 2024
- Yang et al., *Depth Anything V2*, 2024
