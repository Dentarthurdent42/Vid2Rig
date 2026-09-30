"""Motion segmentation: split point tracks into rigidly moving parts.

1. Pairwise rigidity: points on the same rigid part keep constant distances,
   so the temporal std of pairwise distances gives an affinity matrix.
2. Spectral clustering of that affinity gives an initial labelling (number of
   parts from the eigengap unless given).
3. "Rigid k-means": alternately fit one SE(D) motion per part and reassign
   every point to the motion that explains its trajectory best.
4. Split parts that are not rigid and merge neighbours whose union is rigid,
   using a noise level estimated from the data, so the part count self-corrects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from .geometry import apply_inverse, fill_invalid, kabsch_batch, regauge
from .tracks import Tracks


@dataclass
class PartFit:
    """Result of motion segmentation.

    All motions use the gauge T_k(ref) = identity, so a part's local frame is
    the world frame at the reference frame, and `canonical` holds each point's
    (estimated) position at the reference frame.
    """

    labels: np.ndarray  # (N,) part index, -1 = outlier / unreliable
    R: np.ndarray  # (K, T, D, D)
    t: np.ndarray  # (K, T, D)
    valid: np.ndarray  # (K, T) frames where the part had enough support
    canonical: np.ndarray  # (N, D)
    residual: np.ndarray  # (K, N) mean squared residual of each point under each motion
    ref: int
    noise_var: float

    @property
    def K(self) -> int:
        return self.R.shape[0]


def _pair_std(X, V, min_covis):
    """Std over time of pairwise distances, with co-visibility counts."""
    T, S, _ = X.shape
    s1 = np.zeros((S, S))
    s2 = np.zeros((S, S))
    n = np.zeros((S, S))
    for t in range(T):
        v = V[t]
        if v.sum() < 2:
            continue
        idx = np.flatnonzero(v)
        P = X[t, idx]
        d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        ix = np.ix_(idx, idx)
        s1[ix] += d
        s2[ix] += d * d
        n[ix] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        std = np.sqrt(np.maximum(s2 / n - m * m, 0))
    std[n < min_covis] = np.inf
    return std


def _spectral(A, n_parts, max_parts, rng):
    d = A.sum(1)
    dinv = 1 / np.sqrt(np.maximum(d, 1e-12))
    M = A * dinv[:, None] * dinv[None]
    w, U = np.linalg.eigh(M)
    w, U = w[::-1], U[:, ::-1]
    if n_parts is None:
        m = min(max_parts, len(w) - 1)
        gaps = w[:m] - w[1 : m + 1]
        n_parts = int(np.argmax(gaps)) + 1
    E = U[:, :n_parts]
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if n_parts == 1:
        return np.zeros(len(A), int)
    best, best_cost = None, np.inf
    for _ in range(8):
        c, lab = kmeans2(E, n_parts, minit="++", seed=rng)
        cost = ((E - c[lab]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


class _Fitter:
    def __init__(self, X, V, ref, min_points, huber, touch_dist):
        self.X, self.V, self.ref = X, V, ref
        self.min_points = min_points
        self.huber = huber
        self.touch_dist = touch_dist

    def fit_motion(self, q, members, w_point):
        W = self.V[:, members] * w_point[None]
        R, t, valid = kabsch_batch(q[members], self.X[:, members], W, self.min_points)
        R, t = fill_invalid(R, t, valid)
        R, t = regauge(R, t, self.ref)
        return R, t, valid

    def residuals(self, R, t, valid):
        """Canonical positions and mean squared residual of every point under one motion."""
        Y = apply_inverse(R, t, self.X)
        w = (self.V & valid[:, None]).astype(float)
        n = w.sum(0)
        q = (w[..., None] * Y).sum(0) / np.maximum(n, 1)[:, None]
        r = (w * ((Y - q[None]) ** 2).sum(-1)).sum(0) / np.maximum(n, 1)
        r[n < 2] = np.inf
        return q, r

    def refine(self, labels, iters=15):
        """Rigid k-means: fit one motion per part, reassign points, repeat."""
        labels = labels.copy()
        ks = [k for k in np.unique(labels) if k >= 0 and (labels == k).sum() >= self.min_points]
        if not ks:
            ks, labels = [0], np.zeros_like(labels)
        labels = _compact(labels, ks)
        Q = Rres = None
        for _ in range(iters):
            motions = []
            for k in range(labels.max() + 1):
                members = labels == k
                if Q is None:
                    q, wp = self.X[self.ref], self.V[self.ref, members].astype(float)
                else:
                    # Huber-style per-point weights from the previous residuals.
                    q = Q[k]
                    r = np.sqrt(Rres[k, members])
                    c = self.huber * max(np.median(r), 1e-12)
                    wp = np.nan_to_num(np.minimum(1.0, c / np.maximum(r, 1e-12)), nan=0.0)
                motions.append(self.fit_motion(q, members, wp))
            Q, Rres = map(np.stack, zip(*(self.residuals(*m) for m in motions)))
            new = _argmin_labels(Rres)
            counts = np.bincount(new[new >= 0], minlength=len(motions))
            keep = np.flatnonzero(counts >= self.min_points)
            if len(keep) == 0:
                keep = np.array([np.argmax(counts)])
            if len(keep) < len(motions):
                motions = [motions[k] for k in keep]
                Q, Rres = Q[keep], Rres[keep]
                new = _argmin_labels(Rres)
            converged = np.array_equal(new, labels)
            labels = new
            if converged:
                break
        return labels, motions, Q, Rres


def _argmin_labels(Rres):
    lab = np.argmin(Rres, 0)
    lab[~np.isfinite(Rres.min(0))] = -1
    return lab


def _compact(labels, ks):
    out = np.full_like(labels, -1)
    for j, k in enumerate(ks):
        out[labels == k] = j
    return out


def _assigned_residual(labels, Rres):
    ok = labels >= 0
    r = np.full(len(labels), np.inf)
    r[ok] = Rres[labels[ok], np.flatnonzero(ok)]
    return r


def segment_parts(
    tracks: Tracks,
    n_parts: int | None = None,
    max_parts: int = 16,
    max_cluster_points: int = 800,
    min_frames: int = 5,
    min_points: int | None = None,
    split_gate: float = 1.3,
    split_ratio: float = 2.5,
    merge_ratio: float = 1.5,
    outlier_factor: float = 4.0,
    seed: int = 0,
) -> PartFit:
    """Segment tracks into rigid parts.

    n_parts: force a number of parts (skips eigengap selection and split/merge).
    split_gate: only parts whose mean residual exceeds split_gate x the
        estimated noise variance are candidates for splitting.
    split_ratio / merge_ratio: thresholds on the pair statistic (how much worse
        one shared motion explains two parts than separate motions); a split is
        kept above split_ratio, two parts are merged below merge_ratio.
    """
    rng = np.random.default_rng(seed)
    X, V = tracks.points, tracks.visible.copy()
    T, N, D = X.shape
    ext = tracks.extent()
    eps = (1e-4 * ext) ** 2
    min_points = min_points or max(D + 2, 8)

    reliable = V.sum(0) >= min_frames
    V[:, ~reliable] = False
    ref = int(np.argmax(V.sum(1)))

    # Noise scale from the distance jitter of nearby points (mostly same-part pairs).
    rel_idx = np.flatnonzero(reliable)
    sub = rng.choice(rel_idx, size=min(max_cluster_points, len(rel_idx)), replace=False)
    std = _pair_std(X[:, sub], V[:, sub], min_covis=min_frames)
    vis_ref = V[ref, sub]
    P = X[ref, sub]
    tree = cKDTree(P[vis_ref])
    _, nn = tree.query(P[vis_ref], k=min(6, vis_ref.sum()))
    loc = np.flatnonzero(vis_ref)
    nn_std = std[loc[:, None], loc[nn[:, 1:]]]
    nn_std = np.median(nn_std[np.isfinite(nn_std)]) if np.isfinite(nn_std).any() else 0.0
    noise_var = max(D * nn_std**2 / 2, eps)
    sigma = max(3 * nn_std, 2e-3 * ext)

    A = np.exp(-((std / sigma) ** 2))
    A[~np.isfinite(std)] = 0.0
    np.fill_diagonal(A, 1.0)
    lab_sub = _spectral(A, n_parts, max_parts, rng)

    labels = np.full(N, -1)
    labels[sub] = lab_sub
    fitter = _Fitter(X, V, ref, min_points, huber=2.5, touch_dist=0.05 * ext)
    labels, motions, Q, Rres = fitter.refine(labels)

    if n_parts is None:
        labels, motions, Q, Rres = _split_merge(
            fitter, labels, motions, Q, Rres, sub, std, sigma, noise_var,
            split_gate, split_ratio, merge_ratio, max_parts, rng,
        )

    # Final assignment with outlier rejection.
    r = _assigned_residual(labels, Rres)
    for k in range(len(motions)):
        m = labels == k
        if m.any():
            thr = outlier_factor**2 * max(np.median(r[m]), noise_var)
            labels[m & (r > thr)] = -1
    labels[~reliable] = -1
    canonical = np.zeros((N, D))
    ok = labels >= 0
    canonical[ok] = Q[labels[ok], np.flatnonzero(ok)]
    # Tracks too short to be trusted still get a rest position (and residuals,
    # for skinning) from whichever motion explains their few observations best.
    short = ~reliable & (tracks.visible.sum(0) >= 1)
    if short.any():
        vis_short = tracks.visible[:, short]
        for k, (R, t, valid) in enumerate(motions):
            Y = apply_inverse(R, t, X[:, short])
            w = vis_short.astype(float)
            n = w.sum(0)
            q = (w[..., None] * Y).sum(0) / np.maximum(n, 1)[:, None]
            Q[k][short] = q
            Rres[k, short] = (w * ((Y - q[None]) ** 2).sum(-1)).sum(0) / np.maximum(n, 1) + noise_var
    best = np.argmin(Rres, 0)
    canonical[~ok] = Q[best[~ok], np.flatnonzero(~ok)]

    return PartFit(
        labels=labels,
        R=np.stack([m[0] for m in motions]),
        t=np.stack([m[1] for m in motions]),
        valid=np.stack([m[2] for m in motions]),
        canonical=canonical,
        residual=Rres,
        ref=ref,
        noise_var=float(noise_var),
    )


def _pair_stat(fitter, labels, Q, Rres, a, b, noise_var):
    """How much worse parts a and b are explained by one shared motion.

    For each of the two parts: mean residual of its points under the best
    single motion of the union, divided by its mean residual under its own
    motion. The max over the two is ~1 for a rigid union and large when the
    two move differently (even if one of them is small).
    """
    ma, mb = labels == a, labels == b
    members = ma | mb
    q = np.where(mb[:, None], Q[b], Q[a])
    R, t, valid = fitter.fit_motion(q, members, np.ones(members.sum()))
    _, ru = fitter.residuals(R, t, valid)
    stat = 0.0
    for c, m in ((a, ma), (b, mb)):
        own = max(np.mean(Rres[c, m]), noise_var)
        stat = max(stat, np.mean(ru[m]) / own)
    return stat


def _split_proposals(fitter, labels, Rres, k, K, sub, sub_pos, std, sigma, rng):
    members = np.flatnonzero(labels == k)
    out = []
    # Spectral 2-way cut of the pairwise-rigidity affinity inside the part.
    ms = np.array([sub_pos[i] for i in members if i in sub_pos], dtype=int)
    if len(ms) >= 2 * fitter.min_points:
        A = np.exp(-((std[np.ix_(ms, ms)] / sigma) ** 2))
        A[~np.isfinite(A)] = 0.0
        np.fill_diagonal(A, 1.0)
        half = _spectral(A, 2, 2, rng)
        trial = labels.copy()
        trial[members] = -1
        trial[sub[ms[half == 0]]] = k
        trial[sub[ms[half == 1]]] = K
        out.append(trial)
    # Seed a new part at the worst-fitting point and its spatial neighbourhood.
    X, V = fitter.X, fitter.V
    r = Rres[k, members]
    if len(members) >= 2 * fitter.min_points and np.isfinite(r).any():
        worst = members[np.nanargmax(np.where(np.isfinite(r), r, -1))]
        n_seed = max(fitter.min_points, len(members) // 10)
        # Distance at the frames where the worst point is visible.
        vis = V[:, worst]
        d = np.zeros(len(members))
        cnt = np.zeros(len(members))
        for tt in np.flatnonzero(vis)[:: max(1, vis.sum() // 10)]:
            v = V[tt, members]
            d[v] += np.linalg.norm(X[tt, members[v]] - X[tt, worst], axis=1)
            cnt[v] += 1
        d = np.where(cnt > 0, d / np.maximum(cnt, 1), np.inf)
        seed = members[np.argsort(d)[:n_seed]]
        trial = labels.copy()
        trial[seed] = K
        out.append(trial)
    return out


def _touching(labels, Q, K, dist):
    """(K, K) bool: parts whose rest-pose point sets come within `dist` of each other."""
    pts = [Q[k][labels == k] for k in range(K)]
    trees = [cKDTree(p) if len(p) else None for p in pts]
    out = np.zeros((K, K), bool)
    for a in range(K):
        for b in range(a + 1, K):
            if trees[a] is None or trees[b] is None:
                continue
            d, _ = trees[a].query(pts[b], k=1, distance_upper_bound=dist)
            out[a, b] = out[b, a] = np.isfinite(d).any()
    return out


def _part_medians(labels, Rres):
    r = _assigned_residual(labels, Rres)
    return np.array([np.median(r[labels == k]) if (labels == k).any() else np.inf for k in range(len(Rres))])


def _incoherent(labels, Rres, noise_var, factor):
    """Parts whose typical residual is far above that of a typical part.

    Such "parts" collect badly tracked points rather than a rigid piece of the
    object. The reference is the median over parts, which unlike the local
    noise estimate also reflects spatially correlated errors (e.g. from depth).
    """
    med = _part_medians(labels, Rres)
    ref = max(np.median(med[np.isfinite(med)]), noise_var)
    return med > factor * ref


def _split_merge(fitter, labels, motions, Q, Rres, sub, std, sigma, noise_var,
                 split_gate, split_ratio, merge_ratio, max_parts, rng, dissolve_factor=4.0, rounds=None):
    sub_pos = {p: i for i, p in enumerate(sub)}
    for _ in range(rounds or 3 * max_parts):
        changed = False
        r = _assigned_residual(labels, Rres)
        K = len(motions)
        means = np.array([np.mean(r[labels == k]) if (labels == k).any() else 0 for k in range(K)])
        if K < max_parts:
            for k in np.argsort(-means):
                if means[k] <= split_gate * noise_var:
                    break
                members = labels == k
                best = None
                for trial in _split_proposals(fitter, labels, Rres, k, K, sub, sub_pos, std, sigma, rng):
                    t_lab, t_mot, t_Q, t_R = fitter.refine(trial)
                    if len(t_mot) <= K:
                        continue
                    # The two parts that now hold most of the old part's points.
                    c = np.bincount(t_lab[members & (t_lab >= 0)], minlength=len(t_mot))
                    a, b = np.argsort(-c)[:2]
                    if c[b] < fitter.min_points:
                        continue
                    stat = _pair_stat(fitter, t_lab, t_Q, t_R, a, b, noise_var)
                    if stat > split_ratio and (best is None or stat > best[0]):
                        best = (stat, t_lab, t_mot, t_Q, t_R)
                if best is not None:
                    _, labels, motions, Q, Rres = best
                    changed = True
                    break
        # Merge the pair whose union is most clearly rigid (only parts that touch).
        K = len(motions)
        best = None
        near = _touching(labels, Q, K, fitter.touch_dist)
        for a in range(K):
            for b in range(a + 1, K):
                if not near[a, b]:
                    continue
                stat = _pair_stat(fitter, labels, Q, Rres, a, b, noise_var)
                if stat < merge_ratio and (best is None or stat < best[0]):
                    best = (stat, a, b)
        if best is not None:
            _, a, b = best
            trial = labels.copy()
            trial[trial == b] = a
            labels, motions, Q, Rres = fitter.refine(trial)
            changed = True
        if not changed:
            break
    # Dissolve incoherent parts; their points are reassigned (or become outliers).
    bad = _incoherent(labels, Rres, noise_var, dissolve_factor)
    if bad.any() and (~bad).any():
        trial = labels.copy()
        trial[np.isin(trial, np.flatnonzero(bad))] = -1
        labels, motions, Q, Rres = fitter.refine(trial)
    return labels, motions, Q, Rres
