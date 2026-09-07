"""
Generic "split one layer into several pieces" utilities for ComfyUI-See-through.

Generalises see-through's ``cluster_inpaint_part`` (hair front/back, KMeans k=2 on depth)
to K pieces with several label sources:

- ``depth_kmeans``     : KMeans on depth only (the original behaviour, any K)
- ``depth_position``   : KMeans on (x, y, depth * weight) -> spatially coherent chunks
- ``lineart_watershed``: seeds from depth_position, boundaries snapped to line-art edges
- ``masks``            : caller-supplied binary masks (mask editor, SAM, ...)

The output pieces are ordered front (index 0) to back. Each piece except the last is
cut out of the running RGB, and the region it covered is inpainted (LaMa or cv2) so the
pieces behind it become complete layers.

Only depends on numpy / cv2 / scikit-learn / scipy.
"""
import numpy as np
import cv2

ALPHA_VISIBLE = 15  # same threshold see-through uses everywhere

SPLIT_MODES = ["depth_position", "lineart_watershed", "depth_kmeans", "masks"]


# ----------------------------------------------------------------------------------
# label generation
# ----------------------------------------------------------------------------------

def _normalize_depth(depth, mask):
    dmin, dmax = float(depth[mask].min()), float(depth[mask].max())
    return (depth - dmin) / (dmax - dmin + 1e-6), dmin, dmax


def _kmeans(features, k, seed=0, max_samples=4000):
    from sklearn.cluster import KMeans
    n = len(features)
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    fit_x = features if n <= max_samples else features[rng.choice(n, max_samples, replace=False)]
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(fit_x)
    return km.predict(features)


def _fill_unlabeled(labels, mask):
    """Assign every masked pixel with label < 0 to the nearest labelled pixel."""
    from scipy import ndimage
    holes = np.bitwise_and(mask, labels < 0)
    if not np.any(holes):
        return labels
    known = np.bitwise_and(mask, labels >= 0)
    if not np.any(known):
        labels[mask] = 0
        return labels
    idx = ndimage.distance_transform_edt(np.bitwise_not(known), return_distances=False, return_indices=True)
    filled = labels[idx[0], idx[1]]
    labels[holes] = filled[holes]
    return labels


def _drop_small_components(labels, mask, min_area):
    """Components smaller than min_area px are dissolved into their neighbours."""
    for lab in np.unique(labels[mask]):
        if lab < 0:
            continue
        m = (labels == lab).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for ci in range(1, n):
            if stats[ci, cv2.CC_STAT_AREA] < min_area:
                labels[comp == ci] = -1
    return _fill_unlabeled(labels, mask)


def _split_components(labels, mask, min_area):
    """Give every large connected component of a label its own label id."""
    out = np.full_like(labels, -1)
    nxt = 0
    for lab in np.unique(labels[mask]):
        if lab < 0:
            continue
        m = (labels == lab).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = [ci for ci in range(1, n) if stats[ci, cv2.CC_STAT_AREA] >= min_area]
        if not big:  # keep the whole cluster as one piece, it will be dissolved later if tiny
            out[m > 0] = nxt
            nxt += 1
            continue
        for ci in big:
            out[comp == ci] = nxt
            nxt += 1
    return _fill_unlabeled(out, mask)


def _merge_to_max(labels, mask, max_pieces):
    """Merge the smallest pieces into their best-connected neighbour until <= max_pieces remain."""
    if max_pieces <= 0:
        return labels
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while True:
        ids, counts = np.unique(labels[mask], return_counts=True)
        ids = [int(i) for i in ids if i >= 0]
        if len(ids) <= max_pieces:
            return labels
        counts = {int(i): int(c) for i, c in zip(*np.unique(labels[mask], return_counts=True)) if i >= 0}
        small = min(ids, key=lambda l: counts[l])
        m = (labels == small).astype(np.uint8)
        ring = np.bitwise_and(cv2.dilate(m, el) > 0, m == 0)
        ring = np.bitwise_and(ring, mask)
        neigh = labels[ring]
        neigh = neigh[neigh >= 0]
        if len(neigh) == 0:
            labels[m > 0] = -1
            labels = _fill_unlabeled(labels, mask)
        else:
            target = int(np.bincount(neigh).argmax())
            labels[m > 0] = target


def _relabel_by_depth(labels, depth, mask):
    """Renumber labels so 0 is the frontmost (smallest median depth)."""
    ids = [int(l) for l in np.unique(labels[mask]) if l >= 0]
    meds = {l: float(np.median(depth[np.bitwise_and(mask, labels == l)])) for l in ids}
    order = sorted(ids, key=lambda l: meds[l])
    remap = {l: i for i, l in enumerate(order)}
    out = np.full_like(labels, -1)
    for l, i in remap.items():
        out[labels == l] = i
    return out, [meds[l] for l in order]


def _lineart_gradient(rgb, mask):
    """Edge strength image used by watershed: strong on drawn lines and hard shading edges."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 30, 5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # dark line-art pixels are boundaries too, even where the gradient is symmetric
    dark = (255 - gray.astype(np.float32))
    edge = grad + 0.5 * dark
    edge[np.bitwise_not(mask)] = 0
    edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edge


def _watershed(rgb, mask, seed_labels, seed_erode):
    """Grow seeds along the least-edge paths. seed_labels: -1 outside, else cluster id."""
    edge = _lineart_gradient(rgb, mask)
    edge3 = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    markers = np.zeros(mask.shape, dtype=np.int32)
    # background marker = everything outside the part
    markers[np.bitwise_not(mask)] = 1
    ids = [int(l) for l in np.unique(seed_labels[mask]) if l >= 0]
    if seed_erode > 0:
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed_erode + 1, 2 * seed_erode + 1))
    for l in ids:
        m = (seed_labels == l).astype(np.uint8)
        core = cv2.erode(m, el) if seed_erode > 0 else m
        if not np.any(core):
            core = m
        markers[core > 0] = l + 2
    ws = cv2.watershed(edge3, markers)
    labels = ws - 2
    labels[ws <= 1] = -1          # background / unknown (-1 in ws)
    labels[np.bitwise_not(mask)] = -1
    return _fill_unlabeled(labels, mask)


def compute_labels(img, depth, mask, mode="depth_position", k=4, depth_weight=1.0,
                   min_area_ratio=0.02, split_components=True, masks=None,
                   order="depth", seed=0, max_pieces=0):
    """
    img   : HxWx4 uint8 RGBA (cropped part)
    depth : HxW float32 in [0,1] (smaller = closer)
    mask  : HxW bool, visible pixels of the part
    masks : list of HxW bool, only for mode == "masks" (front piece first when order == "input")

    returns labels (HxW int32, -1 outside), list of per-label median depth (front to back)
    """
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return np.full((h, w), -1, dtype=np.int32), []
    min_area = max(16, int(area * min_area_ratio))
    dn, _, _ = _normalize_depth(depth.astype(np.float32), mask)
    labels = np.full((h, w), -1, dtype=np.int32)

    if mode == "masks":
        if not masks:
            raise ValueError("mode='masks' needs at least one mask")
        remaining = mask.copy()
        for i, m in enumerate(masks):
            sel = np.bitwise_and(remaining, m)
            labels[sel] = i
            remaining[sel] = False
        if remaining.sum() >= min_area:
            labels[remaining] = len(masks)
        else:
            labels = _fill_unlabeled(labels, mask)
        labels = _drop_small_components(labels, mask, min_area)
        if order == "input":
            ids = [int(l) for l in np.unique(labels[mask]) if l >= 0]
            meds = [float(np.median(depth[np.bitwise_and(mask, labels == l)])) for l in ids]
            remap = np.full(max(ids) + 1, -1, dtype=np.int32)
            for i, l in enumerate(ids):
                remap[l] = i
            out = np.full_like(labels, -1)
            out[mask] = remap[labels[mask]]
            return out, meds
        return _relabel_by_depth(labels, depth, mask)

    ys, xs = np.nonzero(mask)
    if mode == "depth_kmeans":
        feats = dn[ys, xs][:, None]
    else:
        feats = np.stack([xs / float(w), ys / float(h), dn[ys, xs] * float(depth_weight)], axis=1)
    labels[ys, xs] = _kmeans(feats, k, seed=seed)

    if mode == "lineart_watershed":
        seed_erode = max(2, int(round(min(h, w) * 0.01)))
        labels = _watershed(img[..., :3], mask, labels, seed_erode)

    if split_components and mode != "depth_kmeans":
        labels = _split_components(labels, mask, min_area)
    labels = _drop_small_components(labels, mask, min_area)
    labels = _merge_to_max(labels, mask, max_pieces)
    return _relabel_by_depth(labels, depth, mask)


# ----------------------------------------------------------------------------------
# cut + inpaint
# ----------------------------------------------------------------------------------

def _get_inpaint_fn(inpaint):
    if inpaint == "lama":
        from annotators.lama_inpainter import apply_inpaint
        return apply_inpaint
    return lambda im, m, *a, **kw: cv2.inpaint(im, m, 3, cv2.INPAINT_NS)


def unload_lama():
    try:
        import annotators.lama_inpainter as lm
        if getattr(lm, "model", None) is not None:
            lm.model = None
            import torch
            torch.cuda.empty_cache()
    except Exception:
        pass


def split_part_by_labels(img, depth, labels, inpaint="lama", dilate=3, feather_sigma=0.7):
    """
    Cut the part into len(unique labels) pieces, front to back, inpainting the running
    RGB/alpha under each removed piece (same recipe as see-through's cluster_inpaint_part).

    img   : HxWx4 uint8, depth: HxW float32 [0,1], labels: HxW int32 (-1 outside), 0 = front
    returns list of dict(img=RGBA uint8, depth=float32 HxW, depth_median=float)
    """
    rgb = img[..., :3].copy()
    alpha = img[..., 3].copy()
    mask = labels >= 0
    if not np.any(mask):
        return []
    dn, dmin, dmax = _normalize_depth(depth.astype(np.float32), mask)
    d = np.round(np.clip(dn, 0, 1) * 255).astype(np.uint8)
    ids = sorted(int(l) for l in np.unique(labels[mask]))
    inpaint_fn = _get_inpaint_fn(inpaint)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1), (dilate, dilate))

    pieces = []
    for n, lab in enumerate(ids):
        to_mask = labels == lab
        imask = to_mask.astype(np.uint8) * 255
        is_last = n == len(ids) - 1

        if is_last:
            pieces.append({
                "img": np.concatenate([rgb, alpha[..., None]], axis=-1),
                "depth_median": float(np.median(depth[to_mask])),
                "depth": d.astype(np.float32) / 255. * (dmax - dmin + 1e-6) + dmin,
            })
            break

        imask_inpaint = cv2.dilate(imask, element)
        if feather_sigma > 0:
            imask = cv2.GaussianBlur(imask, (3, 3), sigmaX=feather_sigma, sigmaY=feather_sigma)
        piece_alpha = np.minimum(imask, alpha)

        pieces.append({
            "img": np.concatenate([rgb, piece_alpha[..., None]], axis=-1),
            "depth_median": float(np.median(depth[to_mask])),
            "depth": d.astype(np.float32) / 255. * (dmax - dmin + 1e-6) + dmin,
        })

        valid_mask = np.clip(alpha.astype(np.int32) - imask.astype(np.int32), 0, 255) > 50
        if inpaint == "lama":
            a = alpha[..., None] / 255.
            fill = np.array([255] * 3) if (np.any(valid_mask) and np.mean(rgb[valid_mask]) < 100) else np.array([0] * 3)
            rgb = np.round(rgb * a + (1 - a) * fill).astype(np.uint8)

        rgb = inpaint_fn(rgb, imask_inpaint)

        if inpaint == "lama":
            dist_map = np.mean(np.abs(rgb.astype(np.float32) - fill[None, None]), axis=2)
            m = dist_map > 15
            dist_map[m] = 255
            dist_map[imask_inpaint <= 127] = alpha[imask_inpaint <= 127]
            alpha = np.round(dist_map).astype(np.uint8)
        else:
            alpha = inpaint_fn(alpha, imask)
        if np.any(valid_mask):
            d[imask > 127] = np.median(d[valid_mask])

    return pieces


# ----------------------------------------------------------------------------------
# visualisation
# ----------------------------------------------------------------------------------

_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def label_overlay(rgb, labels, alpha=0.55):
    """Tint each label with a distinct colour over the RGB image (uint8 HxWx3)."""
    out = rgb.astype(np.float32).copy()
    ids = [int(l) for l in np.unique(labels) if l >= 0]
    for lab in ids:
        c = _PALETTE[lab % len(_PALETTE)].astype(np.float32)
        sel = labels == lab
        out[sel] = out[sel] * (1 - alpha) + c * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    for lab in ids:  # label number at the piece centroid
        ys, xs = np.nonzero(labels == lab)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(out, str(lab), (max(cx - 6, 0), min(cy + 6, out.shape[0] - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
