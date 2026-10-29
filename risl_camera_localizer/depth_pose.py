"""Robust marker-surface depth, using a rectified color optical frame."""

import cv2
import numpy as np


def marker_depth_translation(depth_m, corners, camera_matrix, tvec,
                             min_depth=0.15, max_depth=3.0,
                             max_disagreement=0.10, diagnostics=None):
    """Intersect the marker-origin ray with a robust interior depth plane.

    Reject holes, boundary/background samples, sparse or nonplanar surfaces,
    and gross disagreement with the visual pose. Return None for RGB fallback.
    Depth is optical Z in meters, not Euclidean distance. No temporal filling
    is used, so moving objects cannot inherit an earlier surface's depth.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.clear()
    diagnostics['visual_z_m'] = float(np.asarray(tvec).reshape(3)[2])

    def reject(reason):
        diagnostics['reason'] = reason
        return None

    if depth_m is None:
        return reject('no depth frame')
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    inner = corners.mean(axis=0) + .65 * (corners - corners.mean(axis=0))
    h, w = depth_m.shape
    lo = np.maximum(np.floor(inner.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(inner.max(axis=0)).astype(int), [w - 1, h - 1])
    if np.any(hi < lo):
        return reject('marker outside depth image')
    # Cap work for very large markers while retaining spatial coverage.
    step = max(1, int(np.sqrt(np.prod(hi - lo + 1) / 600)))
    u, v = np.meshgrid(np.arange(lo[0], hi[0] + 1, step),
                       np.arange(lo[1], hi[1] + 1, step))
    pixels = np.column_stack((u.ravel(), v.ravel()))
    mask = np.zeros((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(inner - lo).astype(np.int32), 1)
    inside = mask[pixels[:, 1] - lo[1], pixels[:, 0] - lo[0]].astype(bool)
    pixels = pixels[inside]
    z = depth_m[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    diagnostics['samples'] = int(len(z))
    diagnostics['valid_samples'] = int(valid.sum())
    diagnostics['valid_fraction'] = float(valid.mean()) if len(z) else 0.
    if valid.any():
        diagnostics['median_depth_m'] = float(np.median(z[valid]))
    if valid.sum() < max(12, .5 * len(z)):
        return reject('sparse depth')
    pixels, z = pixels[valid], z[valid]
    median = np.median(z)
    gate = max(.004, 3. * 1.4826 * np.median(np.abs(z - median)))
    keep = np.abs(z - median) <= gate
    if keep.sum() < 12:
        return reject('too few depth inliers')
    rays = np.column_stack((pixels[keep], np.ones(keep.sum()))) @ np.linalg.inv(camera_matrix).T
    points = rays * z[keep, None]
    for _ in range(2):
        center = points.mean(axis=0)
        _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
        if singular[1] < 1e-5:
            return reject('insufficient spatial coverage')
        normal = vt[-1]
        residual = np.abs((points - center) @ normal)
        diagnostics['plane_rms_m'] = float(np.sqrt(np.mean(residual ** 2)))
        keep = residual <= .003
        if keep.sum() < max(12, .7 * len(points)):
            return reject('noisy depth surface')
        if keep.all():
            break
        points = points[keep]
    ray = np.asarray(tvec, dtype=float).reshape(3)
    if not np.isfinite(ray).all() or ray[2] <= 0:
        return reject('invalid visual pose')
    ray = ray / ray[2]
    denominator = normal @ ray
    if abs(denominator) < .1:
        return reject('depth plane nearly edge-on')
    measured_z = (normal @ center) / denominator
    diagnostics['measured_z_m'] = float(measured_z)
    if not min_depth <= measured_z <= max_depth:
        return reject('depth outside configured range')
    if abs(measured_z - np.asarray(tvec).reshape(3)[2]) > max_disagreement:
        return reject('depth disagrees with RGB')
    diagnostics['reason'] = 'accepted'
    return ray * measured_z
