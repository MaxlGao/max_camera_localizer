"""Fit and track horizontal markers with fixed or estimated world height."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def camera_pose(state, height, cam_pos, cam_quat):
    """Convert world X, Y, yaw (and optional Z) into a camera pose."""
    world_to_cam = Rotation.from_quat(cam_quat).inv()
    position = np.array([state[0], state[1], state[3] if len(state) == 4 else height])
    rotation = world_to_cam * Rotation.from_euler('z', state[2])
    return rotation.as_rotvec(), world_to_cam.apply(position - cam_pos)


def fit_planar_pose(image_points, object_points, camera_matrix, dist_coeffs,
                    cam_pos, cam_quat, height, free_height=False):
    """Minimize corner reprojection error over X, Y, yaw and optionally Z.

    Marker +Z is world +Z. Height is the marker surface's absolute world Z,
    including table elevation, block thickness, and any mounting offset.
    With free_height, recover Z from the known marker size; height is ignored.
    Return None if the plane cannot be intersected in front of the camera.
    """
    # Keep projection output in float64 for numerical optimization derivatives.
    object_points = np.asarray(object_points, dtype=float)
    image_points = np.asarray(image_points, dtype=float).reshape(4, 2)
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_to_world = Rotation.from_quat(cam_quat)
    normalized = cv2.undistortPoints(
        image_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs).reshape(-1, 2)
    rays = cam_to_world.apply(np.column_stack((normalized, np.ones(4))))
    if np.any(np.abs(rays[:, 2]) < 1e-9):
        return None
    # A temporary horizontal plane establishes ray geometry for free Z. Its
    # arbitrary distance is then resolved using the marker's metric size.
    plane_offset = np.sign(np.mean(rays[:, 2])) if free_height else height - cam_pos[2]
    distances = plane_offset / rays[:, 2]
    if np.any(distances <= 0):
        return None
    intersections = cam_pos + distances[:, None] * rays
    source = np.asarray(object_points, dtype=float)[:, :2]
    target = intersections[:, :2]
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    cosine = np.sum(source_centered * target_centered)
    sine = np.sum(source_centered[:, 0] * target_centered[:, 1]
                  - source_centered[:, 1] * target_centered[:, 0])
    initial = [*target.mean(axis=0), np.arctan2(sine, cosine)]
    if free_height:
        spread = np.sum(target_centered ** 2)
        if spread <= 0:
            return None
        scale = np.hypot(cosine, sine) / spread
        if not np.isfinite(scale) or scale <= 0:
            return None
        center = cam_pos[:2] + scale * (target.mean(axis=0) - cam_pos[:2])
        initial = [*center, initial[2], cam_pos[2] + scale * plane_offset]

    def residual(state):
        rvec, tvec = camera_pose(state, height, cam_pos, cam_quat)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs)
        return (projected.reshape(4, 2) - image_points).ravel()

    result = least_squares(residual, initial, x_scale='jac')
    if not result.success or not np.isfinite(result.x).all():
        return None
    rvec, tvec = camera_pose(result.x, height, cam_pos, cam_quat)
    corners_cam = Rotation.from_rotvec(rvec).apply(object_points) + tvec
    if np.any(corners_cam[:, 2] <= 0):
        return None
    return rvec.reshape(3, 1), tvec.reshape(3, 1)


class PlanarPoseTracker:
    """Smooth X, Y, circular yaw and optional Z without advancing on reads."""

    def __init__(self, height, cam_pos, cam_quat, alpha=0.5, free_height=False):
        self.height = height
        self.cam_pos = np.asarray(cam_pos, dtype=float)
        self.cam_quat = np.asarray(cam_quat, dtype=float)
        self.alpha = alpha
        self.free_height = free_height
        self.state = None

    def correct(self, tvec, rvec):
        cam_to_world = Rotation.from_quat(self.cam_quat)
        position = self.cam_pos + cam_to_world.apply(np.asarray(tvec).reshape(3))
        rotation = cam_to_world * Rotation.from_rotvec(np.asarray(rvec).reshape(3))
        axis = rotation.as_matrix()[:, 0]
        measured = np.array([position[0], position[1], np.arctan2(axis[1], axis[0])])
        if self.free_height:
            measured = np.append(measured, position[2])
        if self.state is None:
            self.state = measured
        else:
            delta = measured - self.state
            delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            self.state += self.alpha * delta

    def predict(self):
        rvec, tvec = camera_pose(self.state, self.height, self.cam_pos, self.cam_quat)
        return tvec, rvec
