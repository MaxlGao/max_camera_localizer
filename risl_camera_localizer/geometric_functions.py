# geometric_function.py
import cv2
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
import numpy as np

def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]"""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector"""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec

def transform_points_world_to_img(points_world, cam_pos_world, cam_quat_world, camera_matrix):
    image_points = []
    for pt in points_world:
        cam_pt = transform_point_world_to_cam(pt, cam_pos_world, cam_quat_world)
        if cam_pt[2] <= 0.01:
            continue  # skip points behind the camera or too close
        u = int(camera_matrix[0, 0] * cam_pt[0] / cam_pt[2] + camera_matrix[0, 2])
        v = int(camera_matrix[1, 1] * cam_pt[1] / cam_pt[2] + camera_matrix[1, 2])
        image_points.append((u,v))
    return image_points

def transform_point_cam_to_world(point_cam, cam_pos_world, cam_quat_world):
    r_cam_world = R.from_quat(cam_quat_world)
    return cam_pos_world + r_cam_world.apply(point_cam)

def transform_point_world_to_cam(point_world, cam_pos_world, cam_quat_world):
    r_world_cam = R.from_quat(cam_quat_world).inv()
    return r_world_cam.apply(point_world - cam_pos_world)

def transform_orientation_cam_to_world(marker_quat_cam, cam_quat_world):
    r_marker_cam = R.from_quat(marker_quat_cam)
    r_cam_world = R.from_quat(cam_quat_world)
    r_marker_world = r_cam_world * r_marker_cam
    return r_marker_world.as_quat()

def slerp_quat(q1, q2, blend=0.5):
    """Spherical linear interpolation between two quaternions"""
    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)
    rots = R.concatenate([rot1, rot2])
    slerp = scipy.spatial.transform.Slerp([0, 1], rots)
    return slerp(blend).as_quat()

def complete_triangle(p1, p2, side_lengths, tolerance=5.0):
    """
    Given two points p1 and p2, and triangle side lengths (in mm),
    return up to four 3D candidate positions for the third point p3
    that complete the triangle.

    Returns: list of np.array (3D points)
    """

    side_a, side_b, side_c = side_lengths
    d = np.linalg.norm(p1 - p2)
    sides = sorted([side_a, side_b, side_c])

    # Identify which side corresponds to p1-p2
    known_side = None
    for i, s in enumerate(sides):
        if abs(s - 1000 * d) < tolerance:
            known_side = i
            break
    if known_side is None:
        return None  # can't infer triangle

    # Other two sides
    idx = [0, 1, 2]
    idx.remove(known_side)
    s1, s2 = sides[idx[0]] / 1000, sides[idx[1]] / 1000  # convert to meters

    # Basis vectors
    e_x = (p2 - p1) / d

    # Orthogonal vector not aligned with e_x
    e_y = np.cross(e_x, np.array([0, 0, 1]))
    if np.linalg.norm(e_y) < 1e-6:
        e_y = np.cross(e_x, np.array([0, 1, 0]))
    e_y = e_y / np.linalg.norm(e_y)

    e_z = np.cross(e_x, e_y)  # Complete right-handed frame

    # Triangle geometry
    x = (s1**2 - s2**2 + d**2) / (2 * d)
    h_sq = s1**2 - x**2
    if h_sq < 0:
        return None  # no triangle possible
    h = np.sqrt(h_sq)

    # Four candidate points
    p3a = p1 + x * e_x + h * e_y
    p3b = p1 + x * e_x - h * e_y
    p3c = p2 - x * e_x + h * e_y
    p3d = p2 - x * e_x - h * e_y

    # Return unique ones only
    candidates = []
    for p in [p3a, p3b, p3c, p3d]:
        if not any(np.allclose(p, existing, atol=1e-6) for existing in candidates):
            candidates.append(p)

    return candidates

def pick_best_candidate(candidates, prev_position):
    """
    Given a list of candidate points and the previous position of the object,
    return the one closest to the previous position.
    """
    if prev_position is None or len(candidates) == 1:
        return candidates[0]

    distances = [np.linalg.norm(candidate - prev_position) for candidate in candidates]
    best_index = np.argmin(distances)
    return candidates[best_index]

def wrench_world_to_body(wrench_world: np.ndarray, body_theta: float):
    """
    Transforms a wrench from world frame to body frame. 
    This probably shouldn't be used ever. But it's just a matter of negating body angle.
    """
    c, s = np.cos(-body_theta), np.sin(-body_theta)
    rot = np.array([[c, -s], [s, c]])
    wrench_body_lin = rot @ wrench_world[0:2, :]
    wrench_body = np.vstack((wrench_body_lin, wrench_world[2, :]))
    return wrench_body

def wrench_body_to_world(wrench_body: np.ndarray, body_theta: float):
    """
    Transforms a wrench from body frame to world frame. 
    """
    c, s = np.cos(body_theta), np.sin(body_theta)
    rot = np.array([[c, -s], [s, c]])
    wrench_world_lin = rot @ wrench_body[0:2, :]
    wrench_world = np.vstack((wrench_world_lin, wrench_body[2, :]))
    return wrench_world