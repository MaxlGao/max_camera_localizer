"""Check world-plane fitting and tracking with synthetic camera observations."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from risl_camera_localizer.planar_pose import fit_planar_pose, PlanarPoseTracker
from risl_camera_localizer.detection_functions import estimate_pose


K = np.array([[1000., 0., 640.], [0., 980., 360.], [0., 0., 1.]])
DISTORTION = np.array([-.08, .02, .001, -.002, 0.])
CORNERS = np.array([[-.0095, .0095, 0.], [.0095, .0095, 0.],
                    [.0095, -.0095, 0.], [-.0095, -.0095, 0.]])
HEIGHT = .014
CAM_POS = np.array([.1, -.4, .55])


def observation(yaw, cam_quat, noise=0., height=HEIGHT):
    """Project independently constructed marker world corners."""
    rotation = Rotation.from_euler('z', yaw)
    position = np.array([.14, -.43, height])
    corners_world = rotation.apply(CORNERS) + position
    world_to_cam = Rotation.from_quat(cam_quat).inv()
    points_cam = world_to_cam.apply(corners_world - CAM_POS)
    image, _ = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), K, DISTORTION)
    image = image.reshape(4, 2)
    image += np.random.default_rng(12).normal(0., noise, image.shape)
    return image, position, rotation


@pytest.mark.parametrize('camera_angles', [[180., 0., 0.], [162., 12., 20.]])
@pytest.mark.parametrize('yaw', [0., 1.7, -3.13])
@pytest.mark.parametrize('noise', [0., .3])
def test_fit_horizontal_marker(camera_angles, yaw, noise):
    """Recover position and yaw despite camera inclination and corner noise."""
    cam_quat = Rotation.from_euler('xyz', camera_angles, degrees=True).as_quat()
    image, expected_pos, expected_rot = observation(yaw, cam_quat, noise)
    rvec, tvec = fit_planar_pose(image, CORNERS.astype(np.float32), K, DISTORTION,
                                CAM_POS, cam_quat, HEIGHT)
    actual_pos = CAM_POS + Rotation.from_quat(cam_quat).apply(tvec.ravel())
    actual_rot = Rotation.from_quat(cam_quat) * Rotation.from_rotvec(rvec.ravel())
    np.testing.assert_allclose(actual_pos, expected_pos, atol=.0005 if noise else 1e-8)
    np.testing.assert_allclose(actual_rot.as_matrix()[:, 2], [0., 0., 1.], atol=1e-12)
    assert (expected_rot.inv() * actual_rot).magnitude() < (.03 if noise else 1e-7)


def test_plane_behind_camera_is_rejected():
    """Do not silently fall back to a tilted 3D pose for invalid geometry."""
    assert fit_planar_pose(np.array([[600., 300.], [650., 300.],
                                    [650., 350.], [600., 350.]]),
                           CORNERS, K, DISTORTION, CAM_POS, [0., 0., 0., 1.],
                           HEIGHT) is None


@pytest.mark.parametrize('free_height', [False, True])
def test_tracking_yaw_wrap_and_repeated_reads(free_height):
    """Smooth across +/-pi and preserve the world plane under camera motion."""
    cam_quat = Rotation.from_euler('x', 180., degrees=True).as_quat()
    tracker = PlanarPoseTracker(HEIGHT, CAM_POS, cam_quat, free_height=free_height)
    for yaw, height in ((np.deg2rad(179.), .04), (np.deg2rad(-179.), .08)):
        image, _, _ = observation(yaw, cam_quat, height=height if free_height else HEIGHT)
        rvec, tvec = fit_planar_pose(image, CORNERS, K, DISTORTION,
                                    CAM_POS, cam_quat, HEIGHT, free_height=free_height)
        tracker.correct(tvec, rvec)
    tracker.cam_pos = CAM_POS + [.02, -.01, .03]
    tracker.cam_quat = Rotation.from_euler('xyz', [170., 5., 10.], degrees=True).as_quat()
    tvec, rvec = tracker.predict()
    np.testing.assert_allclose(tracker.predict(), (tvec, rvec))
    world_rot = Rotation.from_quat(tracker.cam_quat) * Rotation.from_rotvec(rvec)
    world_pos = tracker.cam_pos + Rotation.from_quat(tracker.cam_quat).apply(tvec)
    np.testing.assert_allclose(world_pos, [.14, -.43, .06 if free_height else HEIGHT], atol=1e-8)
    assert (Rotation.from_euler('z', np.pi).inv() * world_rot).magnitude() < 1e-7


def test_planar_mode_through_detection_and_occlusion(monkeypatch):
    """The published tracker stays flat after confirmation and expires normally."""
    def unexpected_pnp(*args, **kwargs):
        raise AssertionError("Planar mode must not use unconstrained PnP")

    monkeypatch.setattr(cv2, 'solvePnPGeneric', unexpected_pnp)
    cam_quat = Rotation.from_euler('xyz', [170., 7., 0.], degrees=True).as_quat()
    image, position, rotation = observation(.7, cam_quat)
    filters, stability, seen = {}, {}, {}
    for frame_id in range(1, 7):
        estimate_pose(None, [image.reshape(1, 4, 2)], [6], K, DISTORTION,
                      .019, filters, stability, seen, frame_id, CAM_POS, cam_quat,
                      talk=False, pose_mode='planar', marker_height=HEIGHT)
    assert stability[6]['confirmed']
    tvec, rvec = filters[6].predict()
    np.testing.assert_allclose(CAM_POS + Rotation.from_quat(cam_quat).apply(tvec),
                               position, atol=1e-8)
    actual = Rotation.from_quat(cam_quat) * Rotation.from_rotvec(rvec)
    assert (rotation.inv() * actual).magnitude() < 1e-7
    estimate_pose(None, [], [], K, DISTORTION, .019, filters, stability, seen,
                  21, CAM_POS, cam_quat, talk=False, pose_mode='planar')
    assert not stability[6]['confirmed']


@pytest.mark.parametrize('pose_mode', ['planar-z', '3d'])
def test_cli_defaults_and_3d_selection(monkeypatch, pose_mode):
    """The app defaults to planar-z while 3D remains explicitly selectable."""
    from risl_camera_localizer import merged_localization as app

    monkeypatch.setattr(app, 'remove_ros_args', lambda: ['localize'])
    args = app.parse_args()
    assert args.pose_mode == 'planar-z'
    assert args.camera_backend == 'realsense'
    assert args.marker_height == HEIGHT
    monkeypatch.setattr(app, 'remove_ros_args', lambda: [
        'localize', '--pose-mode', pose_mode, '--marker-height', '0.025'])
    args = app.parse_args()
    assert args.pose_mode == pose_mode
    assert args.marker_height == .025


@pytest.mark.parametrize('camera_angles', [[180., 0., 0.], [162., 12., 20.]])
@pytest.mark.parametrize('height', [.014, .042, .15])
@pytest.mark.parametrize('yaw', [-3.13, 1.7])
@pytest.mark.parametrize('noise', [0., .3])
def test_fit_free_height(camera_angles, height, yaw, noise):
    """Estimate metric Z from tag size with no dependence on a height prior."""
    cam_quat = Rotation.from_euler('xyz', camera_angles, degrees=True).as_quat()
    image, expected_pos, expected_rot = observation(yaw, cam_quat, noise, height)
    # Deliberately put the supplied fixed height behind the camera: free Z
    # must initialize independently, using the marker's known metric size.
    result = fit_planar_pose(image, CORNERS.astype(np.float32), K, DISTORTION,
                            CAM_POS, cam_quat, height=1., free_height=True)
    assert result is not None
    rvec, tvec = result
    actual_pos = CAM_POS + Rotation.from_quat(cam_quat).apply(tvec.ravel())
    actual_rot = Rotation.from_quat(cam_quat) * Rotation.from_rotvec(rvec.ravel())
    np.testing.assert_allclose(actual_pos, expected_pos, atol=.006 if noise else 1e-7)
    np.testing.assert_allclose(actual_rot.as_matrix()[:, 2], [0., 0., 1.], atol=1e-12)
    assert (expected_rot.inv() * actual_rot).magnitude() < (.03 if noise else 1e-7)


def test_multiple_marker_heights_and_occlusion(monkeypatch):
    """Independent trackers retain their estimated Z when the camera moves."""
    def unexpected_pnp(*args, **kwargs):
        raise AssertionError("planar-z must not use unconstrained PnP")

    monkeypatch.setattr(cv2, 'solvePnPGeneric', unexpected_pnp)
    cam_quat = Rotation.from_euler('xyz', [170., 7., 0.], degrees=True).as_quat()
    heights = [.014, .042]
    corners = [observation(.7, cam_quat, height=h)[0].reshape(1, 4, 2)
               for h in heights]
    filters, stability, seen = {}, {}, {}
    for frame_id in range(1, 7):
        estimate_pose(None, corners, [6, 7], K, DISTORTION, .019,
                      filters, stability, seen, frame_id, CAM_POS, cam_quat,
                      talk=False, pose_mode='planar-z')
    moved_cam = CAM_POS + [.01, -.02, .03]
    moved_quat = Rotation.from_euler('xyz', [168., 4., 10.], degrees=True).as_quat()
    estimate_pose(None, [], [], K, DISTORTION, .019, filters, stability, seen,
                  7, moved_cam, moved_quat, talk=False, pose_mode='planar-z')
    for marker_id, height in zip([6, 7], heights):
        assert stability[marker_id]['confirmed']
        tvec, rvec = filters[marker_id].predict()
        np.testing.assert_allclose(moved_cam + Rotation.from_quat(moved_quat).apply(tvec),
                                   [.14, -.43, height], atol=1e-7)
        actual = Rotation.from_quat(moved_quat) * Rotation.from_rotvec(rvec)
        np.testing.assert_allclose(actual.as_matrix()[:, 2], [0., 0., 1.], atol=1e-12)
