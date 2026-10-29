"""Verify RGB-D geometry and fallback without requiring a connected camera."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from risl_camera_localizer.depth_pose import marker_depth_translation
from risl_camera_localizer.detection_functions import estimate_pose

K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
CORNERS = np.array([[300., 220.], [340., 220.], [340., 260.], [300., 260.]])


def test_tilted_surface_with_holes_and_outliers():
    """Recover optical Z from an oblique plane despite isolated bad pixels."""
    v, u = np.indices((480, 640))
    depth = .5 / (1. + .4 * (u - 320) / 800 + .2 * (v - 240) / 800)
    depth[::5, ::5] = 0.
    depth[::7, ::7] = .9
    actual = marker_depth_translation(depth, CORNERS, K, [0., 0., .53])
    np.testing.assert_allclose(actual, [0., 0., .5], atol=1e-7)


@pytest.mark.parametrize('value', [0., np.nan, np.inf, .1, 4., .8])
def test_invalid_or_disagreeing_depth_falls_back(value):
    depth = np.full((480, 640), value)
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5]) is None


def test_only_marker_interior_is_sampled():
    depth = np.full((480, 640), 1.)
    depth[225:256, 305:336] = .5
    np.testing.assert_allclose(
        marker_depth_translation(depth, CORNERS, K, [0., 0., .5]), [0., 0., .5])
    depth[225:256, 305:336] = 0.
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5]) is None


def test_nonplanar_depth_is_rejected():
    rng = np.random.default_rng(10)
    depth = rng.uniform(.47, .53, (480, 640))
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5]) is None


@pytest.mark.parametrize('mode', ['planar', 'planar-z', '3d'])
@pytest.mark.parametrize('valid_depth', [True, False])
def test_depth_through_tracker(mode, valid_depth):
    """Depth corrects scale-derived height; fixed-plane mode stays fixed."""
    half = .019 / 2
    points = np.array([[-half, half, 0.], [half, half, 0.],
                       [half, -half, 0.], [-half, -half, 0.]])
    rotation = Rotation.from_euler('x', 180, degrees=True)
    image, _ = cv2.projectPoints(points, rotation.as_rotvec(),
                                 np.array([0., 0., .5]), K, np.zeros(5))
    filters, stability, seen = {}, {}, {}
    # The visual pose implies Z=.5; measured depth supplies Z=.48 instead.
    depth = np.full((480, 640), .48 if valid_depth else 0.)
    for idx in range(1, 7):
        estimate_pose(None, [image.reshape(1, 4, 2)], [6], K, np.zeros(5),
                      .019, filters, stability, seen, idx,
                      np.array([0., 0., .6]), rotation.as_quat(), talk=False,
                      pose_mode=mode, marker_height=.1, depth_m=depth)
    assert stability[6]['confirmed']
    tvec, _ = filters[6].predict()
    if mode == "3d":
        # The legacy Kalman filter converges gradually from its zero state.
        tvec = stability[6]["last_tvec"]
    assert tvec[2] == pytest.approx(.48 if valid_depth and mode != 'planar' else .5, abs=.001)
    assert stability[6]['depth_source'] == (
        'fixed plane' if mode == 'planar' else 'RGB-D' if valid_depth else 'RGB fallback')


def test_depth_diagnostics_explain_sparse_coverage():
    """Report usable interior coverage while keeping the existing rejection."""
    depth = np.zeros((480, 640))
    depth[::3, ::3] = .5
    info = {'reason': 'stale'}
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5],
                                    diagnostics=info) is None
    assert info['reason'] == 'sparse depth'
    assert 0.05 < info['valid_fraction'] < .2
    assert info['median_depth_m'] == pytest.approx(.5)
    assert info['visual_z_m'] == pytest.approx(.5)


def test_depth_diagnostics_explain_disagreement_and_reset():
    """Distinguish RGB disagreement from absent depth without stale metrics."""
    info = {}
    depth = np.full((480, 640), .8)
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5],
                                    diagnostics=info) is None
    assert info['reason'] == 'depth disagrees with RGB'
    assert info['valid_fraction'] == 1.
    assert info['measured_z_m'] == pytest.approx(.8)
    assert marker_depth_translation(None, CORNERS, K, [0., 0., .5],
                                    diagnostics=info) is None
    assert info['reason'] == 'no depth frame'
    assert 'measured_z_m' not in info


def test_depth_diagnostics_explain_noisy_surface():
    """Dense but inconsistent depth must not be mistaken for sparse depth."""
    info = {}
    depth = np.random.default_rng(10).uniform(.47, .53, (480, 640))
    assert marker_depth_translation(depth, CORNERS, K, [0., 0., .5],
                                    diagnostics=info) is None
    assert info['reason'] == 'noisy depth surface'
    assert info['plane_rms_m'] > .003
