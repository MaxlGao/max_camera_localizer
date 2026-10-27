"""Exercise the localizer's OpenCV calls without a camera or GUI."""

import cv2
import numpy as np
import pytest

from risl_camera_localizer import detection_functions as detection
from risl_camera_localizer import drawing_functions as drawing
from risl_camera_localizer.kalman_functions import BlobKalman, QuaternionKalman


CAMERA_MATRIX = np.array(
    [[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]], dtype=np.float32)


@pytest.mark.parametrize('dictionary_id', [
    cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_5X5_250,
])
@pytest.mark.parametrize('marker_ids', [(6, 7), (0, 249)])
def test_marker_detection_and_id_filter(dictionary_id, marker_ids):
    """Detect generated markers using the actual Python ArUco bindings."""
    gray = np.full((400, 700), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    gray[100:300, 50:250] = cv2.aruco.generateImageMarker(dictionary, marker_ids[0], 200)
    gray[100:300, 450:650] = cv2.aruco.generateImageMarker(dictionary, marker_ids[1], 200)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids = detection.detect_markers(
        frame, gray, {'test': dictionary_id}, parameters)
    assert set(ids) == set(marker_ids)
    assert corners[0].shape == (1, 4, 2)
    _, filtered_ids = detection.detect_markers(
        frame, gray, {'test': dictionary_id}, parameters, min_id=6, max_id=6)
    assert filtered_ids == ([6] if 6 in marker_ids else [])
    blank = np.full_like(gray, 255)
    assert detection.detect_markers(
        frame, blank, {'test': dictionary_id}, parameters) == ([], [])


def test_pose_estimation_and_tracking():
    """Recover a known square pose through IPPE and confirm the tracker."""
    object_points = np.array([
        [-.01, .01, 0.], [.01, .01, 0.],
        [.01, -.01, 0.], [-.01, -.01, 0.],
    ], dtype=np.float32)
    translation = np.array([.01, -.02, .4])
    distortion = np.zeros((5, 1), dtype=np.float32)
    corner, _ = cv2.projectPoints(
        object_points, np.array([3., .1, .05]), translation,
        CAMERA_MATRIX, distortion)
    filters, stability, seen = {}, {}, {}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for frame_id in range(1, 9):
        detection.estimate_pose(
            frame, [corner.reshape(1, 4, 2)], [6], CAMERA_MATRIX,
            distortion, .02, filters, stability, seen, frame_id,
            np.zeros(3), [0., 0., 0., 1.], talk=False)
    assert stability[6]['confirmed']
    np.testing.assert_allclose(stability[6]['last_tvec'], translation, atol=1e-4)
    assert seen[6] == 8


def test_color_blob_detection():
    """Exercise HSV thresholds, contours, and the projected blob position."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 20, (255, 0, 0), -1)
    world, image = detection.detect_color_blobs(
        frame, [np.array([100, 80, 80]), np.array([140, 255, 255])],
        (255, 0, 0), CAMERA_MATRIX, np.array([0., 0., -.5]),
        [0., 0., 0., 1.])
    np.testing.assert_allclose(world, [[0., 0., .01]], atol=1e-6)
    # The projection helper truncates floating point coordinates to pixels.
    np.testing.assert_allclose(image, [(320, 240)], atol=1)


def test_kalman_bindings():
    """Predict and correct both float32 filter models."""
    for filter_type in (QuaternionKalman, BlobKalman):
        tracker = filter_type()
        tracker.predict()
        translation = np.array([.01, -.02, .4])
        if filter_type is QuaternionKalman:
            tracker.correct(translation, np.array([3., .1, .05]))
        else:
            tracker.correct(translation)
        assert np.isfinite(tracker.predict()).all()


def test_drawing_bindings():
    """Draw projected contours and telemetry into an image without a window."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    angles = np.linspace(0., 2 * np.pi, 100)
    contour = np.column_stack((
        .01 * np.cos(angles), .01 * np.sin(angles), np.full(100, .4)))
    objects = [{
        'name': 'wrench', 'points': contour[[0, 30, 60]],
        'position': np.array([0., 0., .4]),
        'quaternion': [0., 0., 0., 1.], 'contour': {'xyz': contour},
    }]
    quaternion = [0., 0., 0., 1.]
    telemetry = np.array([1., 2., .1])
    drawing.draw_object_lines(
        frame, CAMERA_MATRIX, np.zeros(3), quaternion, objects)
    drawing.draw_wrench(frame, quaternion, telemetry, telemetry, 0.)
    drawing.draw_twist(frame, quaternion, telemetry, telemetry)
    drawing.draw_text(
        frame, np.zeros(3), quaternion, objects, 1, 0.,
        np.zeros(3), quaternion)
    assert np.count_nonzero(frame) > 0
