"""Exercise SDK ownership, calibration, alignment, and metric frame delivery."""

import sys
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from risl_camera_localizer.realsense_camera import RealSenseCamera, build_rectification_maps


def fake_sdk(monkeypatch, fail_read=False, bad_model=False):
    calls = []
    intr = NS(fx=100., fy=101., ppx=4., ppy=3., width=8, height=6,
              coeffs=[0.] * 5, model=99 if bad_model else 0)
    stream = NS(get_intrinsics=lambda: intr, fps=lambda: 30)
    stream.as_video_stream_profile = lambda: stream
    def depth_profile(width, height, fps=30):
        video = NS(width=lambda: width, height=lambda: height,
                   stream_type=lambda: 2, format=lambda: 2, fps=lambda: fps)
        video.as_video_stream_profile = lambda: video
        return video

    profiles = [depth_profile(w, h) for w, h in
                [(424, 240), (640, 360), (848, 480), (1280, 720)]]
    profiles.append(depth_profile(640, 480, fps=60))
    device = NS(first_depth_sensor=lambda: NS(get_depth_scale=lambda: .001,
                                             get_stream_profiles=lambda: profiles),
                get_info=lambda key: 'fake D415', supports=lambda key: True)
    profile = NS(get_stream=lambda kind: stream, get_device=lambda: device)
    color = np.full((6, 8, 3), 120, dtype=np.uint8)
    depth = np.full((6, 8), 500, dtype=np.uint16)
    frames = NS(get_color_frame=lambda: NS(get_data=lambda: color),
                get_depth_frame=lambda: NS(get_data=lambda: depth))

    def wait(**kwargs):
        if fail_read:
            raise RuntimeError('disconnected')
        return frames

    pipeline = NS(start=lambda config: profile,
                  stop=lambda: calls.append('stop'), wait_for_frames=wait,
                  try_wait_for_frames=lambda **kwargs: (True, wait(**kwargs)))
    config = NS(enable_device=lambda serial: calls.append(serial),
                enable_stream=lambda *args: calls.append(args))
    sdk = NS(pipeline=lambda: pipeline, config=lambda: config,
             stream=NS(color=1, depth=2), format=NS(bgr8=1, z16=2),
             distortion=NS(none=0, brown_conrady=1, modified_brown_conrady=2,
                           inverse_brown_conrady=3, ftheta=4, kannala_brandt4=5),
             rs2_project_point_to_pixel=lambda intr, point: [
                 intr.fx * point[0] + intr.ppx, intr.fy * point[1] + intr.ppy],
             camera_info=NS(name=0, serial_number=1, usb_type_descriptor=2),
             align=lambda target: NS(process=lambda data: (
                 calls.append(('align', target)) or data)))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', sdk)
    return calls


def test_metric_depth_and_calibrated_color(monkeypatch):
    calls = fake_sdk(monkeypatch)
    camera = RealSenseCamera(serial='chosen-serial', width=8, height=6)
    try:
        success, frame = camera.read()
        assert success
        np.testing.assert_equal(frame, np.full((6, 8, 3), 120))
        np.testing.assert_allclose(camera.depth_m, .5)
        np.testing.assert_allclose(camera.camera_matrix,
                                   [[100., 0., 4.], [0., 101., 3.], [0., 0., 1.]])
        assert camera.get(cv2.CAP_PROP_FPS) == 30
        assert 'chosen-serial' in calls
        assert ('align', 1) in calls
    finally:
        camera.release()
        camera.release()
    assert calls.count('stop') == 1


def test_initialization_error_stops_pipeline(monkeypatch):
    calls = fake_sdk(monkeypatch, bad_model=True)
    with pytest.raises(RuntimeError, match='Unsupported color distortion'):
        RealSenseCamera()
    assert calls.count('stop') == 1


def test_capture_error_can_be_cleaned_up(monkeypatch):
    calls = fake_sdk(monkeypatch, fail_read=True)
    camera = RealSenseCamera()
    try:
        with pytest.raises(RuntimeError, match='disconnected'):
            camera.read()
    finally:
        camera.release()
    assert calls.count('stop') == 1


def test_missing_sdk_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyrealsense2', None)
    with pytest.raises(RuntimeError, match='pip install pyrealsense2'):
        RealSenseCamera()


@pytest.mark.parametrize('model_name', [
    'none', 'brown_conrady', 'modified_brown_conrady',
    'inverse_brown_conrady', 'ftheta', 'kannala_brandt4',
])
def test_rectification_matches_real_sdk_projection(model_name):
    """Nonzero distortion maps use the real SDK, including inverse Brown."""
    rs = pytest.importorskip('pyrealsense2')
    intr = rs.intrinsics()
    intr.width, intr.height = 32, 24
    intr.fx, intr.fy, intr.ppx, intr.ppy = 22., 23., 15.5, 11.5
    intr.model = getattr(rs.distortion, model_name)
    intr.coeffs = [.15, -.03, .002, -.003, .005]
    map_x, map_y = build_rectification_maps(rs, intr)
    for u, v in [(0, 0), (31, 0), (16, 12), (0, 23), (31, 23)]:
        expected = rs.rs2_project_point_to_pixel(
            intr, [(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, 1.])
        np.testing.assert_allclose([map_x[v, u], map_y[v, u]], expected, atol=1e-5)
    if model_name != 'none':
        assert abs(float(map_x[0, 0])) + abs(float(map_y[0, 0])) > .01


def test_inverse_model_through_capture(monkeypatch):
    """A camera reporting inverse Brown starts and delivers aligned RGB-D."""
    calls = fake_sdk(monkeypatch)
    rs = sys.modules['pyrealsense2']
    original_start = rs.pipeline().start

    def start(config):
        profile = original_start(config)
        profile.get_stream(rs.stream.color).get_intrinsics().model = rs.distortion.inverse_brown_conrady
        return profile

    rs.pipeline().start = start
    camera = RealSenseCamera(width=8, height=6)
    try:
        success, frame = camera.read()
        assert success
        assert frame.shape == (6, 8, 3)
        np.testing.assert_allclose(camera.depth_m, .5)
    finally:
        camera.release()
    assert calls.count('stop') == 1


@pytest.mark.parametrize('misses', [1, 2, 3])
def test_frame_timeouts_are_bounded_and_clear_stale_depth(monkeypatch, misses):
    """Transient timeouts recover; persistent loss fails with useful diagnostics."""
    fake_sdk(monkeypatch)
    rs = sys.modules['pyrealsense2']
    pipeline = rs.pipeline()
    frames = pipeline.wait_for_frames()
    attempts = []

    def wait(timeout_ms):
        attempts.append(timeout_ms)
        return (False, None) if len(attempts) <= misses else (True, frames)

    pipeline.try_wait_for_frames = wait
    camera = RealSenseCamera(width=8, height=6)
    camera.depth_m = np.ones((6, 8))
    try:
        with pytest.warns(RuntimeWarning, match='retrying'):
            if misses == 3:
                with pytest.raises(RuntimeError, match='serial number is optional'):
                    camera.read()
                assert camera.depth_m is None
            else:
                success, _ = camera.read()
                assert success
                np.testing.assert_allclose(camera.depth_m, .5)
        assert len(attempts) == min(misses + 1, 3)
        assert attempts == [5000] * len(attempts)
    finally:
        camera.release()



def test_independent_depth_resolution_and_runtime_switch(monkeypatch):
    """Keep RGB fixed, recreate alignment, and drop stale depth during restart."""
    calls = fake_sdk(monkeypatch)
    camera = RealSenseCamera(width=8, height=6, depth_resolution=(640, 360))
    original_align = camera.align
    try:
        assert (1, 8, 6, 1, 30) in calls
        assert (2, 640, 360, 2, 30) in calls
        assert (640, 480) not in camera.available_depth_resolutions  # 60 FPS only
        camera.read()
        assert camera.depth_m is not None
        assert camera.cycle_depth_resolution()
        assert camera.depth_resolution == (848, 480)
        assert camera.depth_m is None
        assert camera.align is not original_align
        assert 'fake D415' in calls  # restart pins the auto-selected serial
        assert (2, 848, 480, 2, 30) in calls
        success, frame = camera.read()
        assert success and frame.shape == (6, 8, 3)
        np.testing.assert_allclose(camera.depth_m, .5)
        assert camera.set_depth_resolution((640, 360))
    finally:
        camera.release()


def test_failed_depth_switch_restores_previous_profile(monkeypatch):
    calls = fake_sdk(monkeypatch)
    camera = RealSenseCamera(width=8, height=6, depth_resolution=(640, 360))
    original_start = camera.pipeline.start
    attempted = []

    def start(config):
        attempted.append(True)
        if len(attempted) == 1:
            raise RuntimeError('USB profile unavailable')
        return original_start(config)

    camera.pipeline.start = start
    try:
        with pytest.warns(RuntimeWarning, match='restored'):
            assert not camera.set_depth_resolution((848, 480))
        assert camera.started
        assert camera.depth_resolution == (640, 360)
        assert camera.depth_m is None
        assert camera.read()[0]
    finally:
        camera.release()
    assert calls.count('stop') == 2


def test_unadvertised_profile_does_not_stop_stream(monkeypatch):
    calls = fake_sdk(monkeypatch)
    camera = RealSenseCamera(width=8, height=6, depth_resolution=(640, 360))
    try:
        with pytest.warns(RuntimeWarning, match='not advertised'):
            assert not camera.set_depth_resolution((640, 320))
        assert 'stop' not in calls
        assert camera.depth_resolution == (640, 360)
    finally:
        camera.release()


def test_failed_restore_leaves_pipeline_stopped(monkeypatch):
    fake_sdk(monkeypatch)
    camera = RealSenseCamera(width=8, height=6, depth_resolution=(640, 360))

    def fail(config):
        raise RuntimeError('device disconnected')

    camera.pipeline.start = fail
    with pytest.raises(RuntimeError, match='restoring .* also failed'):
        camera.set_depth_resolution((848, 480))
    assert not camera.started
    assert camera.depth_m is None
    camera.release()
