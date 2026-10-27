"""Check device selection without opening a camera or GUI."""

from unittest.mock import Mock, call

import numpy as np
import pytest

from risl_camera_localizer import camera_selection as cameras


@pytest.mark.parametrize('platform, backend', [
    ('linux', cameras.cv2.CAP_V4L2), ('darwin', cameras.cv2.CAP_ANY),
])
def test_open_camera_sets_backend_and_resolution(monkeypatch, platform, backend):
    capture = Mock()
    capture.isOpened.return_value = True
    constructor = Mock(return_value=capture)
    monkeypatch.setattr(cameras.sys, 'platform', platform)
    monkeypatch.setattr(cameras.cv2, 'VideoCapture', constructor)

    assert cameras.open_camera(8, 1280, 720) is capture
    constructor.assert_called_once_with(8, backend)
    assert capture.set.call_args_list == [
        call(cameras.cv2.CAP_PROP_FRAME_WIDTH, 1280),
        call(cameras.cv2.CAP_PROP_FRAME_HEIGHT, 720),
    ]


def test_discovery_checks_frames_and_releases_every_capture(monkeypatch):
    captures = [Mock() for _ in range(3)]
    captures[0].isOpened.return_value = False
    captures[1].isOpened.return_value = True
    captures[1].read.return_value = (False, None)
    captures[2].isOpened.return_value = True
    captures[2].read.return_value = (True, np.zeros((720, 1280, 3)))
    opener = Mock(side_effect=captures)
    monkeypatch.setattr(cameras, 'open_camera', opener)

    assert cameras.detect_available_cameras(3) == [2]
    assert opener.call_args_list == [call(i, 1280, 720) for i in range(3)]
    for capture in captures:
        capture.release.assert_called_once()


@pytest.mark.parametrize('keys, expected', [([27, ord('s')], 8), ([27, 27], None)])
def test_preview_skip_and_select(monkeypatch, keys, expected):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    captures = [Mock(), Mock()]
    for capture in captures:
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
    opener = Mock(side_effect=captures)
    labels = Mock()
    cleanup = Mock()
    monkeypatch.setattr(cameras, 'open_camera', opener)
    monkeypatch.setattr(cameras.cv2, 'putText', labels)
    monkeypatch.setattr(cameras.cv2, 'imshow', Mock())
    monkeypatch.setattr(cameras.cv2, 'waitKey', Mock(side_effect=keys))
    monkeypatch.setattr(cameras.cv2, 'destroyAllWindows', cleanup)

    assert cameras.select_camera([1, 8]) == expected
    assert opener.call_args_list == [call(1, 1280, 720), call(8, 1280, 720)]
    assert '1280x720' in labels.call_args_list[0].args[1]
    assert cleanup.call_count == 2
    for capture in captures:
        capture.release.assert_called_once()
