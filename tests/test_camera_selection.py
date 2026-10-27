from risl_camera_localizer.camera_selection import find_camera_id_from_text


def test_find_camera_id_from_text_returns_exact_match():
    assert find_camera_id_from_text("2", [0, 2, 5]) == 2
    assert find_camera_id_from_text("10", [1, 10, 15]) == 10


def test_find_camera_id_from_text_rejects_unknown_camera():
    assert find_camera_id_from_text("9", [0, 2, 5]) is None
