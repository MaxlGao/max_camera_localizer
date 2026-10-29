"""Visibility evidence, stacked-block occlusion, and world-fixed retention."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from risl_camera_localizer.object_permanence import ObjectPermanence
from risl_camera_localizer.detection_functions import estimate_pose
from risl_camera_localizer.drawing_functions import draw_object_lines


K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
DIST = np.zeros(5)
SHAPE = (720, 1280, 3)
CAM = np.array([0., 0., .5])
QUAT = Rotation.from_euler('x', 180., degrees=True).as_quat()
IDENTITY = np.array([0., 0., 0., 1.])
MARKER = .019


def pose(x=0., y=0., z=.014, yaw=0.):
    return np.array([x, y, z]), Rotation.from_euler('z', yaw).as_quat()


def state(value):
    return dict(confirmed=True, world_position=value[0].copy(),
                world_quaternion=value[1].copy(), measurement_frame=1)


def classify(target, others=None, camera=CAM, quaternion=QUAT):
    return ObjectPermanence().classify(3, target, others or {}, K, DIST, SHAPE,
                                      camera, quaternion, MARKER)


@pytest.mark.parametrize('yaw', [0., .4, 1.57])
def test_cross_stacked_block_occludes_underlying_tag(yaw):
    target = pose()
    top = pose(z=.028, yaw=yaw)
    visibility, fraction, ids = classify(target, {4: top})
    assert visibility == 'occluded'
    assert fraction >= .3
    assert ids == [4]


def test_full_block_occludes_even_when_occluder_marker_does_not_overlap():
    # The top block extends 36 mm each way, but its marker only 9.5 mm.
    visibility, coverage, _ = classify(pose(x=.025), {4: pose(z=.028)})
    assert visibility == 'occluded'
    assert coverage >= .3


@pytest.mark.parametrize('occluder', [pose(x=.12, z=.028), pose(z=0.), pose(z=.014)])
def test_distant_behind_or_coplanar_block_cannot_protect_target(occluder):
    assert classify(pose(), {4: occluder})[0] == 'visible'


def test_outside_frame_and_partly_clipped_are_protected():
    assert classify(pose(x=1.))[0] == 'out_of_frame'
    # Put marker center at the image edge: it cannot be fully decoded.
    edge = (1280 - 640) * (.5 - .014) / 900
    assert classify(pose(x=edge))[0] == 'out_of_frame'
    assert classify(pose(z=.6))[0] == 'out_of_frame'


def test_back_facing_and_tiny_markers_are_not_counted_as_clear_view():
    assert classify((np.array([0., 0., .014]), QUAT))[0] == 'edge_on'
    assert classify(pose(), camera=np.array([0., 0., 2.]))[0] == 'too_small'


def test_camera_motion_and_oblique_view_recompute_visibility():
    target = pose(x=.03, y=-.02)
    camera = np.array([.04, -.03, .5])
    quaternion = Rotation.from_euler('xyz', [170., 5., 10.], degrees=True).as_quat()
    assert classify(target, {4: pose(x=.03, y=-.02, z=.028)}, camera, quaternion)[0] == 'occluded'
    assert classify(target, {}, np.array([1., 0., .5]))[0] == 'out_of_frame'


def test_only_consecutive_clear_view_misses_drop_tracks():
    policy = ObjectPermanence(visible_miss_frames=3)
    target = pose()
    states = {3: state(target)}
    def update(frame, others):
        return policy.update(states, others, set(), frame, K, DIST, SHAPE, CAM, QUAT, MARKER)
    update(2, {})
    update(3, {})
    assert states[3]['visible_misses'] == 2
    for frame in range(4, 40):
        assert update(frame, {4: pose(z=.028)}) == []
    assert states[3]['visible_misses'] == 0
    assert states[3]['visibility'] == 'occluded'
    assert update(40, {}) == []
    assert update(41, {}) == []
    assert update(42, {}) == [3]
    assert not states[3]['confirmed']
    assert states[3]['hold_counter'] == 0
    assert states[3]['dropped']


def test_remembered_tracks_do_not_protect_each_other():
    states = {3: state(pose()), 4: state(pose(z=.028))}
    policy = ObjectPermanence(visible_miss_frames=2)
    for frame in (2, 3):
        policy.update(states, {}, set(), frame, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert not states[3]['confirmed'] and not states[4]['confirmed']


def test_out_of_frame_retains_until_camera_returns_and_location_is_empty():
    states = {3: state(pose(x=1.))}
    policy = ObjectPermanence(visible_miss_frames=2)
    for frame in range(2, 100):
        policy.update(states, {}, set(), frame, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[3]['confirmed']
    moved = CAM + [1., 0., 0.]
    for frame in (100, 101):
        policy.update(states, {}, set(), frame, K, DIST, SHAPE, moved, QUAT, MARKER)
    assert not states[3]['confirmed']


def test_optional_protected_limit_and_seen_marker_reset():
    states = {3: state(pose(x=1.))}
    policy = ObjectPermanence(protected_miss_frames=2)
    policy.update(states, {}, set(), 2, K, DIST, SHAPE, CAM, QUAT, MARKER)
    policy.update(states, {}, {3}, 3, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[3]['protected_misses'] == 0
    policy.update(states, {}, set(), 4, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[3]['confirmed']
    policy.update(states, {}, set(), 5, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert not states[3]['confirmed']


def corners(value, camera=CAM, quaternion=QUAT):
    half = MARKER / 2
    square = np.array([[-half, half, 0.], [half, half, 0.],
                       [half, -half, 0.], [-half, -half, 0.]])
    world = Rotation.from_quat(value[1]).apply(square) + value[0]
    points = Rotation.from_quat(quaternion).inv().apply(world - camera)
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, DIST)[0].reshape(1, 4, 2)


def test_stack_removed_object_and_camera_motion_through_estimator():
    """The screenshot case: lower tags stay, empty remembered location expires."""
    filters, states, seen = {}, {}, {}
    policy = ObjectPermanence(visible_miss_frames=3)
    values = {3: pose(), 6: pose(y=.004), 4: pose(z=.028, yaw=np.pi/2), 2: pose(x=.12)}
    image = np.zeros(SHAPE, np.uint8)
    def run(frame, ids, camera=CAM):
        estimate_pose(image, [corners(values[mid], camera) for mid in ids], ids,
                      K, DIST, MARKER, filters, states, seen, frame, camera, QUAT,
                      talk=False, pose_mode='planar-z', permanence=policy)
    for frame in range(1, 7):
        run(frame, [3, 6, 4, 2])
    np.testing.assert_allclose(states[3]['world_position'], values[3][0], atol=1e-7)
    # Only the top tag remains visible. The moved block is absent at its old
    # projected location and has no known occluder there.
    for frame in range(7, 20):
        run(frame, [4])
    assert states[3]['confirmed'] and states[6]['confirmed']
    assert states[3]['visibility'] == 'occluded'
    assert not states[2]['confirmed']
    # Move the camera far away; all retained world poses must remain fixed.
    before = states[3]['world_position'].copy()
    run(20, [], CAM + [1., 0., 0.])
    assert states[3]['visibility'] == 'out_of_frame'
    np.testing.assert_array_equal(states[3]['world_position'], before)
    # Seeing the empty location again removes the lower block after 3 misses.
    for frame in range(21, 24):
        run(frame, [])
    assert not states[3]['confirmed']
    # A dropped ID must go through fresh confirmation when it reappears.
    run(24, [3])
    assert not states[3]['confirmed'] and states[3]['hold_counter'] == 1
    for frame in range(25, 29):
        run(frame, [3])
    assert states[3]['confirmed']


def test_drawing_remembered_object_behind_camera_does_not_crash():
    value = pose(z=.6)
    obj = dict(name='jenga_3', points=[value[0]], position=value[0],
               quaternion=value[1], inferred=True, visibility='out_of_frame',
               contour={'xyz': np.tile(value[0], (40, 1))})
    draw_object_lines(np.zeros(SHAPE, np.uint8), K, CAM, QUAT, [obj])



def test_multiple_fresh_blocks_combine_occlusion_evidence():
    policy = ObjectPermanence(occluded_fraction=.7)
    left = pose(x=-.014, z=.028, yaw=np.pi / 2)
    right = pose(x=.014, z=.028, yaw=np.pi / 2)
    def classify(others):
        return policy.classify(3, pose(), others, K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert classify({4: left})[0] == 'visible'
    assert classify({4: left, 5: right})[0] == 'occluded'


@pytest.mark.parametrize('mode', ['planar', 'planar-z', '3d'])
def test_missing_world_pose_is_frozen_in_all_pose_modes(mode):
    filters, states, seen = {}, {}, {}
    policy = ObjectPermanence(visible_miss_frames=2)
    image = np.zeros(SHAPE, np.uint8)
    for frame in range(1, 7):
        estimate_pose(image, [corners(pose())], [3], K, DIST, MARKER,
                      filters, states, seen, frame, CAM, QUAT, talk=False,
                      pose_mode=mode, permanence=policy)
    before = states[3]['world_position'].copy()
    for frame in range(7, 30):
        estimate_pose(image, [], [], K, DIST, MARKER, filters, states, seen,
                      frame, CAM + [1., 0., 0.], QUAT, talk=False,
                      pose_mode=mode, permanence=policy)
    assert states[3]['confirmed']
    np.testing.assert_array_equal(states[3]['world_position'], before)


def test_localizer_publishes_retained_poses_and_stops_removed_pose(monkeypatch):
    """Exercise the application loop and inferred output, with capture/ROS stubbed."""
    from types import SimpleNamespace as NS
    from risl_camera_localizer import merged_localization as app

    published = []
    values = {3: pose(), 6: pose(y=.004), 4: pose(z=.028, yaw=np.pi/2), 2: pose(x=.12)}
    bridge = NS(
        frame_idx=0, start_time=0., allocation_pushers_xyxy=[], v_g_body=[], target_name=None,
        get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: NS(sec=1, nanosec=0))),
        get_ee_pose=lambda: (CAM, QUAT), get_camera_pose=lambda: (CAM, QUAT),
        publish_camera_pose=lambda *args: None, publish_contour=lambda *args: None,
        publish_object_poses=lambda objects: published.append(objects), publish_frame=lambda: None)
    cap = NS(read=lambda: (True, np.zeros(SHAPE, np.uint8)), get=lambda prop: 30.)
    args = NS(blue_blobs=False, object_permanence=True, visible_miss_frames=3,
              protected_miss_frames=0, pose_mode='planar-z', marker_height=.014,
              recommend_push='none', camera_backend='opencv')

    def detections(*args, **kwargs):
        ids = [3, 6, 4, 2] if bridge.frame_idx <= 6 else [4]
        return [corners(values[mid]) for mid in ids], ids

    # Keep the synthetic tag size independent of the user's measured hardware calibration.
    monkeypatch.setattr(app, 'MARKER_SIZE', MARKER)
    monkeypatch.setattr(app, 'detect_markers', detections)
    monkeypatch.setattr(app, 'draw_text', lambda *args: None)
    monkeypatch.setattr(app, 'draw_object_lines', lambda *args: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *args: ord('q') if bridge.frame_idx == 10 else -1)
    app.run_localization(args, cap, K, DIST, bridge, False, {}, {}, {})
    objects = {obj['name']: obj for obj in published[-1]}
    assert set(objects) == {'jenga_3', 'jenga_6', 'jenga_4'}
    assert objects['jenga_3']['inferred']
    assert objects['jenga_3']['visibility'] == 'occluded'
    assert not objects['jenga_4']['inferred']
    np.testing.assert_allclose(objects['jenga_3']['position'], values[3][0], atol=1e-7)


def tower():
    """Two bottom blocks, crosswise middle, and an offset top block."""
    return {10: pose(x=-.012, yaw=np.pi/2),
            20: pose(x=.012, yaw=np.pi/2),
            30: pose(z=.028),
            40: pose(x=.012, z=.042, yaw=np.pi/2)}


@pytest.mark.parametrize('order', [(10, 20, 30, 40), (40, 30, 20, 10), (20, 40, 10, 30)])
def test_protected_middle_covers_both_bottom_blocks(order):
    values = tower()
    policy = ObjectPermanence(visible_miss_frames=3)
    states = {mid: state(values[mid]) for mid in order}
    states[90] = state(pose(x=.12))  # absent from a clear remembered location
    # Reproduce why the previous single-layer policy lost one bottom block.
    assert classify(values[10], {40: values[40]})[0] == 'visible'
    assert classify(values[20], {40: values[40]})[0] == 'occluded'
    for frame in range(2, 30):
        policy.update(states, {40: values[40]}, {40}, frame,
                      K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert all(states[mid]['confirmed'] for mid in values)
    assert not states[90]['confirmed']
    assert states[10]['occlusion_chain'] == [40, 30, 10]
    assert states[10]['occlusion_roots'] == [40]
    assert states[10]['occluders'] == [30]
    assert states[20]['occlusion_chain'] == [40, 20]
    for mid in values:
        np.testing.assert_array_equal(states[mid]['world_position'], values[mid][0])


@pytest.mark.parametrize('top_is_seen', [True, False])
def test_old_chain_breaks_when_visible_anchor_moves_or_disappears(top_is_seen):
    values = tower()
    states = {mid: state(value) for mid, value in values.items()}
    policy = ObjectPermanence(visible_miss_frames=3)
    policy.update(states, {40: values[40]}, {40}, 2,
                  K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[10]['occlusion_chain'] == [40, 30, 10]
    observations = {40: pose(x=.12, z=.042, yaw=np.pi/2)} if top_is_seen else {}
    for frame in range(3, 6):
        policy.update(states, observations, {40} if top_is_seen else set(), frame,
                      K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert not any(states[mid]['confirmed'] for mid in [10, 20, 30])
    assert states[10]['occlusion_chain'] == []


def test_expiring_protected_middle_cannot_extend_occlusion_chain():
    values = tower()
    policy = ObjectPermanence(visible_miss_frames=3, protected_miss_frames=2)
    states = {mid: state(value) for mid, value in values.items()}
    policy.update(states, {40: values[40]}, {40}, 2,
                  K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[10]['confirmed']
    # Give the bottom track a fresh protected budget while the middle is at
    # its cap. Expiring middle geometry must not shield the bottom this frame.
    states[10]['protected_misses'] = 0
    policy.update(states, {40: values[40]}, {40}, 3,
                  K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert not states[30]['confirmed']
    assert states[10]['visibility'] == 'visible'
    assert states[10]['visible_misses'] == 1
    assert states[10]['occlusion_chain'] == []


def test_three_layer_chain_through_actual_pose_estimator():
    filters, states, seen = {}, {}, {}
    values = tower()
    policy = ObjectPermanence(visible_miss_frames=3)
    image = np.zeros(SHAPE, np.uint8)
    for frame in range(1, 20):
        ids = list(values) if frame <= 6 else [40]
        estimate_pose(image, [corners(values[mid]) for mid in ids], ids,
                      K, DIST, MARKER, filters, states, seen, frame, CAM, QUAT,
                      talk=False, pose_mode='planar-z', permanence=policy)
    assert all(states[mid]['confirmed'] for mid in values)
    assert states[10]['occlusion_chain'] == [40, 30, 10]
    assert states[20]['visibility'] == 'occluded'


def test_occlusion_propagates_through_multiple_ghost_layers():
    values = {10: pose(x=-.024, yaw=np.pi/2),
              20: pose(z=.028),
              30: pose(x=.012, z=.042, yaw=np.pi/2),
              40: pose(x=.024, z=.056, yaw=np.pi/2)}
    policy = ObjectPermanence()
    states = {mid: state(value) for mid, value in values.items()}
    policy.update(states, {40: values[40]}, {40}, 2,
                  K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert states[10]['occlusion_chain'] == [40, 30, 20, 10]
    assert all(states[mid]['visibility'] == 'occluded' for mid in [10, 20, 30])


def test_unconfirmed_middle_cannot_protect_bottom_block():
    values = tower()
    states = {mid: state(value) for mid, value in values.items()}
    states[30]['confirmed'] = False
    policy = ObjectPermanence(visible_miss_frames=2)
    for frame in (2, 3):
        policy.update(states, {40: values[40]}, {40}, frame,
                      K, DIST, SHAPE, CAM, QUAT, MARKER)
    assert not states[10]['confirmed']
    assert states[20]['confirmed']  # top directly covers this bottom block
