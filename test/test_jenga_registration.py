"""Exercise instance registration and publication without a ROS executor."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from rclpy.node import Node

from risl_camera_localizer.localizer_bridge import LocalizerBridge
from risl_camera_localizer.merged_localization import select_allocation_object
from risl_camera_localizer.process_stl import CONTOUR_JENGA


def test_allocation_topics_exist_before_detection_and_are_reused(monkeypatch):
    """Run bridge initialization with ROS transport mocked, without detections."""
    monkeypatch.setattr(Node, '__init__', lambda self, *args, **kwargs: None)
    create_publisher = Mock(side_effect=lambda *args: Mock())
    monkeypatch.setattr(LocalizerBridge, 'create_publisher', create_publisher)
    monkeypatch.setattr(LocalizerBridge, 'create_subscription', Mock())
    clock = Mock()
    clock.now.return_value.to_msg.return_value = Time(sec=1)
    monkeypatch.setattr(LocalizerBridge, 'get_clock', lambda self: clock)

    node = LocalizerBridge()
    expected_names = {'jenga_6', 'wrench', 'allen_key'}
    assert set(node.object_pose_publishers) == expected_names
    publisher_groups = (
        ('object_poses', node.object_pose_publishers),
        ('object_twists_world', node.object_vel_world_publishers),
        ('object_twists_body', node.object_vel_body_publishers),
        ('object_twists_push_avg', node.object_vel_avg_publishers),
    )
    topics = {call.args[1] for call in create_publisher.call_args_list}
    for topic_group, publishers in publisher_groups:
        assert set(publishers) == expected_names
        for name, publisher in publishers.items():
            assert f'/vision/{topic_group}/{name}' in topics
            publisher.publish.assert_not_called()
    assert all(not history for history in node.object_poses_2D_history.values())

    # Seeing a different tag first must preserve the pre-created target topics.
    target_publisher = node.object_pose_publishers['jenga_6']
    node.publish_object_poses([
        {'name': 'jenga_7', 'position': [.1, .2, .014],
         'quaternion': [0., 0., 0., 1.]},
    ])
    target_publisher.publish.assert_not_called()
    publisher_count = create_publisher.call_count

    # The allocation target can enter the frame later and reuse its publisher.
    node.publish_object_poses([
        {'name': 'jenga_6', 'position': [.3, .2, .014],
         'quaternion': [0., 0., 0., 1.]},
    ])
    assert node.object_pose_publishers['jenga_6'] is target_publisher
    target_publisher.publish.assert_called_once()
    assert create_publisher.call_count == publisher_count


@pytest.fixture
def bridge():
    node = SimpleNamespace(
        object_poses_2D_history={}, v_b_world={}, v_b_body={},
        v_b_body_data={}, v_b_body_avg={}, object_pose_publishers={},
        object_vel_world_publishers={}, object_vel_body_publishers={},
        object_vel_avg_publishers={}, create_publisher=Mock(),
        get_clock=Mock(), history_size=10, push_active=False,
        contour_publisher_body=Mock(), contour_normals_publisher_body=Mock(),
    )
    node.create_publisher.side_effect = lambda *args: Mock()
    node.register_object = lambda name: LocalizerBridge.register_object(node, name)
    node.get_clock.return_value.now.return_value.to_msg.return_value = (
        Time(sec=1, nanosec=0))
    return node


def test_multiple_jenga_instances_publish_independent_poses_and_twists(bridge):
    objects = [
        {'name': f'jenga_{marker_id}', 'position': [float(marker_id), 0., .014],
         'quaternion': [0., 0., 0., 1.]}
        for marker_id in (0, 6, 7, 249)
    ]
    LocalizerBridge.publish_object_poses(bridge, objects)
    assert bridge.create_publisher.call_count == 4 * len(objects)

    # A second observation reuses publishers and estimates each object's twist.
    bridge.get_clock.return_value.now.return_value.to_msg.return_value.sec = 2
    objects[2]['position'][0] += .1
    LocalizerBridge.publish_object_poses(bridge, objects)
    assert bridge.create_publisher.call_count == 4 * len(objects)
    for obj in objects:
        name = obj['name']
        assert len(bridge.object_poses_2D_history[name]) == 2
        assert bridge.object_pose_publishers[name].publish.call_count == 2
        assert bridge.object_vel_world_publishers[name].publish.call_count == 2
        assert bridge.object_vel_body_publishers[name].publish.call_count == 2
        assert bridge.object_vel_avg_publishers[name].publish.call_count == 1
        expected_vx = .1 if name == 'jenga_7' else 0.
        np.testing.assert_allclose(bridge.v_b_world[name], [expected_vx, 0., 0.])
        np.testing.assert_allclose(bridge.v_b_body[name], [expected_vx, 0., 0.])


@pytest.mark.parametrize('marker_id', [0, 6, 7, 249])
def test_jenga_instances_share_body_contour(bridge, marker_id):
    LocalizerBridge.publish_contour(bridge, f'jenga_{marker_id}')
    points = bridge.contour_publisher_body.publish.call_args.args[0].data
    normals = bridge.contour_normals_publisher_body.publish.call_args.args[0].data
    np.testing.assert_allclose(points, (CONTOUR_JENGA['xyz'][:, :2] / 1000).ravel())
    np.testing.assert_allclose(normals, CONTOUR_JENGA['normals'][:, :2].ravel())


@pytest.mark.parametrize('target_name', [None, 'jenga_6', 'wrench', 'allen_key'])
def test_allocation_contour_ignores_other_jenga_tags(target_name):
    objects = [{'name': name} for name in
               ('jenga_7', 'jenga_249', 'jenga_6', 'wrench', 'allen_key')]
    selected = select_allocation_object(objects, target_name)
    assert selected['name'] == (target_name or 'jenga_6')


@pytest.mark.parametrize('target_name', [None, 'jenga_6', 'jenga_7'])
def test_allocation_does_not_fall_back_to_other_tag(target_name):
    assert select_allocation_object([{'name': 'jenga_7'}], target_name) is None
