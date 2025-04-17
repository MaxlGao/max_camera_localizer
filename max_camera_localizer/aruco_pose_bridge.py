# aruco_pose_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class ArucoPoseBridge(Node):
    def __init__(self):
        super().__init__('aruco_pose_bridge')
        
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.03, -0.01])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        # --- Latest EE Pose (using values here if no ROS input) ---
        self.ee_position = np.zeros(3)
        self.ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")

        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.marker_pub = self.create_publisher(PoseArray, '/marker_poses', 10)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x,
                                     msg.pose.orientation.y,
                                     msg.pose.orientation.z,
                                     msg.pose.orientation.w])

    def get_camera_pose(self):
        with self.lock:
            r_ee = R.from_quat(self.ee_quat)
            r_cam_offset = R.from_quat(self.cam_offset_quat)
            cam_pos_world = self.ee_position + r_ee.apply(self.cam_offset_position)
            cam_quat_world = (r_ee * r_cam_offset).as_quat()
        return cam_pos_world, cam_quat_world

    def publish_camera_pose(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.cam_pose_pub.publish(msg)

    def publish_marker_poses(self, marker_data):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        for marker_id, (pos, rot) in marker_data.items():
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = rot
            msg.poses.append(p)
        self.marker_pub.publish(msg)

# def start_ros_listener():
#     import threading
#     rclpy.init()
#     node = ArucoPoseBridge()
#     thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     thread.start()
