# aruco_pose_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class ArucoPoseBridge(Node):
    def __init__(self):
        super().__init__('aruco_pose_bridge')
        
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.048, -0.01])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        # --- Latest EE Pose (using values here if no ROS input - Home position) ---
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")

        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.marker_publishers = {}  # { marker_id: publisher }
        self.object_publishers = {}
        self.contact_position_publishers = {}  # { object_name: { idx: publisher } }
        self.contact_normal_publishers = {}    # { object_name: { idx: publisher } }

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def get_ee_pose(self):
        return self.ee_position, self.ee_quat

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
        now = self.get_clock().now().to_msg()

        for marker_id, (pos, rot, contacts) in marker_data.items():
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = rot

            # Publish individual PoseStamped per marker
            if marker_id not in self.marker_publishers:
                topic = f'/marker_poses/marker_{marker_id}'
                self.marker_publishers[marker_id] = self.create_publisher(PoseStamped, topic, 10)
                self.get_logger().info(f"Created publisher for marker {marker_id} -> {topic}")

            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "base"
            pose_msg.pose = p
            self.marker_publishers[marker_id].publish(pose_msg)

    def publish_object_poses(self, object_data):
        now = self.get_clock().now().to_msg()

        for obj in object_data:
            name = obj["name"]
            pos = obj["position"]
            quat = obj["quaternion"]
            contacts = obj["contacts"]
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat

            if name not in self.object_publishers:
                topic = f'/object_poses/object_{name}'
                self.object_publishers[name] = self.create_publisher(PoseStamped, topic, 10)
                self.get_logger().info(f"Created publisher for object {name} -> {topic}")

            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "base"
            pose_msg.pose = p
            self.object_publishers[name].publish(pose_msg)
            
            for idx, position, normal in contacts:
                # Initialize nested dicts if needed
                if name not in self.contact_position_publishers:
                    self.contact_position_publishers[name] = {}
                    self.contact_normal_publishers[name] = {}

                # Create publishers if not already created
                if idx not in self.contact_position_publishers[name]:
                    pos_topic = f'/object_poses/{name}/contacts/c_{idx}/position'
                    norm_topic = f'/object_poses/{name}/contacts/c_{idx}/normal'
                    self.contact_position_publishers[name][idx] = self.create_publisher(PointStamped, pos_topic, 10)
                    self.contact_normal_publishers[name][idx] = self.create_publisher(Vector3Stamped, norm_topic, 10)
                    self.get_logger().info(f"Created contact publishers for object {name}, contact {idx}")

                # Publish position
                pos_msg = PointStamped()
                pos_msg.header.stamp = now
                pos_msg.header.frame_id = "base"
                pos_msg.point.x, pos_msg.point.y, pos_msg.point.z = position
                self.contact_position_publishers[name][idx].publish(pos_msg)

                # Publish normal
                norm_msg = Vector3Stamped()
                norm_msg.header.stamp = now
                norm_msg.header.frame_id = "base"
                norm_msg.vector.x, norm_msg.vector.y, norm_msg.vector.z = normal
                self.contact_normal_publishers[name][idx].publish(norm_msg)
