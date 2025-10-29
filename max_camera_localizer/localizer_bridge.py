# localizer_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point
from std_msgs.msg import Header, ColorRGBA, Int32, Float32MultiArray
from max_camera_msgs.msg import PusherInfo
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class LocalizerBridge(Node):
    def __init__(self):
        super().__init__('localizer_bridge')
        
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
        self.allocation_contacts_sub = self.create_subscription(
            Float32MultiArray, "/allocation/push_points", self.push_point_callback, 10)
        self.allocation_pushers_xyxy = []
        self.allocation_desired = self.create_subscription(
            Float32MultiArray, "/allocation/wrench_desired", self.wrench_desired_callback, 10)
        self.w_d = None


        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.object_publishers = {}
        self.pusher_publishers = {}
        self.frame_num_publsher = self.create_publisher(Int32, '/camera_frame_number', 10)
        self.recommended_publishers = {"pusher_1_position": self.create_publisher(PointStamped, '/recommended_pusher_1/position', 10), 
                                       "pusher_2_position": self.create_publisher(PointStamped, '/recommended_pusher_2/position', 10),
                                       "pusher_1_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_1/normal', 10), 
                                       "pusher_2_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_2/normal', 10)}
        self.contour_publisher = self.create_publisher(Float32MultiArray, '/vision/boundary_points', 10)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def push_point_callback(self, msg: Float32MultiArray):
        self.allocation_pushers_xyxy = msg.data

    def wrench_desired_callback(self, msg: Float32MultiArray):
        self.w_d = msg.data
        
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

    def publish_object_poses(self, object_data):
        now = self.get_clock().now().to_msg()

        for obj in object_data:
            name = obj["name"]
            pos = obj["position"]
            quat = obj["quaternion"]

            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat

            if name not in self.object_publishers:
                topic = f'/object_poses/{name}'
                self.object_publishers[name] = self.create_publisher(PoseStamped, topic, 10)
                self.get_logger().info(f"Created publisher for object {name} -> {topic}")

            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "base"
            pose_msg.pose = p
            self.object_publishers[name].publish(pose_msg)

    def publish_contacts(self, pushers):
        now = self.get_clock().now().to_msg()

        for pusher in pushers:
            msg = PusherInfo()
            msg.header = Header()
            msg.header.stamp = now
            msg.header.frame_id = "base"
            msg.frame_num = pusher['frame_number']

            msg.pusher_name = pusher['pusher_name']
            if msg.pusher_name not in self.pusher_publishers:
                topic = f"/pusher_data_{msg.pusher_name}"
                self.pusher_publishers[msg.pusher_name] = self.create_publisher(PusherInfo, topic, 10)
            
            r, g, b = pusher['color']
            msg.color = ColorRGBA(r=r/255.0, g=g/255.0, b=b/255.0, a=1.0)

            msg.pusher_location = Point(
                x=float(pusher['pusher_location'][0]),
                y=float(pusher['pusher_location'][1]),
                z=float(pusher['pusher_location'][2])
            )

            msg.nearest_point = Point(
                x=float(pusher['nearest_point'][0]),
                y=float(pusher['nearest_point'][1]),
                z=float(pusher['nearest_point'][2])
            )

            msg.kappa = float(pusher['kappa'])
            msg.object_index = pusher['object_index']
            msg.local_contour_index = pusher['local_contour_index']
            self.pusher_publishers[msg.pusher_name].publish(msg)

    def publish_recommended_contacts(self, recommended):
        now = self.get_clock().now().to_msg()

        (pos_1, norm_1), (pos_2, norm_2) = recommended
        pos_1_msg = PointStamped()
        pos_1_msg.header.stamp = now
        pos_1_msg.header.frame_id = "base"
        pos_1_msg.point.x, pos_1_msg.point.y, pos_1_msg.point.z = pos_1
        self.recommended_publishers["pusher_1_position"].publish(pos_1_msg)

        pos_2_msg = PointStamped()
        pos_2_msg.header.stamp = now
        pos_2_msg.header.frame_id = "base"
        pos_2_msg.point.x, pos_2_msg.point.y, pos_2_msg.point.z = pos_2
        self.recommended_publishers["pusher_2_position"].publish(pos_2_msg)

        norm_1_msg = Vector3Stamped()
        norm_1_msg.header.stamp = now
        norm_1_msg.header.frame_id = "base"
        norm_1_msg.vector.x, norm_1_msg.vector.y, norm_1_msg.vector.z = norm_1
        self.recommended_publishers["pusher_1_normal"].publish(norm_1_msg)

        norm_2_msg = Vector3Stamped()
        norm_2_msg.header.stamp = now
        norm_2_msg.header.frame_id = "base"
        norm_2_msg.vector.x, norm_2_msg.vector.y, norm_2_msg.vector.z = norm_2
        self.recommended_publishers["pusher_2_normal"].publish(norm_2_msg)

    def publish_contour(self, contour: np.ndarray):
        contour_xyz = contour # shape: (N, 3)
        contour_xy = contour_xyz[:,:2] # clip out z column
        contour_xy = contour_xy.ravel()
        contour_xy_lis = contour_xy.tolist()
        contour_msg = Float32MultiArray()
        contour_msg.data = contour_xy_lis
        self.contour_publisher.publish(contour_msg)