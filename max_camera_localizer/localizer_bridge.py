# localizer_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point, TwistStamped
from std_msgs.msg import Header, ColorRGBA, Int32, Float32MultiArray, String
from max_camera_msgs.msg import PusherInfo
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

# It's the freakin Bifrost at this point

class LocalizerBridge(Node):
    def __init__(self):
        super().__init__('localizer_bridge')
        
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.038, -0.01])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        # --- Latest EE Pose (using values here if no ROS input - Home position) ---
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.lock = threading.Lock() # what does this even do

        self.frame_idx = 0

        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")
        self.allocation_contacts_sub = self.create_subscription(
            Float32MultiArray, "/allocation/push_points", self.push_point_callback, 10)
        self.allocation_pushers_xyxy = []
        self.allocation_pushers_xyxy_buffer = []

        self.allocation_desired = self.create_subscription(
            Float32MultiArray, "/allocation/wrench_desired", self.wrench_desired_callback, 10)
        self.w_d = None
        self.w_d_buffer = None
        self.v_g_body = None
        self.v_g_body_buffer = None

        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.object_publishers = { "wrench": self.create_publisher(PoseStamped, '/object_poses/wrench', 10),
                                  "jenga_6": self.create_publisher(PoseStamped, '/object_poses/jenga_6', 10),
                                  "allen_key": self.create_publisher(PoseStamped, '/object_poses/allen_key', 10)}
        self.pusher_publishers = {}
        self.frame_num_publsher = self.create_publisher(Int32, '/camera_frame_number', 10)
        self.recommended_publishers = {"pusher_1_position": self.create_publisher(PointStamped, '/recommended_pusher_1/position', 10), 
                                       "pusher_2_position": self.create_publisher(PointStamped, '/recommended_pusher_2/position', 10),
                                       "pusher_1_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_1/normal', 10), 
                                       "pusher_2_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_2/normal', 10)}
        
        # Careful! Boundary points are of a particular object!
        self.contour_publisher_world = self.create_publisher(Float32MultiArray, '/vision/boundary_points_world', 10)
        self.contour_publisher_body = self.create_publisher(Float32MultiArray, '/vision/boundary_points_body', 10)
        self.twist_desired_body = self.create_subscription(TwistStamped, '/allocation/gripper_twist', self.twist_desired_body_callback, 10)
        self.twist_desired_world = self.create_publisher(TwistStamped, 'recommended_gripper_twist_world', 10)

        self.buffer_advance_listener = self.create_subscription(String, 'buffer_advance_channel', self.buffer_advance, 10)
        self.buffer_advance_confirmer = self.create_publisher(String, 'buffer_advance_channel', 10)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def push_point_callback(self, msg: Float32MultiArray):
        """These will be in the body frame of some object. Can't embed that info in Float32."""

        # MAKE SURE OverController IS READY FOR THIS TO HAPPEN
        self.allocation_pushers_xyxy_buffer = msg.data
        if self.allocation_pushers_xyxy is None:
            # First time buffer advance is ok
            self.allocation_pushers_xyxy = msg.data

    def wrench_desired_callback(self, msg: Float32MultiArray):
        """This will be in world frame"""
        self.w_d_buffer = msg.data
        if self.w_d is None:
            # First time buffer advance is ok
            self.w_d = msg.data

    def twist_desired_body_callback(self, msg: TwistStamped):
        """This will be in body frame"""
        self.v_g_body_buffer = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]
        if self.v_g_body is None:
            # First time buffer advance is ok
            self.v_g_body = self.v_g_body_buffer

    def buffer_advance(self, msg: String):
        if "Request" in msg.data:
            self.allocation_pushers_xyxy = self.allocation_pushers_xyxy_buffer
            self.w_d = self.w_d_buffer
            self.v_g_body = self.v_g_body_buffer
            # Wait until merged_localization finishes with the pusher stuff?
            # Otherwise we're sending the wrong message to OverController.
            # Guarenteed would be waiting for frame_idx to increment by 2.
            current_idx = self.frame_idx
            while self.frame_idx - current_idx < 2:
                rclpy.spin_once(self, timeout_sec=0.01)
            msg = String()
            time = self.get_clock().now().to_msg()
            msg.data = f"Buffer Advance Confirmed at t = {time}. New w_d is {self.w_d}, pushers at {self.allocation_pushers_xyxy}"
            self.buffer_advance_confirmer.publish(msg)


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

    def publish_recommended_twist(self, twist_world):
        now = self.get_clock().now().to_msg()
        twist_msg = TwistStamped()
        twist_msg.header.stamp = now
        twist_msg.header.frame_id = "base"

        twist_msg.twist.linear.x = twist_world[0]
        twist_msg.twist.linear.y = twist_world[1]
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = twist_world[2]
        self.twist_desired_world.publish(twist_msg)

    def publish_contour(self, contour: np.ndarray, obj):
        # Given 'contour' is np.array of shape (N, 3), rows in m
        # Said contour may be a world frame representation. A back transform is needed.
        # Why can't we just send the contour as plainly defined in process_stl? 
        # We'd still need to extract name information
        contour_xyz = contour
        contour_xy = contour_xyz[:,:2] # clip out z column

        # Keeping in world frame...
        contour_xy_flat = contour_xy.ravel()
        contour_xy_lis = contour_xy_flat.tolist() # Acquire list [x1, y1, x2, y2, ...]
        contour_msg = Float32MultiArray()
        contour_msg.data = contour_xy_lis
        self.contour_publisher_world.publish(contour_msg)

        # Transform to body frame
        def rot2(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s], [s, c]])

        pos = obj["position"]
        quat = obj["quaternion"]
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)
        com = np.array([pos[0], pos[1]])
        ori = euler[2]
        contour_xy_translated = contour_xy - com
        contour_xy_rotated = (rot2(-ori) @ contour_xy_translated.T).T
        contour_xy_flat = contour_xy_rotated.ravel()
        contour_xy_lis = contour_xy_flat.tolist() # Acquire list [x1, y1, x2, y2, ...]
        contour_msg.data = contour_xy_lis
        self.contour_publisher_body.publish(contour_msg)
