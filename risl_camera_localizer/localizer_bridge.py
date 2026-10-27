# localizer_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Twist, Vector3Stamped, PointStamped, Point, PolygonStamped
from std_msgs.msg import Header, ColorRGBA, Int32, Float32MultiArray, String, Bool
from risl_camera_msgs.msg import PusherInfo
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading
from risl_camera_localizer.process_stl import CONTOUR_LIBRARY
from risl_camera_localizer.object_models import DEFAULT_OBJECT_NAMES, object_kind
import time

# It's the freakin Bifrost at this point
# Three major nodes: Allocation, Vision (Localizer), and Robot (OverController)

class LocalizerBridge(Node):
    def __init__(self):
        super().__init__('localizer_bridge')
        
        self.frame_idx = 0
        stt = self.get_clock().now().to_msg()
        self.start_time = stt.sec + stt.nanosec * 1e-9
        self.clock = 0.0

        self.push_start_time = None

        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.038, -0.01])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        # --- Latest EE Pose (using values here if no ROS input - Home position) ---
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.lock = threading.Lock() # what does this even do

        self.ee_pose_sub =             self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.ee_pose_callback, 10)
        self.allocation_contacts_sub = self.create_subscription(Float32MultiArray, "/allocation/push_points", self.push_point_callback, 10)
        self.allocation_desired_sub =  self.create_subscription(Float32MultiArray, "/allocation/wrench_desired", self.wrench_desired_callback, 10)
        self.target_name_sub =         self.create_subscription(String, "/robot/target_name", self.target_name_callback, 10)
        self.pushing_status_sub =      self.create_subscription(Bool, "/robot/currently_pushing", self.pushing_status_callback, 10)
        self.allocation_pushers_xyxy = []
        self.allocation_pushers_xyxy_buffer = []
        self.w_d_body = None
        self.w_d_body_buffer = None
        self.v_g_body = None
        self.v_g_body_buffer = None
        self.target_name = None
        self.target_name_buffer = None

        # Gripper tip point visualization (from gripper_utils)
        self.gripper_tip_points = None
        self.gripper_tip_sub = self.create_subscription(PolygonStamped, '/gripper/tip_points', self.gripper_tip_callback, 10)

        # --- Object Twist Estimation ---
        self.push_active = False
        self.history_size = 10 # average over last n samples. Samples are about 0.1s.
        self.object_poses_2D_history = {} # list of [x, y, theta, time]
        self.v_b_world = {} # Object twist in world frame
        self.v_b_body = {} # Object twist in body frame
        self.v_b_body_data = {} # Object twist history over a push, in body frame
        self.v_b_body_avg = {} # Object twist average over a push, in body frame

        # --- Publishers ---
        self.cam_pose_publisher = self.create_publisher(PoseStamped, '/vision/camera_pose', 10)
        self.frame_num_publsher = self.create_publisher(Int32, '/vision/camera_frame_number', 10)
        self.object_pose_publishers = {}
        self.object_vel_world_publishers = {}
        self.object_vel_body_publishers = {}
        self.object_vel_avg_publishers = {}
        # Advertise allocation's expected topics before any camera detections.
        # Additional Jenga instances are registered when first observed.
        for name in DEFAULT_OBJECT_NAMES:
            self.register_object(name)
        
        self.pusher_publishers = {}
        self.recommended_publishers = {"pusher_1_position": self.create_publisher(PointStamped, '/vision/recommended_pusher_1/position', 10), 
                                       "pusher_2_position": self.create_publisher(PointStamped, '/vision/recommended_pusher_2/position', 10),
                                       "pusher_1_normal": self.create_publisher(Vector3Stamped, '/vision/recommended_pusher_1/normal', 10), 
                                       "pusher_2_normal": self.create_publisher(Vector3Stamped, '/vision/recommended_pusher_2/normal', 10)}
        
        # Careful! Boundary points are of a particular object!
        self.gripper_twist_desired_body_sub =  self.create_subscription(TwistStamped,  '/allocation/recommended_gripper_twist_body', self.gripper_twist_desired_body_callback, 10)
        # self.contour_publisher_world = self.create_publisher(Float32MultiArray, '/vision/boundary_points_world', 10) # currently unused.
        self.contour_publisher_body =  self.create_publisher(Float32MultiArray, '/vision/boundary_points_body', 10)
        self.contour_normals_publisher_body = self.create_publisher(Float32MultiArray, '/vision/boundary_normals_body', 10)
        self.gripper_twist_desired_world =     self.create_publisher(TwistStamped,      '/vision/recommended_gripper_twist_world', 10)

        self.buffer_advance_listener = self.create_subscription(String, 'buffer_advance_channel', self.buffer_advance, 10)
        self.buffer_advance_confirmer =   self.create_publisher(String, 'buffer_advance_channel', 10)

    def register_object(self, name):
        """Create independent tracking state and ROS publishers once per instance."""
        if name in self.object_pose_publishers:
            return
        self.object_poses_2D_history[name] = []
        self.v_b_world[name] = [0.0, 0.0, 0.0]
        self.v_b_body[name] = [0.0, 0.0, 0.0]
        self.v_b_body_data[name] = []
        self.v_b_body_avg[name] = [0.0, 0.0, 0.0]
        self.object_pose_publishers[name] = self.create_publisher(
            PoseStamped, f'/vision/object_poses/{name}', 10)
        self.object_vel_world_publishers[name] = self.create_publisher(
            TwistStamped, f'/vision/object_twists_world/{name}', 10)
        self.object_vel_body_publishers[name] = self.create_publisher(
            TwistStamped, f'/vision/object_twists_body/{name}', 10)
        self.object_vel_avg_publishers[name] = self.create_publisher(
            TwistStamped, f'/vision/object_twists_push_avg/{name}', 10)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def pushing_status_callback(self, msg: Bool):
        self.push_active = msg.data
        print(f"Push active: {self.push_active}")
        if msg.data:
            self.push_start_time = time.time()
        else:
            self.push_start_time = None

    def gripper_tip_callback(self, msg: PolygonStamped):
        """Save current gripper tip points for visualization in merged_localization."""
        # print(f"Received gripper tip points: {[ (p.x, p.y, p.z) for p in msg.polygon.points ]}")
        # Store as list of [x, y, z] in base frame.
        self.gripper_tip_points = [[p.x, p.y, p.z] for p in msg.polygon.points]

    # region Allocation callbacks
    # buffered to make sure OverController receives the right inputs
    def push_point_callback(self, msg: Float32MultiArray):
        """These will be in the body frame of some object. Can't embed that info in Float32."""
        self.allocation_pushers_xyxy_buffer = msg.data
        if self.allocation_pushers_xyxy is None:
            # First time buffer advance is ok
            self.allocation_pushers_xyxy = msg.data

    def target_name_callback(self, msg: String):
        self.target_name_buffer = msg.data
        if self.target_name is None:
            # First time buffer advance is ok
            self.target_name = self.target_name_buffer

    def wrench_desired_callback(self, msg: Float32MultiArray):
        """This will be in body frame"""
        self.w_d_body_buffer = msg.data
        if self.w_d_body is None:
            # First time buffer advance is ok
            self.w_d_body = msg.data

    def gripper_twist_desired_body_callback(self, msg: TwistStamped):
        """This will be in body frame"""
        self.v_g_body_buffer = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]
        if self.v_g_body is None:
            # First time buffer advance is ok
            self.v_g_body = self.v_g_body_buffer

    def buffer_advance(self, msg: String):
        print("Params Updated!")
        if "Request" in msg.data:
            self.allocation_pushers_xyxy = self.allocation_pushers_xyxy_buffer
            self.w_d_body = self.w_d_body_buffer
            self.v_g_body = self.v_g_body_buffer
            self.target_name = self.target_name_buffer
            # Wait until merged_localization finishes with the pusher stuff?
            # Otherwise we're sending the wrong message to OverController.
            # Guarenteed would be waiting for frame_idx to increment by 2.
            current_idx = self.frame_idx
            while self.frame_idx - current_idx < 2:
                rclpy.spin_once(self, timeout_sec=0.01)
            msg = String()
            time = self.get_clock().now().to_msg()
            msg.data = f"Buffer Advance Confirmed at t = {time}. New w_d_body is {self.w_d_body}, pushers at {self.allocation_pushers_xyxy}"
            self.buffer_advance_confirmer.publish(msg)
    # endregion

    def get_ee_pose(self):
        return self.ee_position, self.ee_quat

    def get_camera_pose(self):
        with self.lock:
            r_ee = R.from_quat(self.ee_quat)
            r_cam_offset = R.from_quat(self.cam_offset_quat)
            cam_pos_world = self.ee_position + r_ee.apply(self.cam_offset_position)
            cam_quat_world = (r_ee * r_cam_offset).as_quat()
        return cam_pos_world, cam_quat_world

    def publish_frame(self):
        frame_msg = Int32()
        frame_msg.data = self.frame_idx
        self.frame_num_publsher.publish(frame_msg)

    def publish_camera_pose(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.cam_pose_publisher.publish(msg)

    def publish_object_poses(self, object_data: list):
        now = self.get_clock().now().to_msg()

        def _now_as_sec(now):
            return now.sec + now.nanosec * 1e-9
        
        def _angle_diff(a, b):
            '''in radians'''
            d = a - b
            # wrap to -pi, pi
            return (d + np.pi) % (2*np.pi) - np.pi

        def _array_to_msgtwist_2d(twist_arr, now, frame = "base"):
            ts = TwistStamped()
            t = Twist()
            t.linear.x = twist_arr[0]
            t.linear.y = twist_arr[1]
            t.linear.z = 0.0
            t.angular.x = 0.0
            t.angular.y = 0.0
            t.angular.z = twist_arr[2]

            ts.twist = t
            ts.header.stamp = now
            ts.header.frame_id = frame
            return ts

        for obj in object_data:
            name = obj["name"]
            self.register_object(name)
            pos = obj["position"]
            quat = obj["quaternion"]
            # Save to list
            now_t = _now_as_sec(now)
            rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
            self.object_poses_2D_history[name].append([pos[0], pos[1], rpy[2], now_t])
            # Clip to last n
            self.object_poses_2D_history[name] = self.object_poses_2D_history[name][-self.history_size:]

            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat

            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "base"
            pose_msg.pose = p
            self.object_pose_publishers[name].publish(pose_msg)

            # consider throwing some of this localization-looking stuff into the main script
            hist = self.object_poses_2D_history[name]
            if len(hist) < 2:  # give zero twist but continue for other objects
                twist_world_msg = _array_to_msgtwist_2d([0.0,0.0,0.0], now, "base")
                self.object_vel_world_publishers[name].publish(twist_world_msg)

                twist_body_msg = _array_to_msgtwist_2d([0.0,0.0,0.0], now, "body")
                self.object_vel_body_publishers[name].publish(twist_body_msg)

                self.v_b_world[name] = [0.0, 0.0, 0.0]
                self.v_b_body[name] = [0.0, 0.0, 0.0]
                continue

            # Otherwise, use earliest and latest for a simple window-average smoothing
            x_latest, y_latest, yaw_latest, t_latest = hist[-1]
            x_earliest, y_earliest, yaw_earliest, t_earliest = hist[0]

            dt_total = t_latest - t_earliest
            if dt_total <= 0.0:
                # fallback: tiny dt or identical timestamps -> produce zero twist
                dt_total = 1e-6

            dx = x_latest - x_earliest
            dy = y_latest - y_earliest
            dyaw = _angle_diff(yaw_latest, yaw_earliest)

            vx_world = dx / dt_total
            vy_world = dy / dt_total
            omega = dyaw / dt_total # Rad/s

            # Construct world-frame twist
            twist_world_msg = _array_to_msgtwist_2d([vx_world, vy_world, omega], now, "base")
            self.object_vel_world_publishers[name].publish(twist_world_msg)

            # Transform linear velocity into the object's body frame
            # v_body = R_z(-yaw_latest) * v_world_linear
            cy = np.cos(yaw_latest)
            sy = np.sin(yaw_latest)
            vx_body = cy * vx_world + sy * vy_world
            vy_body = -sy * vx_world + cy * vy_world

            twist_body_msg = _array_to_msgtwist_2d([vx_body, vy_body, omega], now, "base")
            self.object_vel_body_publishers[name].publish(twist_body_msg)

            self.v_b_world[name] = [vx_world, vy_world, omega]
            self.v_b_body[name] = [vx_body, vy_body, omega]

            # Update body twist history, if robot is pushing RIGHT NOW
            if self.push_active and self.push_start_time is not None and (time.time() - self.push_start_time > 0.5):
                self.v_b_body_data[name].append(self.v_b_body[name])
                self.v_b_body_avg[name] = np.mean(self.v_b_body_data[name], axis=0)
            else:
                self.v_b_body_data[name] = []
                self.v_b_body_avg[name] = [0.0, 0.0, 0.0]
            
            twist_avg_msg = _array_to_msgtwist_2d(self.v_b_body_avg[name], now, "body")
            self.object_vel_avg_publishers[name].publish(twist_avg_msg)

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
        self.gripper_twist_desired_world.publish(twist_msg)

    def publish_contour(self, object_name):
        contour = CONTOUR_LIBRARY[object_kind(object_name)]
        contour_xyz = contour['xyz'] / 1000
        contour_xy = contour_xyz[:,:2] # clip out z column
        contour_xy_lis = contour_xy.ravel().tolist() # Acquire list [x1, y1, x2, y2, ...]
        contour_msg = Float32MultiArray()
        contour_msg.data = contour_xy_lis
        self.contour_publisher_body.publish(contour_msg)

        contour_normals_xy = contour['normals'][:, :2] # clip out z column
        contour_normals_msg = Float32MultiArray()
        contour_normals_msg.data = contour_normals_xy.ravel().tolist()
        self.contour_normals_publisher_body.publish(contour_normals_msg)

        # ye olde method
        # # Given 'contour' is np.array of shape (N, 3), rows in m
        # # Said contour may be a world frame representation. A back transform is needed.
        # # Why can't we just send the contour as plainly defined in process_stl? 
        # # We'd still need to extract name information
        # contour_xyz = contour
        # contour_xy = contour_xyz[:,:2] # clip out z column

        # # Keeping in world frame...
        # contour_xy_flat = contour_xy.ravel()
        # contour_xy_lis = contour_xy_flat.tolist() # Acquire list [x1, y1, x2, y2, ...]
        # contour_msg = Float32MultiArray()
        # contour_msg.data = contour_xy_lis
        # self.contour_publisher_world.publish(contour_msg)

        # # Transform to body frame
        # def rot2(a):
        #     c, s = np.cos(a), np.sin(a)
        #     return np.array([[c, -s], [s, c]])

        # name = obj["name"]
        # pos = obj["position"]
        # quat = obj["quaternion"]
        # euler = R.from_quat(quat).as_euler('xyz', degrees=False)
        # com = np.array([pos[0], pos[1]])
        # ori = euler[2]
        # contour_xy_translated = contour_xy - com
        # contour_xy_rotated = (rot2(-ori) @ contour_xy_translated.T).T
        # contour_xy_flat = contour_xy_rotated.ravel()
        # contour_xy_lis = contour_xy_flat.tolist() # Acquire list [x1, y1, x2, y2, ...]
        # contour_msg.data = contour_xy_lis
        # self.contour_publisher_body.publish(contour_msg)

    # unused
    def publish_contacts(self, pushers):
        # For recording and rendering human-controlled chopstick pushers
        now = self.get_clock().now().to_msg()

        for pusher in pushers:
            msg = PusherInfo()
            msg.header = Header()
            msg.header.stamp = now
            msg.header.frame_id = "base"
            msg.frame_num = pusher['frame_number']

            msg.pusher_name = pusher['pusher_name']
            # if msg.pusher_name not in self.pusher_publishers:
            #     topic = f"/pusher_data_{msg.pusher_name}"
            #     self.pusher_publishers[msg.pusher_name] = self.create_publisher(PusherInfo, topic, 10)
            
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
