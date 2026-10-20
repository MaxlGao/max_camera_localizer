import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class TCPSubscriber(Node):

    def __init__(self):
        super().__init__('tcp_subscriber')
        
        # Static transform from EE to camera (in EE frame)
        self.cam_offset_position = np.array([0.05, 0.0, 0.1])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion, change if needed

        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("TCPSubscriber node started. Waiting for messages...")

    def listener_callback(self, msg):
        # Extract EE pose
        ee_pos = np.array([msg.pose.position.x,
                           msg.pose.position.y,
                           msg.pose.position.z])

        ee_quat = np.array([msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w])
        # Transform camera position
        r_ee = R.from_quat(ee_quat)
        cam_pos_world = ee_pos + r_ee.apply(self.cam_offset_position)

        # Transform camera orientation: world_cam = world_ee * ee_cam
        r_cam_offset = R.from_quat(self.cam_offset_quat)
        r_cam_world = r_ee * r_cam_offset
        cam_quat_world = r_cam_world.as_quat()

        # Print EE coords
        self.get_logger().info(
            f"[EE Pose] Pos: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}) "
            f"Quat: ({msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, "
            f"{msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f})"
        )
        # Print camera coords
        self.get_logger().info(
            f"[Camera Pose] Pos: ({cam_pos_world[0]:.3f}, "
            f"{cam_pos_world[1]:.3f}, {cam_pos_world[2]:.3f}) "
            f"Quat: ({cam_quat_world[0]:.3f}, {cam_quat_world[1]:.3f}, "
            f"{cam_quat_world[2]:.3f}, {cam_quat_world[3]:.3f})"
        )


def main(args=None):
    rclpy.init(args=args)
    tcpsubscriber = TCPSubscriber()
    rclpy.spin(tcpsubscriber)
    tcpsubscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()