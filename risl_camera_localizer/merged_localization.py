import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from risl_camera_localizer.camera_selection import detect_available_cameras, select_camera, open_camera
from risl_camera_localizer.localizer_bridge import LocalizerBridge
from risl_camera_localizer.object_models import DEFAULT_OBJECT_NAMES
from risl_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from risl_camera_localizer.detection_functions import detect_markers, detect_color_blobs, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
from risl_camera_localizer.object_frame_definitions import define_jenga_contour, hard_define_contour
from risl_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_wrench, draw_twist
from risl_camera_localizer.realsense_camera import RealSenseCamera
from risl_camera_localizer.object_permanence import ObjectPermanence
import threading
import rclpy
from rclpy.utilities import remove_ros_args
import argparse

c_width = 1280 # pix
c_hfov = 69.4 # deg
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))

c_height = 720 # pix
c_vfov = 42.5 # deg
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))

CAMERA_MATRIX = np.array([[fx, 0, c_width / 2],
                          [0, fy, c_height / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.0185  # meters, square side length
BLOCK_LENGTH = 0.072 # meters
BLOCK_WIDTH = 0.024 # meters
BLOCK_THICKNESS = 0.014 # meters
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_250": aruco.DICT_5X5_250
}
OBJECT_DICTS = { # mm
    "allen_key": [38.8, 102.6, 129.5],
    "wrench": [37, 70, 70]
}
TARGET_POSES = {
    # position mm and orientation degrees
    "jenga": ([40, -600, 10], [0, 0, 0]),
    "wrench": ([40, -600, 10], [0, 0, 0]),
    "allen_key": ([40, -600, 10], [0, 0, 0]),
}

blue_range = [np.array([100, 80, 80]), np.array([140, 255, 255])]

trackers = {}

def start_ros_node():
    rclpy.init()
    node = LocalizerBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID.")
    parser.add_argument("--camera-backend", choices=("realsense", "opencv"), default="realsense",
                        help="RealSense synchronized RGB-D (default), or legacy RGB-only capture.")
    parser.add_argument("--realsense-serial", default=None,
                        help="Select a RealSense by stable device serial number.")
    parser.add_argument("--camera-fps", type=int, default=30, choices=(6, 15, 30),
                        help="RealSense color/depth FPS (default: 30).")
    parser.add_argument("--depth-resolution", default="424x240",
                        choices=("424x240", "640x360", "848x480", "1280x720"),
                        help="RealSense depth size (default: 242x240); RGB stays 1280x720. "
                             "Press d to cycle supported presets while running.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--blue-blobs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable blue-blob localization (default); --no-blue-blobs disables it. "
                             "Press b in the preview to toggle while running.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--recommend-push", type=str, default="pso",
                        help="none = Don't recommend; pso (default) = allocation-based")
    parser.add_argument("--pose-mode", choices=("planar", "planar-z", "3d"), default="planar-z",
                        help="planar: X/Y/yaw; planar-z: X/Y/Z/yaw; 3d: original IPPE tracker.")
    parser.add_argument("--marker-height", type=float, default=BLOCK_THICKNESS,
                        help="Marker surface world Z in meters for planar mode (default: 0.014).")
    parser.add_argument("--object-permanence", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep hidden/out-of-frame ArUco tracks; expire missing clear-view tracks.")
    parser.add_argument("--visible-miss-frames", type=int, default=5,
                        help="Consecutive clear-view misses before dropping an ArUco track (default: 5).")
    parser.add_argument("--protected-miss-frames", type=int, default=0,
                        help="Consecutive unobservable misses before dropping; 0 keeps indefinitely.")
    # ROS launch appends its own arguments; leave those for rclpy.init().
    args = parser.parse_args(remove_ros_args()[1:])
    if args.visible_miss_frames < 1 or args.protected_miss_frames < 0:
        parser.error("--visible-miss-frames must be positive; --protected-miss-frames must be nonnegative")
    args.realsense_serial = args.realsense_serial or None
    if args.camera_backend == "realsense" and args.camera_id is not None:
        parser.error("--camera-id requires --camera-backend opencv; RealSense uses --realsense-serial")
    if args.camera_backend == "opencv" and args.realsense_serial is not None:
        parser.error("--realsense-serial requires --camera-backend realsense")
    return args

def pick_closest_blob(blobs, last_position):
    if not blobs:
        return None
    if last_position is None:
        return blobs[0]
    blobs_np = np.array(blobs)
    distances = np.linalg.norm(blobs_np - last_position, axis=1)
    closest_idx = np.argmin(distances)
    return blobs[closest_idx]

def select_allocation_object(objects, target_name):
    """Keep allocation's single contour tied to a supported target."""
    return next((obj for obj in objects
                 if obj['name'] in DEFAULT_OBJECT_NAMES
                 and (target_name is None or obj['name'] == target_name)), None)

def main():
    args = parse_args()

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}

    if args.camera_backend == "realsense":
        cap = RealSenseCamera(serial=args.realsense_serial, width=c_width,
                              height=c_height, fps=args.camera_fps,
                              depth_resolution=tuple(map(int, args.depth_resolution.split("x"))))
        camera_matrix, dist_coeffs = cap.camera_matrix, cap.dist_coeffs
    else:
        cam_id = args.camera_id
        if cam_id is None:
            available = detect_available_cameras(width=c_width, height=c_height)
            if not available:
                return
            cam_id = select_camera(available, width=c_width, height=c_height)
            if cam_id is None:
                return
        cap = open_camera(cam_id, width=c_width, height=c_height)
        if not cap.isOpened():
            cap.release()
            return
        camera_matrix, dist_coeffs = CAMERA_MATRIX, DIST_COEFFS
        print(f'OpenCV approximate intrinsics: fx={fx:.1f}, fy={fy:.1f} px')
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)

    talk = not args.suppress_prints
    bridge_node = None
    try:
        bridge_node = start_ros_node()
        run_localization(args, cap, camera_matrix, dist_coeffs, bridge_node, talk,
                         kalman_filters, marker_stabilities, last_seen_frames)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if bridge_node is not None:
            bridge_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_localization(args, cap, camera_matrix, dist_coeffs, bridge_node, talk,
                     kalman_filters, marker_stabilities, last_seen_frames):
    bridge_node.frame_idx = 0
    parameters = aruco.DetectorParameters()
    # Relax detection thresholds so we keep more marginal markers under blur,
    # scale changes, and uneven lighting.
    parameters.adaptiveThreshWinSizeMin = 13
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.05
    parameters.maxMarkerPerimeterRate = 6.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.02
    parameters.minDistanceToBorder = 1
    parameters.errorCorrectionRate = 0.75
    parameters.detectInvertedMarker = True
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    blue_blobs_enabled = args.blue_blobs
    print("Press 'q' to quit; 'b' to toggle blue blobs; 'd' to cycle RealSense depth resolution.")

    permanence = (ObjectPermanence(args.visible_miss_frames, args.protected_miss_frames,
                                  block_size=(BLOCK_LENGTH, BLOCK_WIDTH, BLOCK_THICKNESS))
                  if args.object_permanence else None)
    detected_objects = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.camera_backend == "realsense":
            camera_matrix, dist_coeffs = cap.camera_matrix, cap.dist_coeffs
        bridge_node.frame_idx += 1
        now = bridge_node.get_clock().now().to_msg()
        now_sec = now.sec + now.nanosec * 1e-9
        bridge_node.clock = now_sec - bridge_node.start_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #region Object Detection

        identified_jenga = []
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()
        push_point_xyxy = bridge_node.allocation_pushers_xyxy # Might be outdated?
        gripper_twist_body = bridge_node.v_g_body

        # Aruco Section
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters, max_id=50)
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:  # Fallback if FPS query fails
            camera_fps = 15.0
        estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, bridge_node.frame_idx, cam_pos, cam_quat, talk, camera_fps=camera_fps,
                    pose_mode=args.pose_mode, marker_height=args.marker_height,
                    depth_m=getattr(cap, "depth_m", None), permanence=permanence)

        # After estimating pose, collect marker world positions
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"]:
                state = marker_stabilities[marker_id]
                # Keep missing objects fixed in the world as the camera moves.
                # Reading a Kalman prediction here would advance hidden tracks.
                world_pos = state['world_position']
                world_rot = state['world_quaternion']
                world_contour = define_jenga_contour(world_pos, world_rot)
                identified_jenga.append({
                                    "name": f"jenga_{marker_id}",
                                    "points": [world_pos],
                                    "position": world_pos,
                                    "quaternion": world_rot,
                                    'inferred': last_seen_frames[marker_id] != bridge_node.frame_idx,
                                    'visibility': state.get('visibility', 'observed'),
                                    "contour": world_contour
                                })

        identified_objects = []
        if blue_blobs_enabled:
            # Blue Blob Section
            world_points, _ = detect_color_blobs(frame, blue_range, (255,0,0), camera_matrix, cam_pos, cam_quat)
            identified_objects = identify_objects_from_blobs(world_points, OBJECT_DICTS)

            # Check for disappeared objects
            missing = False
            for det in detected_objects:
                if not any(obj["name"] == det["name"] for obj in identified_objects):
                    missing = True
        
            # Attempt recovery if any objects are missing
            if missing:
                recovered_objects = attempt_recovery_for_missing_objects(detected_objects, world_points, known_triangles=OBJECT_DICTS)
            else:
                recovered_objects = None

            # Avoid duplicating recovered ones already present
            if recovered_objects:
                for rec in recovered_objects:
                    if not any(obj["name"] == rec["name"] for obj in identified_objects):
                        identified_objects.append(rec)

        # Draw target and contour
        color = (255, 255, 0)
        for obj in identified_objects+identified_jenga:
            name = obj["name"]
            if name in DEFAULT_OBJECT_NAMES and name == bridge_node.target_name:
                if args.recommend_push == "pso" and len(push_point_xyxy) == 4:
                    pp_body_xy = np.array([[push_point_xyxy[0], push_point_xyxy[1]],
                                           [push_point_xyxy[2], push_point_xyxy[3]]])
                    # I'm ASSUMING the object in question is the one processed by PSO...
                    # Transform to world frame...
                    def rot2(a):
                        c, s = np.cos(a), np.sin(a)
                        return np.array([[c, -s], [s, c]])
                    
                    body_name = obj["name"]
                    body_pos = obj["position"]
                    body_quat = obj["quaternion"]
                    body_euler = R.from_quat(body_quat).as_euler('xyz', degrees=False)
                    com = np.array([body_pos[0], body_pos[1]])
                    body_theta = body_euler[2]
                    pp_world_xy = (rot2(body_theta) @ pp_body_xy.T).T + com
                    pp_world_xyxy = pp_world_xy.ravel().tolist()
                    pp_world_xyz = [[pp_world_xyxy[0], pp_world_xyxy[1], 0.01],
                                    [pp_world_xyxy[2], pp_world_xyxy[3], 0.01]]
                    
                    # Do transform the body twist as well
                    gripper_twist_body_lin = np.array(gripper_twist_body[0:2])
                    gripper_twist_body_rotated = (rot2(body_theta) @ gripper_twist_body_lin).tolist()
                    gripper_twist_world = [gripper_twist_body_rotated[0], gripper_twist_body_rotated[1], gripper_twist_body[2]]
                    body_twist_body = bridge_node.v_b_body[body_name]
                    body_twist_world = bridge_node.v_b_world[body_name]

                    # First Draw desired wrench and twist
                    if bridge_node.w_d_body:
                        cv2.rectangle(frame, (960, 0), [frame.shape[1], 600], (100,100,100), -1)
                        frame_h, frame_w = frame.shape[:2]
                        cx = int(frame_w * 0.85) - 50
                        cy = int(frame_h * 0.04)
                        cv2.putText(frame, "Body Frame Telemetry", (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        w_d_world = (rot2(body_theta) @ bridge_node.w_d_body[:2]).tolist()
                        w_d_world.append(bridge_node.w_d_body[2])
                        frame = draw_wrench(frame, cam_quat, w_d_world, bridge_node.w_d_body, body_theta)
                        frame = draw_twist(frame, cam_quat, gripper_twist_world, gripper_twist_body)
                        # Draw estimated twist
                        frame = draw_twist(frame, cam_quat, body_twist_world, body_twist_body, position=(0.85, 0.56), label="Body Twist (mm, deg)/s")

                    recommended = []
                    for pp_xyz in pp_world_xyz:
                        pp_img = transform_points_world_to_img([pp_xyz], cam_pos, cam_quat, camera_matrix)
                        label = f"pusher recommended @ ({1000*pp_xyz[0]:.0f}, {1000*pp_xyz[1]:.0f})"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame,      (pp_img[0][0] - 20,     pp_img[0][1] - h - 25), 
                                                  (pp_img[0][0] + w - 20, pp_img[0][1] - 15), (0, 0, 0), -1)
                        cv2.putText(frame, label, (pp_img[0][0] - 20,     pp_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.circle(frame, pp_img[0], 5, color)
                        recommended.append([pp_xyz, np.array([0.0, 0.0, 1.0])]) # norms are pos z
                    # reck = [ppxyz for [ppxyz, _] in recommended]
                    # print(f"{reck}")
                    bridge_node.publish_recommended_contacts(recommended) # In world frame!!
                    bridge_node.publish_recommended_twist(gripper_twist_world)

        detected_objects = identified_objects.copy()
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        all_objects = identified_objects+identified_jenga
        # Allocation accepts one contour; other localized tags must not replace it.
        selected_object = select_allocation_object(all_objects, bridge_node.target_name)
        if selected_object is not None:
            bridge_node.publish_contour(selected_object['name']) # only name info is needed; contour is in body frame
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, bridge_node.frame_idx, bridge_node.clock, ee_pos, ee_quat)
        draw_object_lines(frame, camera_matrix, cam_pos, cam_quat, identified_objects+identified_jenga)

        # Draw gripper tip estimates (from gripper_utils via /gripper/tip_points)
        if getattr(bridge_node, "gripper_tip_points", None):
            tip_points = bridge_node.gripper_tip_points
            tip_img_pts = transform_points_world_to_img(tip_points, cam_pos, cam_quat, camera_matrix)
            name_list = ["L Tip", "R Tip", "L Mid", "R Mid"] # Assuming 4 points in order
            for i, pt in enumerate(tip_img_pts):
                cv2.circle(frame, pt, 6, (255, 0, 255), -1)
                cv2.putText(frame, f"{name_list[i]}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if len(tip_img_pts) >= 2:
                cv2.line(frame, tuple(tip_img_pts[0]), tuple(tip_img_pts[1]), (255, 0, 255), 2)

        bridge_node.publish_frame()

        #endregion

        if args.camera_backend == "realsense":
            dw, dh = cap.depth_resolution
            cv2.putText(frame, f"Depth: {dw}x{dh} (d to switch)",
                        (max(20, frame.shape[1] - 360), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            visible = [(mid, state) for mid, state in marker_stabilities.items()
                       if state['confirmed'] or state['last_frame'] == bridge_node.frame_idx]
            for row, (mid, state) in enumerate(visible):
                if state['confirmed'] and last_seen_frames[mid] != bridge_node.frame_idx:
                    status = state.get('visibility', 'missing')
                    chain = state.get('occlusion_chain', [])
                    if len(chain) > 2:
                        status += ' via ' + '>'.join(map(str, chain))
                    cv2.putText(frame, f"Remembered {mid}: {status}",
                                (20, frame.shape[0] - 20 - row * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 255), 1)
                    continue
                info = state.get('depth_diagnostics', {})
                source = state.get('depth_source', 'RGB fallback')
                detail = info.get('reason', '') if source == 'RGB fallback' else ''
                coverage = info.get('valid_fraction')
                if coverage is not None:
                    detail += f" | valid {coverage:.0%}"
                visual_z = info.get('visual_z_m')
                if visual_z is not None:
                    detail += f" | RGB Z {visual_z:.2f}m"
                label = f"RealSense {mid}: {source} | {detail}"
                cv2.putText(frame, label, (20, frame.shape[0] - 20 - row * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        cv2.putText(frame, f"Blue blobs: {'ON' if blue_blobs_enabled else 'OFF'} (b to toggle)",
                    (max(20, frame.shape[1] - 360), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.imshow("Merged Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d') and args.camera_backend == "realsense":
            print('Switching depth profile; camera streaming will pause briefly.')
            cap.cycle_depth_resolution()
            # Reconfirm after the capture gap instead of advancing old velocity
            # estimates or recovering blobs from before the stream restart.
            kalman_filters.clear()
            marker_stabilities.clear()
            last_seen_frames.clear()
            detected_objects = []
        if key == ord('b'):
            blue_blobs_enabled = not blue_blobs_enabled
            # Clear recovery history in both directions; re-enabling must
            # start with fresh observations rather than resurrecting old blobs.
            detected_objects = []
            print(f"Blue-blob localization {'enabled' if blue_blobs_enabled else 'disabled'}.")

if __name__ == "__main__":
    main()
