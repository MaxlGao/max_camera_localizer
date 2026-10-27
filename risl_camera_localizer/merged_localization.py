import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from risl_camera_localizer.camera_selection import detect_available_cameras, select_camera, open_camera
from risl_camera_localizer.localizer_bridge import LocalizerBridge
from risl_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from risl_camera_localizer.detection_functions import detect_markers, detect_color_blobs, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
from risl_camera_localizer.object_frame_definitions import define_jenga_contour, hard_define_contour
from risl_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_wrench, draw_twist
import threading
import rclpy
import argparse

c_width = 1280 # pix
c_hfov = 69.4 # deg
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

c_height = 720 # pix
c_vfov = 42.5 # deg
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

CAMERA_MATRIX = np.array([[fx, 0, c_width / 2],
                          [0, fy, c_height / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.019  # meters, square side length
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
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--recommend-push", type=str, default="pso",
                        help="none = Don't recommend; pso (default) = allocation-based")
    return parser.parse_args()

def pick_closest_blob(blobs, last_position):
    if not blobs:
        return None
    if last_position is None:
        return blobs[0]
    blobs_np = np.array(blobs)
    distances = np.linalg.norm(blobs_np - last_position, axis=1)
    closest_idx = np.argmin(distances)
    return blobs[closest_idx]

def main():
    args = parse_args()
    bridge_node = start_ros_node()

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    bridge_node.frame_idx = 0

    if args.camera_id is not None:
        cam_id = args.camera_id
    else:        
        available = detect_available_cameras(width=c_width, height=c_height)
        if not available:
            return
        cam_id = select_camera(available, width=c_width, height=c_height)
        if cam_id is None:
            return

    talk = not args.suppress_prints

    cap = open_camera(cam_id, width=c_width, height=c_height)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    # Relax detection thresholds so we keep more marginal markers under blur,
    # scale changes, and uneven lighting.
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.015
    parameters.maxMarkerPerimeterRate = 6.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.02
    parameters.minDistanceToBorder = 1
    parameters.errorCorrectionRate = 0.75
    parameters.detectInvertedMarker = True
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    print("Press 'q' to quit.")

    detected_objects = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:  # Fallback if FPS query fails
            camera_fps = 15.0
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, bridge_node.frame_idx, cam_pos, cam_quat, talk, camera_fps=camera_fps)

        # After estimating pose, collect marker world positions
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"]:
                tvec, rvec = kalman_filters[marker_id].predict()
                rquat = rvec_to_quat(rvec)
                world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
                world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
                world_contour = define_jenga_contour(world_pos, world_rot)
                identified_jenga.append({
                                    "name": f"jenga_{marker_id}",
                                    "points": [world_pos],
                                    "position": world_pos,
                                    "quaternion": world_rot,
                                    'inferred': False,
                                    "contour": world_contour
                                })

        objects = identified_jenga + detected_objects

        # Blue Blob Section
        world_points, _ = detect_color_blobs(frame, blue_range, (255,0,0), CAMERA_MATRIX, cam_pos, cam_quat)
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
            # if "jenga" in name:
            #     name = "jenga" # generalize jenga (disabled right now)
            if name in ["allen_key", "wrench", "jenga_6"]:
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
                        pp_img = transform_points_world_to_img([pp_xyz], cam_pos, cam_quat, CAMERA_MATRIX)
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
        if all_objects:
            selected_object = all_objects[0] # Critically, only one object is published
            bridge_node.publish_contour(selected_object['name']) # only name info is needed; contour is in body frame
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, bridge_node.frame_idx, bridge_node.clock, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga)

        # Draw gripper tip estimates (from gripper_utils via /gripper/tip_points)
        if getattr(bridge_node, "gripper_tip_points", None):
            tip_points = bridge_node.gripper_tip_points
            tip_img_pts = transform_points_world_to_img(tip_points, cam_pos, cam_quat, CAMERA_MATRIX)
            name_list = ["L Tip", "R Tip", "L Mid", "R Mid"] # Assuming 4 points in order
            for i, pt in enumerate(tip_img_pts):
                cv2.circle(frame, pt, 6, (255, 0, 255), -1)
                cv2.putText(frame, f"{name_list[i]}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if len(tip_img_pts) >= 2:
                cv2.line(frame, tuple(tip_img_pts[0]), tuple(tip_img_pts[1]), (255, 0, 255), 2)

        bridge_node.publish_frame()

        #endregion

        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
