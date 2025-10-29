import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from max_camera_localizer.camera_selection import detect_available_cameras, select_camera
from max_camera_localizer.localizer_bridge import LocalizerBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from max_camera_localizer.detection_functions import detect_markers, detect_color_blobs, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
from max_camera_localizer.object_frame_definitions import define_jenga_contour, hard_define_contour
from max_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_wrench
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
MARKER_SIZE = 0.019  # meters
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
green_range = [np.array([35, 80, 100]), np.array([75, 255, 255])]
yellow_range = [np.array([15, 80, 60]), np.array([35, 255, 255])]

pusher_distance_max = 0.030

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
    parser.add_argument("--no-pushers", action='store_true',
                        help="Stops detecting yellow and green pushers")
    parser.add_argument("--recommend-push", type=str, default="none",
                        help="none = Don't recommend; lfd = Learning from Demonstration, pso = allocation-based")
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
    frame_idx = 0

    if args.camera_id is not None:
        cam_id = args.camera_id
    else:        
        available = detect_available_cameras()
        if not available:
            return
        cam_id = select_camera(available)
        if cam_id is None:
            return

    talk = not args.suppress_prints
    if args.recommend_push == "lfd":
        from max_camera_localizer.data_predict import predict_pusher_outputs

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    print("Press 'q' to quit.")

    detected_objects = []
    last_pushers = {"green": None, "yellow": None}
    unconfirmed_blobs = {"green": None, "yellow": None}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        identified_jenga = []
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()
        push_point_xyxy = bridge_node.allocation_pushers_xyxy

        # Aruco Section
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk)

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

        # Pusher section
        pushers = {"green": None, "yellow": None}
        nearest_pushers = []
        if not args.no_pushers:
            world_points_green, _ = detect_color_blobs(frame, green_range, (0, 255, 0), CAMERA_MATRIX, cam_pos, cam_quat, min_area=150, merge_threshold=0)
            world_points_yellow, _ = detect_color_blobs(frame, yellow_range, (0, 255, 255), CAMERA_MATRIX, cam_pos, cam_quat, min_area=150, merge_threshold=0)

            if world_points_green:
                best_green = pick_closest_blob(world_points_green, last_pushers["green"])
                pushers["green"] = (best_green, (0, 255, 0))
                last_pushers["green"] = best_green

            if world_points_yellow:
                best_yellow = pick_closest_blob(world_points_yellow, last_pushers["yellow"])
                pushers["yellow"] = (best_yellow, (0, 255, 255))
                last_pushers["yellow"] = best_yellow
            

            # Working block for pusher-object interaction detection
            # For now, gets nearest contour point to each pusher
            all_xyz = []
            all_kappa = []
            all_meta = []  # to keep track of which object and index a point came from
            if objects:  # At least one pusher detected
                for obj_idx, obj in enumerate(objects):
                    xyz = obj['contour']['xyz']
                    kappa = obj['contour']['kappa']
                    all_xyz.extend(xyz)
                    all_kappa.extend(kappa)
                    all_meta.extend([(obj_idx, i) for i in range(len(xyz))])

                all_xyz = np.array(all_xyz)
                all_kappa = np.array(all_kappa)

                tree = cKDTree(all_xyz)

                for color, pusher in pushers.items():
                    if pusher is not None:
                        pusher_pos, col = pusher
                        distance, contour_idx = tree.query(pusher_pos)
                        if distance > pusher_distance_max: # Must be within 30mm (accounts for differences in z)
                            continue
                        nearest_point = all_xyz[contour_idx]
                        kappa_value = all_kappa[contour_idx]
                        obj_index, local_contour_index = all_meta[contour_idx]
                        nearest_pushers.append({
                            'pusher_name': color,
                            'frame_number': frame_idx,
                            'color': col,
                            'pusher_location': pusher_pos,
                            'nearest_point': nearest_point,
                            'kappa': kappa_value,
                            'object_index': obj_index,
                            'local_contour_index': local_contour_index
                        })

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
            if "jenga" in name:
                name = "jenga"
            if name in ["allen_key", "wrench", "jenga"]:
                # Bonus: For the ML test run, predict where the pushers should go
                if args.recommend_push == "lfd":
                    posex = obj["position"][0]
                    posey = obj["position"][1]
                    objquat = obj["quaternion"]
                    objeuler = R.from_quat(objquat).as_euler('xyz')
                    oriy = objeuler[2]
                    prediction = predict_pusher_outputs(name, posex, posey, oriy, TARGET_POSES[name])
                    index = prediction['predicted_index']
                    print(index)

                    # Draw predicted points (of the one or two given)
                    recommended = []
                    for ind in index:
                        label = f"pusher recommended @ contour {ind}"
                        pusher_point_world = obj['contour']['xyz'][ind]
                        pusher_point_img = transform_points_world_to_img([pusher_point_world], cam_pos, cam_quat, CAMERA_MATRIX)
                        pusher_point_normal = obj['contour']['normals'][ind]

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - h - 20 - 5), (pusher_point_img[0][0] + w - 20, pusher_point_img[0][1] - 20 + 5), (0, 0, 0), -1)
                        cv2.putText(frame, label, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.circle(frame, pusher_point_img[0], 5, color)

                        recommended.append([pusher_point_world, pusher_point_normal])
                    if len(recommended) == 1:
                        # duplicate single pusher
                        recommended.append(recommended[0])
                    
                    bridge_node.publish_recommended_contacts(recommended)
                elif args.recommend_push == "pso" and len(push_point_xyxy) == 4:
                    pp_world_xyz = [[push_point_xyxy[0], push_point_xyxy[1], 0.01],
                                    [push_point_xyxy[2], push_point_xyxy[3], 0.01]]
                    recommended = []
                    for pp_xyz in pp_world_xyz:
                        pp_img = transform_points_world_to_img([pp_xyz], cam_pos, cam_quat, CAMERA_MATRIX)
                        label = f"pusher recommended @ ({1000*pp_xyz[0]:.0f}, {1000*pp_xyz[1]:.0f})"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (pp_img[0][0] - 20,     pp_img[0][1] - h - 20 - 5), 
                                             (pp_img[0][0] + w - 20, pp_img[0][1] - 20 + 5), (0, 0, 0), -1)
                        cv2.putText(frame, label, (pp_img[0][0] - 20, pp_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.circle(frame, pp_img[0], 5, color)
                        recommended.append([pp_xyz, np.array([0.0, 0.0, 1.0])])
                    bridge_node.publish_recommended_contacts(recommended)

                    # Draw desired wrench
                    frame = draw_wrench(frame, cam_quat, bridge_node.w_d)



                # Draw target contour
                # target_contour = hard_define_contour(TARGET_POSES[name][0], TARGET_POSES[name][1], name)
                # contour_xyz = target_contour["xyz"]
                # contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, CAMERA_MATRIX)
                # contour_img = np.array(contour_img)
                # contour_img.reshape((-1, 1, 2))
                # contour_img = contour_img[::20]
                # cv2.polylines(frame,[contour_img],False,color)

        detected_objects = identified_objects.copy()
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        bridge_node.publish_contacts(nearest_pushers)
        all_objects = identified_objects+identified_jenga
        if all_objects:
            selected_object = all_objects[0]
            bridge_node.publish_contour(selected_object['contour']['xyz'])
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, nearest_pushers)

        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
