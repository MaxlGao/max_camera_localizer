import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from max_camera_localizer.camera_selection import detect_available_cameras, select_camera
from max_camera_localizer.aruco_pose_bridge import ArucoPoseBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, transform_points_world_to_img
from max_camera_localizer.detection_functions import detect_markers, detect_color_blobs, estimate_pose, identify_objects_from_blobs, attempt_recovery_for_missing_objects
from max_camera_localizer.object_frame_definitions import define_jenga_contacts
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
BLOCK_LENGTH = 0.072
BLOCK_WIDTH = 0.024
BLOCK_THICKNESS = 0.014
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_250": aruco.DICT_5X5_250
}
OBJECT_DICTS = {
    "allen_key": [37, 102, 126],
    "wrench": [37, 70, 70]
}

trackers = {}


def draw_overlay(frame, cam_pos, cam_quat, object_data, marker_data, frame_idx, ee_pos, ee_quat):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.6, color, 1)
        y += line_height

    # Frame Index
    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    # End Effector
    ee_rotvec = R.from_quat(ee_quat).as_rotvec(degrees=True)
    ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)
    put_line(f"EE Pos: x={1000*ee_pos[0]:.1f} mm, y={1000*ee_pos[1]:.1f} mm, z={1000*ee_pos[2]:.1f} mm")
    put_line(f"EE Rot: rx={ee_rotvec[0]: 5.1f} deg, ry={ee_rotvec[1]: 5.1f} deg, rz={ee_rotvec[2]: 5.1f} deg")
    put_line(f"EE Euler: r={ee_euler[0]: 5.1f} deg, p={ee_euler[1]: 5.1f} deg, y={ee_euler[2]: 5.1f} deg")

    y += 10  # small gap

    # Camera Info
    cam_euler = R.from_quat(cam_quat).as_euler('xyz', degrees=True)
    put_line(f"Camera Pos: x={1000*cam_pos[0]:.1f} mm, y={1000*cam_pos[1]:.1f} mm, z={1000*cam_pos[2]:.1f} mm", (255, 255, 0))
    put_line(f"Camera Euler: r={cam_euler[0]: 5.1f} deg, p={cam_euler[1]: 5.1f} deg, y={cam_euler[2]: 5.1f} deg", (255, 255, 0))

    y += 10  # gap before object info

    # Generic Object Info
    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        rotvec = R.from_quat(quat).as_rotvec(degrees=True)
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        put_line(f"{name} Pos: x={1000*pos[0]:.1f} mm, y={1000*pos[1]:.1f} mm, z={1000*pos[2]:.1f} mm", (0, 255, 0))
        put_line(f"{name} Euler: r={euler[0]: 5.1f} deg, p={euler[1]: 5.1f} deg, y={euler[2]: 5.1f} deg", (0, 255, 0))
        y += 5
    

    # Markers
    for marker_id, (world_pos, world_quat, contacts) in marker_data.items():
        put_line(f"Marker {marker_id} Pos: x={1000*world_pos[0]:.2f}mm, y={1000*world_pos[1]:.2f}mm, z={1000*world_pos[2]:.2f}mm", (0, 255, 0))

        world_euler = R.from_quat(world_quat).as_euler('xyz', degrees=True)
        put_line(f"Marker {marker_id} Euler: r={world_euler[0]: 5.1f}deg, p={world_euler[1]: 5.1f}deg, y={world_euler[2]: 5.1f}deg", (0, 255, 0))

def draw_identified_triangles(frame, camera_matrix, cam_pos, cam_quat, identified_objects, marker_data, nearest_pushers):
    color_map = {
        "allen_key": (0, 255, 0),   # Green
        "wrench": (0, 0, 255),      # Red
    }

    for obj in identified_objects:
        name = obj["name"]
        world_pts = obj["points"]
        color = color_map.get(name, (255, 255, 255))  # default to white
        if obj.get("inferred"):
            color = (0, 255, 255)  # Yellow for inferred

        image_pts = transform_points_world_to_img(world_pts, cam_pos, cam_quat, camera_matrix)

        if len(image_pts) == 3:
            # Draw triangle edges
            for i in range(3):
                pt1 = image_pts[i]
                pt2 = image_pts[(i + 1) % 3]
                cv2.line(frame, pt1, pt2, color, 2)

            # Draw label with background
            centroid = tuple(np.mean(image_pts, axis=0).astype(int))
            label = f"{name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (centroid[0] - 2, centroid[1] - h - 4), (centroid[0] + w + 2, centroid[1] + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw axes using the object pose
        origin = obj["position"]
        rot = R.from_quat(obj["quaternion"])
        axes_world = [
            origin,                          # origin
            origin + rot.apply([0.01, 0, 0]),  # X axis
            origin + rot.apply([0, 0.01, 0]),  # Y axis
            origin + rot.apply([0, 0, 0.01])   # Z axis
        ]

        axes_image = transform_points_world_to_img(axes_world, cam_pos, cam_quat, camera_matrix)

        o, x, y, z = axes_image
        if o and x:
            cv2.arrowedLine(frame, o, x, (0, 0, 255), 2, tipLength=0.3)  # X: Red
        if o and y:
            cv2.arrowedLine(frame, o, y, (0, 255, 0), 2, tipLength=0.3)  # Y: Green
        if o and z:
            cv2.arrowedLine(frame, o, z, (255, 0, 0), 2, tipLength=0.3)  # Z: Blue

        # Draw contact points
        contact_points = obj["contacts"]
        contact_poses = [contact[1] for contact in contact_points]
        contact_norms = [contact[2] for contact in contact_points]
        contact_axes_start = [contact_pos - 0.02*contact_norm for (contact_pos, contact_norm) in zip(contact_poses, contact_norms)]
        contact_poses_img = transform_points_world_to_img(contact_poses, cam_pos, cam_quat, camera_matrix)
        contact_axes_img = transform_points_world_to_img(contact_axes_start, cam_pos, cam_quat, camera_matrix)
        for pos, ax in zip(contact_poses_img, contact_axes_img):
            cv2.arrowedLine(frame, ax, pos, (255, 255, 255), 2, tipLength=0.3)

        # Draw low-res Contour
        contour = obj["contour"]
        contour_xyz = contour["xyz"]
        contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, camera_matrix)
        contour_img = np.array(contour_img)
        contour_img.reshape((-1, 1, 2))
        cv2.polylines(frame,[contour_img],False,color)

    for marker_id, (world_pos, world_quat, contact_points) in marker_data.items():
        # Draw contact points
        contact_poses = [contact[1] for contact in contact_points]
        contact_norms = [contact[2] for contact in contact_points]
        contact_axes_start = [contact_pos - 0.02*contact_norm for (contact_pos, contact_norm) in zip(contact_poses, contact_norms)]
        contact_poses_img = transform_points_world_to_img(contact_poses, cam_pos, cam_quat, camera_matrix)
        contact_axes_img = transform_points_world_to_img(contact_axes_start, cam_pos, cam_quat, camera_matrix)
        for pos, ax in zip(contact_poses_img, contact_axes_img):
            cv2.arrowedLine(frame, ax, pos, (255, 255, 255), 2, tipLength=0.3)

    for nearest_pusher in nearest_pushers:
        pusher_point_world = nearest_pusher['pusher_location']
        pusher_point_img = transform_points_world_to_img([pusher_point_world], cam_pos, cam_quat, camera_matrix)

        contour_point_world = nearest_pusher['nearest_point']
        contour_point_img = transform_points_world_to_img([contour_point_world], cam_pos, cam_quat, camera_matrix)
        cv2.arrowedLine(frame, pusher_point_img[0], contour_point_img[0], (0, 255, 0), 2, tipLength=0.3)
        
        contour_id = nearest_pusher['local_contour_index']
        contour_kappa = nearest_pusher['kappa']


    return frame

def start_ros_node():
    rclpy.init()
    node = ArucoPoseBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    return parser.parse_args()


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

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    print("Press 'q' to quit.")

    detected_objects = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_data = {}
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

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
                world_contacts = define_jenga_contacts(world_pos, world_rot, BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_THICKNESS)
                marker_data[marker_id] = (world_pos, world_rot, world_contacts)

        # Blue Blob Section
        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([140, 255, 255])
        lower_green = np.array([40, 80, 50])
        upper_green = np.array([80, 255, 255])
        world_points, image_points = detect_color_blobs(frame, [lower_blue, upper_blue], (255,0,0), CAMERA_MATRIX, cam_pos, cam_quat)
        identified_objects = identify_objects_from_blobs(world_points, OBJECT_DICTS)
        world_points_pushers, image_points_pushers = detect_color_blobs(frame, [lower_green, upper_green], (0,255,0), CAMERA_MATRIX, cam_pos, cam_quat)
        

        # Working block for pusher-object interaction detection
        # For now, gets nearest contour point to each pusher
        all_xyz = []
        all_kappa = []
        all_meta = []  # to keep track of which object and index a point came from
        nearest_pushers = []
        if detected_objects and world_points_pushers:

            for obj_idx, obj in enumerate(detected_objects):
                xyz = obj['contour']['xyz']
                kappa = obj['contour']['kappa']
                all_xyz.extend(xyz)
                all_kappa.extend(kappa)
                all_meta.extend([(obj_idx, i) for i in range(len(xyz))])
            all_xyz = np.array(all_xyz)
            all_kappa = np.array(all_kappa)

            tree = cKDTree(all_xyz)
            distances, indices = tree.query(world_points_pushers)
            for pusher_idx, contour_idx in enumerate(indices):
                nearest_point = all_xyz[contour_idx]
                kappa_value = all_kappa[contour_idx]
                obj_index, local_contour_index = all_meta[contour_idx]
                nearest_pushers.append({
                    'pusher_index': pusher_idx,
                    'pusher_location': world_points_pushers[pusher_idx],
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
        
        if missing: # Attempt recovery if any objects are missing
            recovered_objects = attempt_recovery_for_missing_objects(detected_objects, world_points, known_triangles=OBJECT_DICTS)
        else:
            recovered_objects = None

        # Avoid duplicating recovered ones already present
        if recovered_objects:
            for rec in recovered_objects:
                if not any(obj["name"] == rec["name"] for obj in identified_objects):
                    identified_objects.append(rec)

        detected_objects = identified_objects.copy()

        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects)
        bridge_node.publish_marker_poses(marker_data)
        draw_overlay(frame, cam_pos, cam_quat, identified_objects, marker_data, frame_idx, ee_pos, ee_quat)
        draw_identified_triangles(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects, marker_data, nearest_pushers)

        cv2.imshow("Merged Detection", frame)
        if talk:
            print(frame_idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
