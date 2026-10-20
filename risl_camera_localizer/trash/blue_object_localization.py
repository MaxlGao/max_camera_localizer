import cv2
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from risl_camera_localizer.camera_selection import detect_available_cameras, select_camera
from risl_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_pliers
from risl_camera_localizer.aruco_pose_bridge import ArucoPoseBridge
from risl_camera_localizer.color_detection_functions import detect_blue_object_positions
from risl_camera_localizer.geometric_functions import transform_points_world_to_img, complete_triangle, pick_best_candidate
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
ALLEN_KEY_LEN = [37, 102, 126]
PLIERS_LEN = [37, 70, 70]

def identify_objects_from_blobs(world_points, tolerance=5.0):
    known_triangles = {
        "allen_key": ALLEN_KEY_LEN,
        "pliers": PLIERS_LEN
    }

    identified_objects = []

    for tri_pts in combinations(world_points, 3):
        p1, p2, p3 = np.array(tri_pts[0]), np.array(tri_pts[1]), np.array(tri_pts[2])
        sides = sorted([
            1000 * np.linalg.norm(p1 - p2),
            1000 * np.linalg.norm(p2 - p3),
            1000 * np.linalg.norm(p3 - p1)
        ])

        for name, template in known_triangles.items():
            expected = sorted(template)
            diffs = [abs(a - b) for a, b in zip(sides, expected)]
            if all(d < tolerance for d in diffs):
                if name == "allen_key":
                    pos, quat, contacts = define_body_frame_allen_key(p1, p2, p3)
                elif name == "pliers":
                    pos, quat, contacts = define_body_frame_pliers(p1, p2, p3)

                identified_objects.append({
                    "name": name,
                    "points": (p1, p2, p3),
                    "position": pos,
                    "quaternion": quat,
                    'inferred': False,
                    "contacts": contacts
                })
                break  # One match per triangle

    return identified_objects

def draw_overlay(frame, cam_pos, cam_quat, object_data, frame_idx, ee_pos, ee_quat):
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

def draw_identified_triangles(frame, camera_matrix, cam_pos, cam_quat, identified_objects):
    color_map = {
        "allen_key": (0, 255, 0),   # Green
        "pliers": (0, 0, 255),      # Red
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

    return frame

def attempt_recovery_for_missing_objects(last_objects, current_points, known_triangles, merge_threshold=0.02):
    recovered = []

    for prev in last_objects:
        name = prev["name"]
        prev_pts = prev["points"]
        matched_pts = []
        unmatched_prev_pt = None

        # Find current points close to previous ones
        for prev_pt in prev_pts:
            found = False
            for cur_pt in current_points:
                if np.linalg.norm(prev_pt - cur_pt) < merge_threshold:
                    matched_pts.append((prev_pt, cur_pt))
                    found = True
                    break
            if not found:
                unmatched_prev_pt = prev_pt
        if len(matched_pts) < 2:
            continue  # not enough info to infer

        # If more than two points matched, pick best two (closest to original positions)
        if len(matched_pts) > 2:
            matched_pts.sort(key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
            unmatched_prev_pt = matched_pts[2][0]
            matched_pts = matched_pts[:2]

        if len(matched_pts) == 2:
            cur_pts = [pair[1] for pair in matched_pts]

            side_lengths = known_triangles[name]
            candidates = complete_triangle(cur_pts[0], cur_pts[1], side_lengths)
            if candidates:
                inferred_p3 = pick_best_candidate(candidates, unmatched_prev_pt)
            else:
                inferred_p3 = None
            print("INFERRED", inferred_p3)

            # for inferred_p3 in candidates:
            try:
                if name == "allen_key":
                    pos, quat, contacts = define_body_frame_allen_key(cur_pts[0], cur_pts[1], inferred_p3)
                elif name == "pliers":
                    pos, quat, contacts = define_body_frame_pliers(cur_pts[0], cur_pts[1], inferred_p3)
                recovered.append({
                    "name": name,
                    "points": (cur_pts[0], cur_pts[1], inferred_p3),
                    "position": pos,
                    "quaternion": quat,
                    "inferred": True,
                    "contacts": contacts
                })
            except:
                continue
    return recovered

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
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        world_points, image_points = detect_blue_object_positions(frame, CAMERA_MATRIX, cam_pos, cam_quat)
        identified_objects = identify_objects_from_blobs(world_points)
        missing = False
        for det in detected_objects:
            if not any(obj["name"] == det["name"] for obj in identified_objects):
                missing = True
        
        if missing:
            # Attempt recovery if any objects are missing
            recovered_objects = attempt_recovery_for_missing_objects(
                detected_objects,
                world_points,
                known_triangles={
                    "allen_key": ALLEN_KEY_LEN,
                    "pliers": PLIERS_LEN
                }
            )
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

        draw_overlay(frame, cam_pos, cam_quat, identified_objects, frame_idx, ee_pos, ee_quat)
        draw_identified_triangles(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects)

        cv2.imshow("Blue Object Detection", frame)
        if talk:
            print(frame_idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
