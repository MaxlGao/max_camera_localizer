import cv2
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
import numpy as np

def canonicalize_euler(orientation):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
        return (0.0, 180.0, (yaw % 360)-180)
    else:
        return orientation

def draw_text(frame, cam_pos, cam_quat, object_data, frame_idx, ee_pos, ee_quat):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.6, color, 2)
        y += line_height

    # Frame Index
    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    # End Effector
    ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)
    ee_euler = canonicalize_euler(ee_euler)
    put_line(f"EE xyz: ({1000*ee_pos[0]:.1f}, {1000*ee_pos[1]:.1f}, {1000*ee_pos[2]:.1f}) mm")
    put_line(f"EE rpy: ({ee_euler[0]: 5.1f}, {ee_euler[1]: 5.1f}, {ee_euler[2]: 5.1f}) deg")

    y += 10  # small gap

    # # Camera Info
    # cam_euler = R.from_quat(cam_quat).as_euler('xyz', degrees=True)
    # put_line(f"Camera Pos: x={1000*cam_pos[0]:.1f} mm, y={1000*cam_pos[1]:.1f} mm, z={1000*cam_pos[2]:.1f} mm", (255, 255, 0))
    # put_line(f"Camera Euler: r={cam_euler[0]: 5.1f} deg, p={cam_euler[1]: 5.1f} deg, y={cam_euler[2]: 5.1f} deg", (255, 255, 0))

    # y += 10  # small gap

    # Generic Object Info
    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        put_line(f"{name} xyz: ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}, {1000*pos[2]:.1f}) mm", (0, 255, 0))
        put_line(f"{name} rpy: ({euler[0]: 5.1f}, {euler[1]: 5.1f}, {euler[2]: 5.1f}) deg", (0, 255, 0))
        y += 5

def draw_object_lines(frame, camera_matrix, cam_pos, cam_quat, identified_objects, nearest_pushers):
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
        # contact_points = obj["contacts"]
        # contact_poses = [contact[1] for contact in contact_points]
        # contact_norms = [contact[2] for contact in contact_points]
        # contact_axes_start = [contact_pos - 0.02*contact_norm for (contact_pos, contact_norm) in zip(contact_poses, contact_norms)]
        # contact_poses_img = transform_points_world_to_img(contact_poses, cam_pos, cam_quat, camera_matrix)
        # contact_axes_img = transform_points_world_to_img(contact_axes_start, cam_pos, cam_quat, camera_matrix)
        # for pos, ax in zip(contact_poses_img, contact_axes_img):
        #     cv2.arrowedLine(frame, ax, pos, (255, 255, 255), 2, tipLength=0.3)

        # Draw low-res Contour
        contour = obj["contour"]
        contour_xyz = contour["xyz"]
        contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, camera_matrix)
        contour_img = np.array(contour_img)
        contour_img.reshape((-1, 1, 2))
        contour_img = contour_img[::20]
        cv2.polylines(frame,[contour_img],False,color)

        # Draw label with background
        # label = f"{name}"
        # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # offset = [-40, 40]
        # cv2.rectangle(frame, (o[0] + offset[0], o[1] - h + offset[1] - 5), 
        #                      (o[0] + w + offset[0], o[1] + offset[1] + 5), (0, 0, 0), -1)
        # cv2.putText(frame, label, (o[0] + offset[0], o[1] + offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    for nearest_pusher in nearest_pushers:
        label = f"pusher_{nearest_pusher['pusher_name']} @ contour {nearest_pusher['local_contour_index']}"
        pusher_point_world = nearest_pusher['pusher_location']
        pusher_point_img = transform_points_world_to_img([pusher_point_world], cam_pos, cam_quat, camera_matrix)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - h - 20 - 5), (pusher_point_img[0][0] + w - 20, pusher_point_img[0][1] - 20 + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nearest_pusher["color"], 2)

        contour_point_world = nearest_pusher['nearest_point']
        contour_point_img = transform_points_world_to_img([contour_point_world], cam_pos, cam_quat, camera_matrix)
        cv2.arrowedLine(frame, pusher_point_img[0], contour_point_img[0], nearest_pusher["color"], 2, tipLength=0.3)

    return frame

def draw_wrench(frame, cam_quat, wrench, scale=40, position=(0.85, 0.15)):
    """
    Draw a 2D representation of a wrench (force + torque) in the top-right corner of the image.

    Args:
        frame: OpenCV image.
        wrench: array 'fx', 'fy', 'tz'  (linear x, linear y, angular z)
        scale: scaling factor for arrow length visualization.
        position: normalized position (x_frac, y_frac) for wrench origin (in image coords).
    """
    h, w = frame.shape[:2]
    cx = int(w * position[0]) - 50
    cy = int(h * position[1])

    fx = wrench[0]
    fy = wrench[1]
    tz = wrench[2]

    # === Transform base-frame directions into camera frame ===
    # Define unit vectors in base frame
    x_dir_world = np.array([1, 0, 0])
    y_dir_world = np.array([0, 1, 0])

    # Rotation: world -> camera
    R_cam = R.from_quat(cam_quat)
    R_world_to_cam = R_cam.inv()

    # Transform into camera coordinates
    x_cam = R_world_to_cam.apply(x_dir_world)
    y_cam = R_world_to_cam.apply(y_dir_world)

    # Project into image plane: we take x,y components of the camera axes
    x_vec_2d = x_cam[:2]
    y_vec_2d = y_cam[:2]

    # Normalize for consistent display
    if np.linalg.norm(x_vec_2d) > 1e-6:
        x_vec_2d /= np.linalg.norm(x_vec_2d)
    if np.linalg.norm(y_vec_2d) > 1e-6:
        y_vec_2d /= np.linalg.norm(y_vec_2d)

    # Compute arrow endpoints
    arrow_len_x = int(scale * fx)
    arrow_len_y = int(scale * fy)

    end_fx = (int(cx + arrow_len_x * x_vec_2d[0]), int(cy + arrow_len_x * x_vec_2d[1]))
    end_fy = (int(cx + arrow_len_y * y_vec_2d[0]), int(cy + arrow_len_y * y_vec_2d[1]))

    # === Draw visualization ===
    color_fx = (0, 0, 255)   # Red
    color_fy = (0, 255, 0)   # Green
    color_tz = (255, 0, 0)   # Blue

    # Draw arrows
    cv2.arrowedLine(frame, (cx, cy), end_fx, color_fx, 2, tipLength=0.3)
    cv2.arrowedLine(frame, (cx, cy), end_fy, color_fy, 2, tipLength=0.3)

    # Draw torque (tz) as a curved arrow (screen-space, not orientation-dependent)
    radius = 20
    start_angle = 0
    end_angle = int(-np.sign(tz) * 270)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, color_tz, 2)
    if tz != 0:
        angle_rad = np.deg2rad(end_angle)
        end_x = int(cx + radius * np.cos(angle_rad))
        end_y = int(cy + radius * np.sin(angle_rad))
        cv2.arrowedLine(frame, (end_x, end_y), (end_x + scale//4, end_y), color_tz, 2, tipLength=1.0)


    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Fx={fx:.2f}", (cx + 50, cy - 40), font, 0.6, color_fx, 2)
    cv2.putText(frame, f"Fy={fy:.2f}", (cx + 50, cy     ), font, 0.6, color_fy, 2)
    cv2.putText(frame, f"Tz={tz:.2f}", (cx + 50, cy + 40), font, 0.6, color_tz, 2)

    # Title box
    cv2.rectangle(frame, (cx - 60, cy - 60), (cx + 150, cy + 60), (50, 50, 50), 2)
    cv2.putText(frame, "Desired Wrench", (cx - 50, cy - 65), font, 0.6, (200, 200, 200), 2)

    return frame