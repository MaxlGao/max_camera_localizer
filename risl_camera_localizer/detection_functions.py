import cv2
import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from risl_camera_localizer.depth_pose import marker_depth_translation
from scipy.spatial.transform import Rotation as R
from risl_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from risl_camera_localizer.kalman_functions import QuaternionKalman
from risl_camera_localizer.planar_pose import fit_planar_pose, PlanarPoseTracker
from risl_camera_localizer.geometric_functions import transform_points_world_to_img, slerp_quat, quat_to_rvec, complete_triangle, pick_best_candidate
from risl_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_wrench


def _display_vector(values, decimals=1, scale=1.):
    """Format console values without changing measurement precision."""
    return '(' + ','.join(f'{float(v) * scale:.{decimals}f}'
                          for v in np.asarray(values).ravel()) + ')'


def _display_depth(stability):
    """Keep the depth source and useful rejection metrics on the pose line."""
    info = stability.get('depth_diagnostics', {})
    fields = [f"depth={stability['depth_source']}"]
    if stability['depth_source'] == 'RGB fallback' and info.get('reason'):
        fields.append(info['reason'])
    if 'valid_fraction' in info:
        fields.append(f"valid={info['valid_fraction']:.0%}")
    depth = info.get('measured_z_m', info.get('median_depth_m'))
    if depth is not None:
        fields.append(f"z={depth * 1000:.1f}mm")
    if 'plane_rms_m' in info:
        fields.append(f"rms={info['plane_rms_m'] * 1000:.1f}mm")
    return ' '.join(fields)

def detect_markers(frame, gray, aruco_dicts, parameters, max_id=None, min_id=None):
    """Detect all marker IDs unless an explicit ID range is requested."""
    all_corners, all_ids = [], []
    for dict_id in aruco_dicts.values():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue
        ids = ids.flatten()
        mask = np.ones(ids.shape, dtype=bool)
        if max_id is not None:
            mask &= ids <= max_id
        if min_id is not None:
            mask &= ids >= min_id
        all_corners.extend([c for c, m in zip(corners, mask) if m])
        all_ids.extend(ids[mask])
        # aruco.drawDetectedMarkers(frame, corners, ids=None) # optional; ids being blue confuse blue-color localization
    return all_corners, all_ids

def detect_color_blobs(frame, color_range, color, camera_matrix, cam_pos, cam_quat, height=0.0, min_area=120, merge_threshold=0.02):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    mask = cv2.inRange(hsv, color_range[0], color_range[1])
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    world_points = []
    image_points = []

    if contours:
        for cnt in contours:
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)            
            if area < min_area:
                continue  # skip tiny blobs
            if M["m00"] > 0:
                cv2.drawContours(frame, [cnt], 0, (255, 255, 255), 1)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Step 1: Ray in camera frame
                pixel = np.array([cx, cy, 1.0])
                ray_cam = np.linalg.inv(camera_matrix) @ pixel

                # Step 2: Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)

                # Step 3: Ray-plane intersection with z = height over table
                t = (height - cam_origin_world[2]) / ray_world[2]
                point_world = cam_origin_world + t * ray_world
                world_points.append(point_world)

    # Step 4: Merge nearby points in world frame
    merged_world_points = []
    used = set()
    for i, pt in enumerate(world_points):
        if i in used:
            continue
        cluster = [pt]
        used.add(i)
        for j in range(i + 1, len(world_points)):
            if j in used:
                continue
            if np.linalg.norm(world_points[j] - pt) < merge_threshold:
                cluster.append(world_points[j])
                used.add(j)
        cluster_avg = np.mean(cluster, axis=0)
        merged_world_points.append(cluster_avg)
    image_points = transform_points_world_to_img(merged_world_points, cam_pos, cam_quat, camera_matrix)
    for (u,v) in image_points:
        cv2.circle(frame, (u, v), 5, color, -1)
    return merged_world_points, image_points

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, talk=True, camera_fps=15.0,
                  pose_mode="3d", marker_height=0.014, depth_m=None, permanence=None):
    if pose_mode not in ("planar", "planar-z", "3d"):
        raise ValueError("pose_mode must be planar, planar-z, or 3d")
    max_movement = 0.05  # meters
    hold_required = 5    # frames it must persist
    half_size = marker_size / 2
    dt = 1.0 / camera_fps  # Convert FPS to seconds per frame

    observations = {}
    detected_ids = set(np.asarray(ids).ravel().astype(int)) if ids is not None else set()

    if len(corners) and ids is not None and len(ids):
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters or marker_stabilities[marker_id].get("dropped", False):
                kalman_filters[marker_id] = (
                    PlanarPoseTracker(marker_height, cam_pos, cam_quat,
                                      free_height=pose_mode == "planar-z")
                    if pose_mode in ("planar", "planar-z") else QuaternionKalman(dt=dt))
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0
                }
                last_seen_frames[marker_id] = 0

            kalman = kalman_filters[marker_id]
            stability = marker_stabilities[marker_id]

            image_points = corner[0].reshape(-1, 2)
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            success = False
            rvec = None
            tvec = None
            if pose_mode in ("planar", "planar-z"):
                # The camera may move with the robot; keep world tracking fixed
                # while converting measurements and outputs with the latest TF.
                kalman.cam_pos = np.asarray(cam_pos, dtype=float)
                kalman.cam_quat = np.asarray(cam_quat, dtype=float)
                planar_result = fit_planar_pose(
                    image_points, object_points, camera_matrix, dist_coeffs,
                    cam_pos, cam_quat, marker_height,
                    free_height=pose_mode == "planar-z")
                if planar_result is not None:
                    rvec, tvec = planar_result
                    success = True
            else:
                generic_result = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if generic_result[0]:
                    rvecs = generic_result[1]
                    tvecs = generic_result[2]
                    reprojection_errors = generic_result[3] if len(generic_result) > 3 else [0.0] * len(rvecs)

                    def pose_score(candidate_rvec, candidate_tvec, reprojection_error):
                        candidate_rvec = np.asarray(candidate_rvec).reshape(3)
                        candidate_tvec = np.asarray(candidate_tvec).reshape(3)
                        reprojection_error = float(np.asarray(reprojection_error).reshape(-1)[0])
                        tilt_penalty = candidate_rvec[0] ** 2 + candidate_rvec[1] ** 2
                        depth_penalty = candidate_tvec[2] ** 2
                        return reprojection_error + 5.0 * tilt_penalty + 5.0 * depth_penalty

                    best_idx = min(
                        range(len(rvecs)),
                        key=lambda idx: pose_score(rvecs[idx], tvecs[idx], reprojection_errors[idx]),
                    )
                    rvec = rvecs[best_idx]
                    tvec = tvecs[best_idx]
                    success = True
                else:
                    guess_rvec = np.zeros((3, 1), dtype=np.float32)
                    guess_tvec = np.array([[0.0], [0.0], [marker_size]], dtype=np.float32)
                    if stability["last_tvec"] is not None:
                        guess_tvec[:, 0] = stability["last_tvec"]
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        rvec=guess_rvec,
                        tvec=guess_tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
            stability["depth_source"] = "fixed plane" if pose_mode == "planar" else "RGB fallback"
            stability['depth_diagnostics'] = {}
            if success and pose_mode != "planar":
                measured = marker_depth_translation(
                    depth_m, image_points, camera_matrix, tvec,
                    diagnostics=stability['depth_diagnostics'])
                if measured is not None:
                    if pose_mode == "planar-z":
                        # Keep the horizontal-block constraint, but measure the
                        # surface height instead of inferring it from tag scale.
                        world_height = transform_point_cam_to_world(measured, cam_pos, cam_quat)[2]
                        result = fit_planar_pose(
                            image_points, object_points, camera_matrix, dist_coeffs,
                            cam_pos, cam_quat, world_height)
                        if result is not None:
                            rvec, tvec = result
                            stability["depth_source"] = "RGB-D"
                        else:
                            stability['depth_diagnostics']['reason'] = 'horizontal fit failed'
                    else:
                        tvec = measured.reshape(3, 1)
                        stability["depth_source"] = "RGB-D"
            if success:
                observations[marker_id] = (
                    transform_point_cam_to_world(np.asarray(tvec).reshape(3), cam_pos, cam_quat),
                    transform_orientation_cam_to_world(rvec_to_quat(rvec), cam_quat))
                tvec_flat = tvec.flatten()
                distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
                movement_ok = distance < max_movement

                if movement_ok:
                    stability["hold_counter"] += 1
                else:
                    stability["hold_counter"] = 0

                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame

                if stability["hold_counter"] >= hold_required:
                    stability["confirmed"] = True

                    if pose_mode in ("planar", "planar-z"):
                        kalman.correct(tvec_flat, rvec)
                        blended_tvec, blended_rvec = kalman.predict()
                        blended_quat = rvec_to_quat(blended_rvec)
                    else:
                        measured_quat = rvec_to_quat(rvec)
                        pred_tvec, pred_rvec = kalman.predict()
                        pred_quat = rvec_to_quat(pred_rvec)
                        blend_factor = 0.995 # Adjust as needed (0 = trust measurement, 1 = trust prediction)
                        blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                        blended_rvec = quat_to_rvec(blended_quat)
                        blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
                        kalman.correct(blended_tvec, blended_rvec)
                    # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, blended_rvec, blended_tvec, marker_size * 0.5)
                    last_seen_frames[marker_id] = current_frame
                    # Convert to world frame
                    marker_pos_world = transform_point_cam_to_world(blended_tvec, cam_pos, cam_quat)
                    marker_quat_world = transform_orientation_cam_to_world(blended_quat, cam_quat)
                    stability['world_position'] = np.asarray(marker_pos_world).copy()
                    stability['world_quaternion'] = np.asarray(marker_quat_world).copy()
                    stability['measurement_frame'] = current_frame
                    if talk:
                        print(f"[{marker_id}] OK cam_mm={_display_vector(tvec_flat, scale=1000)} "
                              f"world_mm={_display_vector(marker_pos_world, scale=1000)} "
                              f"q_xyzw={_display_vector(marker_quat_world, decimals=3)} | "
                              f"{_display_depth(stability)}")
                elif talk and (permanence is None or not stability["confirmed"]):
                    print(f"[{marker_id}] HOLD {stability['hold_counter']}/{hold_required} "
                          f"cam_mm={_display_vector(tvec_flat, scale=1000)} | "
                          f"{_display_depth(stability)}")



    if permanence is not None:
        if frame is None:
            raise ValueError('Visibility-aware permanence requires an image for its dimensions')
        dropped = permanence.update(
            marker_stabilities, observations, detected_ids, current_frame,
            camera_matrix, dist_coeffs, frame.shape, cam_pos, cam_quat, marker_size)
        for marker_id, kalman in kalman_filters.items():
            if pose_mode in ("planar", "planar-z"):
                kalman.cam_pos = np.asarray(cam_pos, dtype=float)
                kalman.cam_quat = np.asarray(cam_quat, dtype=float)
            state = marker_stabilities[marker_id]
            if not talk:
                continue
            if marker_id in dropped:
                print(f"[{marker_id}] DROP {state['visibility']}")
            elif state['confirmed'] and last_seen_frames[marker_id] != current_frame:
                status = state['visibility']
                counter = (f" {state['visible_misses']}/{permanence.visible_miss_frames}"
                           if status == 'visible' else '')
                cover = f" by={state['occluders']}" if status == 'occluded' else ''
                chain = state.get('occlusion_chain', [])
                if len(chain) > 2:
                    cover += ' chain=' + '>'.join(map(str, chain))
                print(f"[{marker_id}] KEEP {status}{counter}{cover} "
                      f"world_mm={_display_vector(state['world_position'], scale=1000)}")
        return

    for marker_id, kalman in kalman_filters.items():
        if pose_mode in ("planar", "planar-z"):
            kalman.cam_pos = np.asarray(cam_pos, dtype=float)
            kalman.cam_quat = np.asarray(cam_quat, dtype=float)
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        frames_until_drop = 15
        if current_frame - last_seen < frames_until_drop:
            pred_tvec, pred_rvec = kalman.predict()
            # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, pred_rvec, pred_tvec, marker_size * 0.5)
            if not current_frame == last_seen:
                # Convert to world frame
                pred_quat = rvec_to_quat(pred_rvec)
                marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                if talk:
                    print(f"[{marker_id}] GHOST {current_frame - last_seen}/{frames_until_drop} "
                          f"cam_mm={_display_vector(pred_tvec, scale=1000)} "
                          f"world_mm={_display_vector(marker_pos_world, scale=1000)} "
                          f"q_xyzw={_display_vector(marker_quat_world, decimals=3)}")
        else:
            stability["confirmed"] = False

def detect_object(p1, p2, p3, name, inferred):
    if name == "allen_key":
        pos, quat, contacts, contour = define_body_frame_allen_key(p1, p2, p3)
    elif name == "wrench":
        pos, quat, contacts, contour = define_body_frame_wrench(p1, p2, p3)
    obj = {
        "name": name,
        "points": (p1, p2, p3),
        "position": pos,
        "quaternion": quat,
        'inferred': inferred,
        "contacts": contacts,
        "contour": contour
    }
    return obj

def identify_objects_from_blobs(world_points, object_dicts, tolerance=10.0):
    identified_objects = []

    for tri_pts in combinations(world_points, 3):
        p1, p2, p3 = np.array(tri_pts[0]), np.array(tri_pts[1]), np.array(tri_pts[2])
        sides = sorted([
            1000 * np.linalg.norm(p1 - p2),
            1000 * np.linalg.norm(p2 - p3),
            1000 * np.linalg.norm(p3 - p1)
        ])

        for name, template in object_dicts.items():
            expected = sorted(template)
            diffs = [abs(a - b) for a, b in zip(sides, expected)]
            if all(d < tolerance for d in diffs):
                identified_objects.append(detect_object(p1, p2, p3, name, False))
                break  # One match per triangle

    return identified_objects

def attempt_recovery_for_missing_objects(last_objects, current_points, known_triangles, merge_threshold=0.03):
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
            # print("INFERRED", inferred_p3)

            # for inferred_p3 in candidates:
            try:
                recovered.append(detect_object(cur_pts[0], cur_pts[1], inferred_p3, name, True))
            except:
                continue
    return recovered
