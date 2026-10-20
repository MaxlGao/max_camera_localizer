import cv2
import cv2.aruco as aruco
import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R


# Define calibration data (replace with real values)
CAMERA_MATRIX = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # Replace if calibrated
MARKER_SIZE = 0.05  # meters
HALF_SIZE = MARKER_SIZE / 2
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_250": aruco.DICT_5X5_250
}
trackers = {}

# Memory to store recent detections
marker_stability = {
    "last_tvec": None,
    "last_frame": -1,
    "confirmed": False,
    "hold_counter": 0
}

class QuaternionKalman:
    def __init__(self):
        # 10 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz]
        self.kf = cv2.KalmanFilter(10, 7)

        dt = 1

        # A: Transition matrix (10x10)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(3):  # x += vx*dt
            self.kf.transitionMatrix[i, i+7] = dt

        # H: Measurement matrix (7x10)
        self.kf.measurementMatrix = np.zeros((7, 10), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)

        # Process and measurement noise
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-6
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((10, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion

    def correct(self, tvec, rvec):
        quat = rvec_to_quat(rvec)
        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)

    def predict(self):
        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        # Normalize quaternion to prevent drift
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec

def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]"""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector"""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec

def slerp_quat(q1, q2, blend=0.5):
    """Spherical linear interpolation between two quaternions"""
    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)
    rots = R.concatenate([rot1, rot2])
    # rots = R.random(2, random_state=2342345)
    slerp = scipy.spatial.transform.Slerp([0, 1], rots)
    return slerp(blend).as_quat()

def detect_available_cameras(max_cams=15):
    """Try to open camera IDs and return a list of working ones."""
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def select_camera(available_ids):
    """Let user preview and select from available cameras."""
    print("Available camera IDs:", available_ids)
    for cam_id in available_ids:
        cap = cv2.VideoCapture(cam_id)
        print(f"Showing preview for camera ID {cam_id} (press any key to continue, or ESC to select this one)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera ID {cam_id}", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return cam_id
            elif key != -1:
                break
        cap.release()
        cv2.destroyAllWindows()
    return available_ids[0] if available_ids else None

def detect_markers(frame, gray, aruco_dicts, parameters):
    all_corners, all_ids = [], []
    for dict_id in aruco_dicts.values():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        print("corners: ", corners)
        print("ids: ", ids)
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size, kalman: QuaternionKalman, current_frame, last_seen_frame):
    marker_found = False
    max_movement = 0.05  # meters
    hold_required = 2    # frames it must persist

    if corners and ids:
        # Assume we're only using the first marker we find
        corner = corners[0]
        image_points = corner[0].reshape(-1, 2)

        object_points = np.array([
            [-HALF_SIZE,  HALF_SIZE, 0],
            [ HALF_SIZE,  HALF_SIZE, 0],
            [ HALF_SIZE, -HALF_SIZE, 0],
            [-HALF_SIZE, -HALF_SIZE, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if success:
            movement_ok = True
            tvec_flat = tvec.flatten()

            if marker_stability["last_tvec"] is not None:
                distance = np.linalg.norm(tvec_flat - marker_stability["last_tvec"])
                movement_ok = distance < max_movement

            if movement_ok:
                marker_stability["hold_counter"] += 1
            else:
                marker_stability["hold_counter"] = 0

            # Save last tvec + frame
            marker_stability["last_tvec"] = tvec_flat
            marker_stability["last_frame"] = current_frame

            if marker_stability["hold_counter"] >= hold_required:
                marker_stability["confirmed"] = True

                # Convert both to quaternions
                measured_quat = rvec_to_quat(rvec)
                pred_tvec, pred_rvec = kalman.predict()
                pred_quat = rvec_to_quat(pred_rvec)

                # SLERP between predicted and measured quaternion
                blend_factor = 0.8 # 0.0 for all prediciton, 1.0 for all measurement
                blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                blended_rvec = quat_to_rvec(blended_quat)

                # Lin interpolation between predicted and measured displacement
                blended_tvec = blend_factor * tvec.flatten() + (1-blend_factor) * pred_tvec

                # Use blended version to correct and draw
                kalman.correct(blended_tvec, blended_rvec)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, blended_rvec, blended_tvec, marker_size * 0.5)
                print(f"[Confirmed] t: {tvec.flatten()} | r: {rvec.flatten()}")
                marker_found = True
                last_seen_frame[0] = current_frame
            else:
                print(f"[Holding] t: {tvec.flatten()} | r: {rvec.flatten()} | Hold frames: {marker_stability['hold_counter']}")


    # Use full prediction when no measurement is available
    pred_tvec, pred_rvec = kalman.predict()

    if not marker_found and marker_stability["confirmed"] and current_frame - last_seen_frame[0] < 15:
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                        pred_rvec.reshape(3, 1), pred_tvec.reshape(3, 1), marker_size * 0.5)
        print(f"[Ghost] t: {pred_tvec} | r: {pred_rvec}")


def main():
    kalman_filter = QuaternionKalman()
    frame_idx = 0
    last_seen_frame = [0]
    available = detect_available_cameras()
    if not available:
        print("No available cameras found.")
        return

    cam_id = select_camera(available)
    if cam_id is None:
        print("No camera selected.")
        return

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Failed to open camera ID {cam_id}")
        return

    parameters = aruco.DetectorParameters()
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)

        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filter, frame_idx, last_seen_frame)

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
