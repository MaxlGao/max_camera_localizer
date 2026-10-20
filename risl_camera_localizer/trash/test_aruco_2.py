import cv2
import cv2.aruco as aruco
import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R


c_width = 1280
c_hfov = 69.4
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

c_height = 720 # pix
c_vfov = 42.5 # deg
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

CAMERA_MATRIX = np.array([[fx, 0, c_width / 2],
                          [0, fy, c_height / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)
MARKER_SIZE = 0.019  # meters
HALF_SIZE = MARKER_SIZE / 2
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_250": aruco.DICT_5X5_250
}
trackers = {}

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

        # Process noise covariance (Q). Lower Values = More Inertia
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-6
        for i in range(3):   # x, y, z
            self.kf.processNoiseCov[i, i] = 1e-4
        for i in range(3, 7):  # quaternion x, y, z, w
            self.kf.processNoiseCov[i, i] = 1e-3
        for i in range(7, 10):  # vx, vy, vz
            self.kf.processNoiseCov[i, i] = 1e-4
        

        # Measurement noise covariance (R). Lower Values = More Trust = You have good cameras
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        for i in range(3):   # position
            self.kf.measurementNoiseCov[i, i] = 1e-4
        for i in range(3, 7):  # quaternion
            self.kf.measurementNoiseCov[i, i] = 1e-4
        
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
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame):
    max_movement = 0.05  # meters
    hold_required = 2    # frames it must persist

    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
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
                [-HALF_SIZE,  HALF_SIZE, 0],
                [ HALF_SIZE,  HALF_SIZE, 0],
                [ HALF_SIZE, -HALF_SIZE, 0],
                [-HALF_SIZE, -HALF_SIZE, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
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

                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    blend_factor = 0.99
                    blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                    blended_rvec = quat_to_rvec(blended_quat)
                    blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
                    kalman.correct(blended_tvec, blended_rvec)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, blended_rvec, blended_tvec, marker_size * 0.5)
                    # kalman.correct(tvec, rvec)
                    # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
                    print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}")
                    last_seen_frames[marker_id] = current_frame
                else:
                    print(f"[{marker_id}] Holding: t={tvec_flat}, hold={stability['hold_counter']}")
    
    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        if current_frame - last_seen < 15:
            pred_tvec, pred_rvec = kalman.predict()
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              pred_rvec.reshape(3, 1), pred_tvec.reshape(3, 1), marker_size * 0.5)
            if not current_frame == last_seen:
                print(f"[{marker_id}] Ghost: t={pred_tvec}, r={pred_rvec}")




def main():
    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0
    available = detect_available_cameras()
    if not available:
        return

    cam_id = select_camera(available)
    if cam_id is None:
        return

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)

        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx)

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
