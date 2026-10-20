import cv2
import numpy as np
from risl_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec
from scipy.spatial.distance import cdist

class QuaternionKalman:
    def __init__(self, dt=0.067):
        # 10 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz]
        self.kf = cv2.KalmanFilter(10, 7)
        
        # dt should match your actual camera framerate. Default 0.067s = 15Hz
        # If you have 30 FPS, use dt=0.033; for 60 FPS use dt=0.0167
        self.dt = dt

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
            self.kf.processNoiseCov[i, i] = 1e-5
        for i in range(3, 7):  # quaternion x, y, z, w
            self.kf.processNoiseCov[i, i] = 1e-1
        for i in range(7, 10):  # vx, vy, vz
            self.kf.processNoiseCov[i, i] = 1e-1
        

        # Measurement noise covariance (R). Lower Values = More Trust = You have good cameras
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        for i in range(3):   # position
            self.kf.measurementNoiseCov[i, i] = 1e0
        for i in range(3, 7):  # quaternion
            self.kf.measurementNoiseCov[i, i] = 1e0
        
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((10, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion
        
        # Drift detection: track max velocity to detect filter divergence
        self.max_vel_threshold = 0.25  # m/s - warn if filter predicts motion faster than this
        self.last_measurement = None

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
        
        # DRIFT DETECTION: Clamp velocity if it grows too large
        # This prevents the Kalman filter from running away with small velocity biases
        velocity = self.kf.statePost[7:10].flatten()
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.max_vel_threshold:
            # Velocity is unrealistic for a slowly-moving object - dampen it
            damping = 0.5  # Scale down by 50%
            self.kf.statePost[7:10] *= damping
            if vel_norm > self.max_vel_threshold * 2:
                # If WAY too fast, reset velocity to zero (filter is diverging)
                self.kf.statePost[7:10] = 0
                
        return pred_tvec, pred_rvec
    

class BlobKalman:
    def __init__(self, dt=1.0):
        # State: [x, y, z, vx, vy, vz]
        self.kf = cv2.KalmanFilter(6, 3)
        
        # Transition matrix (A)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt  # x += vx*dt, etc.

        # Measurement matrix (H)
        self.kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1

        # Process noise (Q): controls filter smoothness
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
        self.kf.processNoiseCov[3:, 3:] *= 10  # more uncertainty in velocity

        # Measurement noise (R): trust in measurement
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-3

        # Initial error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Start at zero
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

        self.age = 0          # number of frames since creation
        self.time_since_update = 0

    def predict(self):
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return prediction[:3].flatten()

    def correct(self, pos):
        """Input is 3D position in world coordinates"""
        measurement = np.array(pos, dtype=np.float32).reshape(3, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0

    def get_position(self):
        return self.kf.statePost[:3].flatten()


class BlobTrackerManager:
    def __init__(self, dist_thresh=0.03, max_missed=5):
        self.trackers = []
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def update(self, detections):
        updated_trackers = []

        if len(self.trackers) == 0:
            # Initialize new trackers
            for det in detections:
                tracker = BlobKalman()
                tracker.correct(det)
                updated_trackers.append(tracker)
        else:
            predicted = np.array([trk.predict() for trk in self.trackers])
            dists = cdist(predicted, detections)

            assigned_dets = set()
            for i, trk in enumerate(self.trackers):
                min_j = np.argmin(dists[i])
                if dists[i, min_j] < self.dist_thresh and min_j not in assigned_dets:
                    trk.correct(detections[min_j])
                    assigned_dets.add(min_j)
                    updated_trackers.append(trk)
                else:
                    updated_trackers.append(trk)

            # Add unassigned detections as new trackers
            for j, det in enumerate(detections):
                if j not in assigned_dets:
                    trk = BlobKalman()
                    trk.correct(det)
                    updated_trackers.append(trk)

        # Remove stale trackers
        self.trackers = [trk for trk in updated_trackers if trk.time_since_update < self.max_missed]
        return self.trackers