import cv2
import sys


def open_camera(cam_id, width=1280, height=720):
    """Use consistent device IDs and capture settings throughout the app."""
    # CAP_ANY can try multiple backends with different device enumerations.
    # On Linux, V4L2 makes ID N refer consistently to /dev/videoN.
    backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_id, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_available_cameras(max_cams=15, width=1280, height=720):
    """Try to open camera IDs and return a list of working ones."""
    available = []
    for i in range(max_cams):
        cap = open_camera(i, width, height)
        try:
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
        finally:
            cap.release()
    return available

def select_camera(available_ids, width=1280, height=720):
    """Let user preview and select from available cameras."""
    print("Available camera IDs:", available_ids)
    for cam_id in available_ids:
        cap = open_camera(cam_id, width, height)
        print(f"Showing preview for camera ID {cam_id} (press any key to select, or ESC to skip)...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                actual_height, actual_width = frame.shape[:2]
                label = f"CAMERA {cam_id}: {actual_width}x{actual_height} (requested {width}x{height})"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Press any key to SELECT this camera", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Press ESC to SKIP this camera", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(f"Camera ID {cam_id}", frame)
                key = cv2.waitKey(1)
                if key != -1:
                    if key & 0xFF == 27:
                        break
                    return cam_id
        finally:
            cap.release()
            cv2.destroyAllWindows()
    return None
