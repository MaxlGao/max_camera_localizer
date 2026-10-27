import cv2
import sys


def find_camera_id_from_text(typed_text, available_ids):
    """Return the exact camera ID if the typed text matches an available camera."""
    if typed_text is None:
        return None

    value = typed_text.strip()
    if not value:
        return None

    try:
        camera_id = int(value)
    except ValueError:
        return None

    if camera_id in available_ids:
        return camera_id
    return None


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
    if not available_ids:
        return None

    current_index = 0
    typed_camera = ""

    while current_index < len(available_ids):
        cam_id = available_ids[current_index]
        cap = open_camera(cam_id, width, height)
        jumped_to_camera = False
        print(f"Showing preview for camera ID {cam_id} (type a number to jump, press any key to select, or ESC to skip)...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                actual_height, actual_width = frame.shape[:2]
                label = f"CAMERA {cam_id}: {actual_width}x{actual_height} (requested {width}x{height})"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Type a camera number to jump instantly", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Press any key to SELECT this camera", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Press ESC to SKIP this camera", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if typed_camera:
                    cv2.putText(frame, f"Typed camera: {typed_camera}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(f"Camera ID {cam_id}", frame)
                key = cv2.waitKey(1)
                if key == -1:
                    continue

                key_code = key & 0xFF
                if key_code == 27:
                    break
                if key_code in (10, 13):
                    return cam_id
                if key_code in (8, 127):
                    typed_camera = typed_camera[:-1]
                    continue
                if 48 <= key_code <= 57:
                    typed_camera += chr(key_code)
                    jumped_camera = find_camera_id_from_text(typed_camera, available_ids)
                    if jumped_camera is not None:
                        current_index = available_ids.index(jumped_camera)
                        typed_camera = ""
                        jumped_to_camera = True
                        break
                    continue

                return cam_id
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if jumped_to_camera:
            continue

        if typed_camera:
            typed_camera = ""

        current_index += 1

    return None
