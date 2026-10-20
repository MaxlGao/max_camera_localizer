import cv2

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
            cv2.putText(frame, f"PREVIEW OF CAMERA {cam_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press ESC to SELECT this camera", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press any key to SKIP this camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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