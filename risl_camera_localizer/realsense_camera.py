"""Synchronized, calibrated RGB-D capture without a ROS camera-driver process."""

import warnings

import cv2
import numpy as np


def build_rectification_maps(rs, intr):
    """Map rectified destination pixels to raw pixels using SDK projection.

    RealSense distortion coefficients are model-specific. In particular, the
    inverse and modified Brown-Conrady models must not be passed to OpenCV as
    ordinary Brown-Conrady coefficients. Build once at startup, then reuse the
    same maps for color and aligned depth on every frame.
    """
    supported = (
        rs.distortion.none, rs.distortion.brown_conrady,
        rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady,
        rs.distortion.ftheta, rs.distortion.kannala_brandt4,
    )
    if intr.model not in supported:
        raise RuntimeError(f'Unsupported color distortion model: {intr.model}')
    maps = np.empty((intr.height, intr.width, 2), dtype=np.float32)
    rays_x = (np.arange(intr.width) - intr.ppx) / intr.fx
    for v in range(intr.height):
        ray_y = (v - intr.ppy) / intr.fy
        maps[v] = [rs.rs2_project_point_to_pixel(intr, [float(x), ray_y, 1.])
                   for x in rays_x]
    if not np.isfinite(maps).all():
        raise RuntimeError(f'Invalid rectification map for color distortion model: {intr.model}')
    return maps[:, :, 0].copy(), maps[:, :, 1].copy()


DEPTH_RESOLUTIONS = ((424, 240), (640, 360), (848, 480), (1280, 720))


class RealSenseCamera:
    """Own the SDK pipeline; expose rectified color and matching metric depth."""

    def __init__(self, serial=None, width=1280, height=720, fps=30, depth_resolution=None):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                'RealSense requires pyrealsense2 in the ROS Python environment. '
                'Install with python -m pip install pyrealsense2, or select '
                '--camera-backend opencv for RGB-only capture.') from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.started = False
        self.serial = serial
        self.color_resolution = (width, height)
        self.depth_resolution = depth_resolution or (width, height)
        self.fps = fps
        self.depth_m = None
        try:
            self._start(self.depth_resolution)
        except Exception:
            self.release()
            raise

    def _start(self, resolution):
        """Start one profile and refresh its calibration and alignment state."""
        rs = self.rs
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, *self.color_resolution, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, *resolution, rs.format.z16, self.fps)
        profile = self.pipeline.start(config)
        self.started = True
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]])
        self.dist_coeffs = np.zeros(5)
        self.maps = build_rectification_maps(rs, intr)
        device = profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        sensor = device.first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()
        self.fps = profile.get_stream(rs.stream.color).fps()
        # Recreate align after each profile change: it caches stream geometry.
        self.align = rs.align(rs.stream.color)
        self.depth_m = None
        self.depth_resolution = resolution
        available = set()
        for stream in sensor.get_stream_profiles():
            if (stream.stream_type() == rs.stream.depth and stream.format() == rs.format.z16
                    and stream.fps() == self.fps):
                video = stream.as_video_stream_profile()
                available.add((video.width(), video.height()))
        self.available_depth_resolutions = [r for r in DEPTH_RESOLUTIONS if r in available]
        self.usb_type = (device.get_info(rs.camera_info.usb_type_descriptor)
                         if device.supports(rs.camera_info.usb_type_descriptor) else 'unknown')
        print(f'RealSense {device.get_info(rs.camera_info.name)} ({self.serial}): '
              f'RGB {intr.width}x{intr.height}, depth {resolution[0]}x{resolution[1]} '
              f'@ {self.fps} FPS; USB {self.usb_type}; distortion {intr.model}')

    def set_depth_resolution(self, resolution):
        """Restart streaming on the same device; restore old settings on failure.

        Return True only when the requested profile starts. SDK startup does
        not guarantee frame delivery; read() still enforces its timeout policy.
        """
        resolution = tuple(resolution)
        if resolution == self.depth_resolution:
            return True
        if resolution not in self.available_depth_resolutions:
            warnings.warn(f'Depth {resolution} is not advertised at {self.fps} FPS; '
                          'keeping current profile.', RuntimeWarning, stacklevel=2)
            return False
        previous = self.depth_resolution
        self.depth_m = None
        self.release()
        try:
            self._start(resolution)
        except Exception as exc:
            self.release()
            try:
                self._start(previous)
            except Exception as restore_exc:
                self.release()
                raise RuntimeError(
                    f'Depth switch to {resolution} failed ({exc}); restoring '
                    f'{previous} also failed ({restore_exc}). Reconnect the camera.') from restore_exc
            warnings.warn(f'Depth switch to {resolution} failed ({exc}); '
                          f'restored {previous}.', RuntimeWarning, stacklevel=2)
            return False
        return True

    def cycle_depth_resolution(self):
        """Select the next advertised preset for the current stream FPS."""
        choices = self.available_depth_resolutions
        if not choices:
            warnings.warn('No supported depth presets advertised at this FPS.',
                          RuntimeWarning, stacklevel=2)
            return False
        index = choices.index(self.depth_resolution) if self.depth_resolution in choices else -1
        return self.set_depth_resolution(choices[(index + 1) % len(choices)])


    def read(self):
        """Return a BGR image with depth from the same aligned frameset."""
        # Clear stale depth before any wait or alignment failure. Allow a
        # bounded startup/transport delay without silently reusing old frames.
        self.depth_m = None
        for attempt in range(3):
            arrived, frames = self.pipeline.try_wait_for_frames(timeout_ms=5000)
            if arrived:
                break
            if attempt < 2:
                warnings.warn('No RealSense frames within 5 seconds; retrying '
                              f'({attempt + 1}/2).', RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(
                f'No RealSense frames after three 5-second waits '
                f'(serial {self.serial}, USB {self.usb_type}, {self.fps} FPS). '
                'A serial number is optional and is not required to receive frames. '
                'Close RealSense Viewer/other camera processes, reconnect directly '
                'to a USB 3 port with a USB 3 data cable, and try --camera-fps 15. '
                'If the timeout persists, test color and depth in RealSense Viewer '
                'with the localizer stopped.')
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            self.depth_m = None
            return False, None
        frame = cv2.remap(np.asanyarray(color.get_data()), *self.maps,
                          interpolation=cv2.INTER_LINEAR)
        self.depth_m = cv2.remap(
            np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale,
            *self.maps, interpolation=cv2.INTER_NEAREST)
        return True, frame

    def get(self, prop):
        """Support the existing capture FPS query."""
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.

    def release(self):
        """Stop streaming, including after startup or processing errors."""
        if self.started:
            self.pipeline.stop()
            self.started = False
