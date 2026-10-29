# Object Localization for Camera mounted on UR End-Effector

This is a ROS 2 Humble Package that outputs aruco tag position and orientation *in the robot base frame* using camera inputs and ROS 2 communication with the UR driver. 

# Setup
Usage of this package will require the UR ROS 2 driver and OpenCV 4.7 or newer
(`opencv-python`, imported as `cv2`). The marker detector uses the modern
`cv2.aruco.ArucoDetector` API. Older OpenCV versions supplied by some ROS 2
Humble systems do not provide this API.

Install OpenCV into the Python environment used to run the ROS nodes:

```bash
python -m pip install 'opencv-python>=4.7'
```

Use the desktop package because the localizer uses `cv2.imshow`; headless
packages cannot display its camera preview. Install only one OpenCV wheel
variant in an environment, since they share the `cv2` namespace.

Synthetic checks of marker detection, pose estimation, color blobs, Kalman
filtering, and drawing passed with OpenCV 4.12.0 and 5.0.0 and NumPy 2.2.6.
Camera capture, exposure controls, and GUI display still require a hardware
check with the intended camera and desktop session.

## Toggle blue-blob localization

Start with only ArUco localization:

```bash
ros2 run risl_camera_localizer localize --no-blue-blobs --suppress-prints
# Or through bringup:
ros2 launch risl_bringup cell.launch.py blue_blobs:=false
```

Press **b** with the preview window focused to toggle blue blobs on or off
while running. The preview shows ON/OFF. Blue blobs remain enabled by default;
`--blue-blobs` explicitly enables them at startup. Disabling skips color
segmentation, object identification, and missing-object recovery, and stops new
blob-derived poses and contours from being published. ArUco continues normally.
Recovery history is cleared on toggle, so re-enabling uses fresh observations.
Existing ROS subscribers may retain their last received poses; disabling does
not publish deletion messages or stop a separately running controller.
The HSV thresholds and camera color settings are unchanged by this switch.

## Visibility-aware ArUco object permanence

Enabled by default. Confirmed tags keep their last measured **world pose**
while hidden; the tracker does not extrapolate hidden objects or move their
stored pose with the camera. The current robot camera transform reprojects
that stored location on every frame to determine whether the tag should be
readable. No valid RealSense depth is required for this policy.

* A fully in-frame, front-facing, sufficiently large marker location with no
  known cover expires after **5 consecutive clear-view misses**.
* A missing marker outside the image, crossing its border margin, behind the
  camera, nearly edge-on/back-facing, or too small to decode is protected.
* A missing marker covered by a **currently observed Jenga block**, or by a
  remembered block protected through an occlusion chain, is protected. Rays to
  a 5x5 grid inside the marker are tested against each trusted block's
  72x24x14 mm oriented volume; at least 30% coverage protects the tag. Intersections
  must be at least 3 mm closer than the remembered surface. Full block geometry
  is used, so the occluding marker itself need not overlap the hidden marker.
* Protected tracks persist indefinitely by default. The clear-view miss counter
  resets while protected or when the tag is seen. When an occluder moves away
  or the camera returns, an empty old location therefore expires promptly.
* Occlusion propagates through remembered tower layers: visible top → hidden
  middle → hidden bottom blocks. Current successful marker pose fits seed the
  chain. Each protected ghost contributes its unchanged block volume to the
  next layer, so protection does not depend on tracker iteration order.
* The chain is rebuilt from current observations on every frame. Old chain
  metadata never seeds protection, and a group of ghosts without an observed
  anchor cannot sustain a cycle. If the top moves away, old protection is
  immediately reevaluated and clear missing locations accumulate misses again.
  Out-of-frame/edge-on/too-small tracks retain their existing protection, but
  do not independently seed occlusion chains. Unconfirmed ghosts and tracks
  expiring under an optional protected timeout cannot extend a chain.
* A tag that expires must pass fresh five-frame confirmation when it reappears.

```bash
ros2 run risl_camera_localizer localize --no-blue-blobs --visible-miss-frames 5
# Optional cap on consecutive protected misses (0 means unlimited):
ros2 run risl_camera_localizer localize --protected-miss-frames 900
# Compare with the original fixed 15-frame expiration:
ros2 run risl_camera_localizer localize --no-object-permanence
# Bringup equivalents:
ros2 launch risl_bringup cell.launch.py object_permanence:=true visible_miss_frames:=5 protected_miss_frames:=0
```

Remembered poses appear in amber with `occluded`, `out_of_frame`, `edge_on`,
`too_small`, or `visible` status; observed poses remain green. Multi-layer
retention shows a chain such as `40>30>10` in the preview and console, meaning
observed 40 protects remembered 30, which protects remembered 10. Console `KEEP`
lines explain retention and `DROP` records expiration. Test by first observing
all blocks, then covering two bottom tags with a crosswise middle block and
covering the middle tag with an offset top block. Both bottom blocks should
remain, even when only one is directly under the top. Move another
block away from an otherwise clear remembered location: it should disappear
after five processed clear-view frames. Pan away and return, then remove the
cover to check that newly empty locations expire. Five frames is about 0.17 s
at 30 processed FPS; thresholds count processed frames, not wall-clock time.

This is geometric evidence about the last observed location, not proof a hidden
object has remained still. Accuracy depends on mounting calibration, marker-to-
block alignment (tag centered on the top face, X along the long side), dimensions,
and pose height. Use `planar-z` or `3d` for stacks; fixed-height `planar` cannot
resolve layers. Hands, grippers, untagged blocks, and blue-blob objects are not
modeled as occluders in this first version. Poor lighting can also produce
clear-view misses. Known out-of-frame tracks can remain indefinitely until the
camera returns or an optional protected timeout expires.

ROS pose topics continue publishing retained world poses as `PoseStamped`;
this existing message carries no inferred/visibility field. Dropping a track
stops new pose/contour output and removes it from the preview, but does not
send a deletion message to subscribers that cache the last pose. Existing
controllers must apply their own freshness/validity policy. Depth-profile
switching still clears tracking history and requires fresh confirmation.

## RealSense D415 capture (default)

Install the SDK Python bindings in the same Python environment as the ROS node:

```bash
source .venv/bin/activate
python -m pip install 'pyrealsense2>=2.55'
python /usr/bin/colcon build --packages-select risl_camera_localizer risl_bringup --symlink-install
source install/setup.bash
ros2 run risl_camera_localizer localize --suppress-prints
```

Build with the virtual environment Python as shown: the installed ROS console
script must use the interpreter containing `pyrealsense2`. Activating a venv
after building does not change an existing console script shebang.

The default backend is now `realsense`, with synchronized 1280x720 color and
640x360 Z16 depth at 30 FPS. It uses SDK factory color intrinsics and depth scale,
aligns depth to color, and rectifies both images into the same color optical
frame. Rectification maps use SDK projection for the reported distortion model,
including inverse Brown-Conrady; coefficients are not treated as interchangeable
OpenCV distortion models. Startup prints the model for diagnostics. All
projection and drawing use that calibration. No separate RealSense
ROS driver should own the same device while this process is running.
Select a device by stable serial, or reduce stream rate if necessary:

```bash
ros2 run risl_camera_localizer localize --realsense-serial YOUR_SERIAL --camera-fps 15
ros2 launch risl_bringup cell.launch.py realsense_serial:=YOUR_SERIAL pose_mode:=planar-z
```

Choose depth resolution independently of RGB:

```bash
ros2 run risl_camera_localizer localize --depth-resolution 640x360 --no-blue-blobs
ros2 launch risl_bringup cell.launch.py depth_resolution:=640x360 blue_blobs:=false
```

Press **d** in the preview to cycle 424x240 → 640x360 → 848x480 → 1280x720,
skipping presets the selected camera does not advertise as Z16 at the current
FPS. The preview shows the native depth resolution. RGB remains 1280x720;
aligned depth is still mapped into that color image for marker sampling.
640x320 is not an offered preset. Selecting a startup profile still requires
that the camera supports it in combination with RGB at the selected FPS.

Switching stops and restarts the SDK pipeline on the same serial number,
briefly interrupting both color and depth. Calibration and alignment are
refreshed, old depth is discarded, and trackers clear so detections must confirm
again after the gap. If a new profile fails to start, the previous one is
restored; if restoration also fails, localization exits with an error. A profile
starting does not guarantee subsequent frame delivery: the existing bounded
frame timeout still applies. Runtime switching is manual; no automatic profile
changes based on distance are performed. Sensor exposure may need to settle
after a restart. Existing subscribers may retain their previous pose during
the capture gap; restarting capture does not stop a controller.

`planar-z` is the default pose mode. Valid depth inside an inset marker polygon
provides surface height; the four RGB corners still determine horizontal
position and yaw. A robust plane fit excludes marker boundaries and isolated
outliers. Depth must be finite, between 0.15 and 3 m, sufficiently dense and
planar, and agree with visual optical Z within 0.10 m. These are conservative
software gates, not guaranteed D415 operating limits or accuracy claims.
Missing or rejected depth uses the existing RGB marker-size estimate for that
frame. The preview labels each visible tag `RGB-D`, `RGB fallback`, or
`fixed plane`. Temporal hole filling is deliberately avoided for moving blocks.

`3d` uses the measured depth plane to refine translation along the visually
estimated marker-origin ray; orientation remains the RGB IPPE estimate.
`planar` explicitly retains the fixed world height and does not use depth.
Blue blob HSV thresholds, morphology, grouping, and fixed-height ray/plane
localization are unchanged; on RealSense they receive the calibrated,
rectified RGB image. SDK color auto exposure/white balance retain device defaults;
the legacy OpenCV exposure values are not portable to SDK sensor controls.
Check blue tape detection under the actual lighting before collecting data.

Legacy capture remains available explicitly:

```bash
ros2 run risl_camera_localizer localize --camera-backend opencv --camera-id 8
ros2 launch risl_bringup cell.launch.py camera_backend:=opencv
```

The legacy backend retains its existing approximate intrinsics and camera
selection. `--camera-id` is only for OpenCV; RealSense uses serial numbers.
Missing SDK bindings or stream startup failures produce an error rather than
silently switching to another physical camera. A serial number is optional;
startup prints the selected device's serial and reported USB connection type.
Frame acquisition retries up to three 5-second waits before failing, and clears
old depth data on failure. For repeated timeouts, close any camera viewers or
other camera processes, reconnect with a USB 3 data cable directly to a USB 3
port, and try `--camera-fps 15`. With localization stopped, test color and depth
in RealSense Viewer to separate transport/driver problems from application code.

Before relying on robot-base coordinates, calibrate the end-effector to **RGB
optical frame** transform in `localizer_bridge.py`; the SDK sensor calibration
does not supply this robot mounting transform. The existing offset remains an
installation-specific approximation. Validate measured marker-surface Z and
X/Y against known positions at several working distances and across the image,
including stacked blocks and tilted camera views. Cover the depth sensors to
verify RGB fallback and compare stationary pose jitter with the OpenCV backend.
No D415 hardware accuracy measurements have been performed in this environment.

### Strategy for a moving camera at 10–60 cm

Use calibrated RGB ArUco geometry as the continuous measurement source, with
depth as an optional correction when the surface is in range and samples pass
quality checks. At the default 640x360 depth setting, plan for RGB-only poses
below about 24 cm and potentially RGB-D poses above it; the minimum range is
not a guarantee of accuracy at the boundary. Flat objects with a known absolute
surface height can use `--pose-mode planar`; varying heights require `planar-z`
or `3d`, with RGB scale-derived height remaining available during depth dropout.
Close-range RGB accuracy still needs validation for focus, visible corners,
marker size, and the robot-to-camera transform.

Keep color at 1280x720 and choose a depth resolution for the task. 848x480 lowers the specified minimum depth range
to 31 cm, and 640x360 to 24 cm, at a precision cost. Even 424x240 has a specified
16 cm minimum: none of these default D415 profiles spans the full 10–60 cm.
Independent depth resolution selection is available; automatic distance-based
profile selection is not implemented. Avoid frequent profile restarts; assess
RGB/depth transitions against known targets before relying on them for control.

### Fragmented depth and RGB fallback

A mostly flat scene should produce a coherent depth surface. A fragmented
rainbow image with many holes does not establish usable marker depth. Check
camera-to-surface distance first: the D415 datasheet lists a 450 mm minimum Z
at 1280x720, 310 mm at 848x480, and 240 mm at 640x360 (default settings). The
localizer defaults to 1280x720 RGB and 640x360 depth; `--camera-fps` changes
frame rate, not minimum range. Its 0.15 m software acceptance limit does not
mean the hardware can measure that close at this stream resolution.

For diagnosis, move a matte textured target to roughly 0.6–0.8 m, enable the
stereo emitter and stereo auto exposure in Viewer, and inspect both infrared
streams for obstruction, underexposure, or saturation. Compare the Default
and High Density presets; High Density increases fill but does not guarantee
accuracy. Avoid filling holes just to make the image look complete. If a
textured flat target remains fragmented at a valid distance, check calibration
using the manufacturer's Depth Quality Tool before changing calibration.

The localizer preview now shows the depth rejection reason, fraction of samples
inside the marker that are finite and within the software range, and visual
optical Z in meters. Running without `--suppress-prints` also includes compact diagnostic
quantities on the marker line, including depth and plane RMS when available. `sparse depth` means fewer than 12 samples or less than 50% coverage;
`noisy depth surface` means the plane consistency check failed;
`depth disagrees with RGB` means the measured optical Z differs by over 10 cm.
These diagnostics do not relax the acceptance checks or change blob behavior.

See [minimum-Z specifications, Table 4-11](https://realsenseai.com/wp-content/uploads/2024/10/Intel-RealSense-D400-Series-Datasheet-October-2024.pdf#page=89)
and [manufacturer tuning guidance](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/).

SDK references: [depth/color alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
and [projection geometry](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0).

To install, place this repository in the src directory of your workspace. Also install the repository `risl_camera_msgs` in the same location. The repository is [here](https://github.com/MaxlGao/risl_camera_msgs). Then, build your workspace. 

To run, first make sure a UR driver is running. Then, run `ros2 run risl_camera_localizer localize` inside your workspace. 

Select `--pose-mode planar` for blocks resting flat on the
world XY plane. It fits world X, Y, and yaw against all four detected marker
corners, fixes roll and pitch to zero, and fixes the marker surface height.
Planar tracking smooths X/Y and yaw (including wraparound) without introducing
tilt or advancing the state on repeated reads. Missing markers use the visibility-aware retention policy above;
`--no-object-permanence` restores fixed 15-frame expiration.

Set `--marker-height` to the **absolute world Z of the marker surface in meters**.
The default is `0.014` (14 mm block thickness, assuming the table is at world
Z = 0). Include table elevation and marker mounting thickness if applicable:

```bash
ros2 run risl_camera_localizer localize --pose-mode planar --marker-height 0.014
```

All markers share this height. Planar mode assumes horizontal, unstacked blocks;
a wrong height or camera-to-world transform biases X/Y. The OpenCV backend retains approximate camera intrinsics; the RealSense backend
uses device calibration. Both require an accurate camera-to-world transform.
For horizontal blocks at different heights, use `planar-z`. This intermediate
mode fits world X, Y, Z, and yaw, keeping roll and pitch at zero:

```bash
ros2 run risl_camera_localizer localize --pose-mode planar-z
```

Each marker has its own estimated height, which is smoothed along with X/Y and
yaw. `--marker-height` applies only to `planar`; `planar-z` recovers metric height
from the configured marker size and camera calibration. Height is the marker
surface's absolute world Z, not the block center or a discrete stack level.
This mode supports horizontal stacked blocks, but does not infer support contacts
or stacking relationships. Small tags and calibration error can cause Z noise.

For objects that tip or require full orientation, select the original IPPE and
quaternion Kalman tracking behavior:

```bash
ros2 run risl_camera_localizer localize --pose-mode 3d
```

With `--camera-backend opencv`, a preview window appears for each accessible camera. Press any key to select
the camera, or ESC to skip it. Skipping every camera cancels selection. Once you
select your camera, a more detailed window will appear marking detected objects
and annotating details in the top left corner. If you know your camera ID and
would like to skip this process, pass `--camera-backend opencv --camera-id [YOUR_ID]`. USB cameras can
change IDs between uses.

Discovery, preview, and localization request the same 1280x720 capture size.
The preview shows the actual frame size, since a camera may negotiate a
different resolution. On Linux, all three use V4L2 explicitly, so camera ID N
refers to `/dev/videoN` consistently rather than using automatic backend
selection. A physical camera can still expose multiple video device nodes.

While running, this script will publish estimated camera pose (in the UR frame) in `/camera_pose` as well as poses of any detected objects in `/object_poses/[object_name]`. Poses are given as `PoseStamped`.  

All detected ArUco IDs in the configured 4x4 and 5x5 dictionaries are tracked as
`jenga_<id>`. Each confirmed instance gets its own pose topic at
`/vision/object_poses/jenga_<id>` and twist topics under
`/vision/object_twists_world/`, `/vision/object_twists_body/`, and
`/vision/object_twists_push_avg/`. Pose and twist publishers for `jenga_6`,
`wrench`, and `allen_key` are registered at node startup, even if those objects
are out of frame, so allocation can subscribe immediately. Other Jenga
instances are registered when first observed. Creating these publishers does
not emit a pose for an unseen object. All Jenga instances share the same body geometry.
IDs must be unique across the configured dictionaries because instance names
use the numeric ID.

Allocation continues to support Jenga 6 only (alongside the wrench and Allen
key). Other Jenga IDs are localized without replacing allocation's single body
contour. When `/robot/target_name` is set, the contour and push overlay use that
target among the supported allocation objects.

Console output uses one line per marker observation: camera/world XYZ in mm
with one decimal, world quaternion in XYZW order with three decimals, and a
compact depth summary (source, coverage, depth and plane RMS when available).
`HOLD` shows confirmation progress; `GHOST` shows frames since last observation.
These formats affect display only. Use `--suppress-prints` to hide per-marker
output; startup and operator messages remain visible.

This script assumes a fixed translation from the end effector frame origin and the focal point of the camera. If the camera is ever repositioned, please measure or calibrate, and update the translation or rotation in `localization_bridge.py`

## Example Commands:
The following commands assume your camera is ID 8. Please remove this argument before first running the command. 

Run the following to display objects and any detected (yellow and green) pushers:

`ros2 run risl_camera_localizer localize --camera-backend opencv --camera-id 8 --suppress-prints`

Run the following to display objects without pushers:

`ros2 run risl_camera_localizer localize --camera-backend opencv --camera-id 8 --suppress-prints --no-pushers`

Run the following to display objects without pushers, while recommending pushes:

`ros2 run risl_camera_localizer localize --camera-backend opencv --camera-id 8 --suppress-prints --no-pushers --recommend-push`

## Workspace Structure:

```
YOUR_ROS2_WORKSPACE/
├── src/
│   └── risl_camera_localizer/
│       ├── risl_camera_localizer/
│       │   ├── trash/
│       │   │   └── [unused python files]
│       │   ├── camera_selection.py
│       │   ├── data_analyze.py
│       │   ├── data_predict.py
│       │   ├── detection_functions.py
│       │   ├── drawing_functions.py
│       │   ├── geometric_functions.py
│       │   ├── kalman_functions.py
│       │   ├── localizer_bridge.py
│       │   ├── merged_localization.py
│       │   ├── object_frame_definitions.py
│       │   └── process_stl.py
│       ├── STL/
│       │   ├── Allen Key.STL
│       │   └── Wrench.STL
│       ├── pusher_data/
│       │   ├── {see the section on `data_predict.py` below}
│       ├── package.xml
│       ├── README.md
│       ├── setup.cfg
│       └── setup.py
```

## Important Parameters

Most of these scripts have parameters at the top of the file. Often, you will find references to `jenga`, `allen_key`, and `wrench` close to each other. These are the three objects of interest that we work with; you will need to add another reference to add another object.

### `data_analyze.py`

This script contains a variable `parent_folder` which references a rosbag directory and expects there to be a `*.db3` file and `metadata.yaml` inside a given folder. This directory *must* be changed before running this file on a new system. 

### `data_predict.py`

This script contains `file_folder` and `data_files`. Both reference data files that appear in the package's share directory, which are copied from pusher_data in the `src/risl_camera_localizer` folder. If swapping out this data for another set, please rebuild the package.

The script assumes the data files are structured as follows:

```
pusher_data/
├── jenga/
│   ├── csv_output/
│   │   ├── _object_poses_jenga_[NUMBER].csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── jenga_0.db3
│   └── metadata.yaml
├── allen_key/
│   ├── csv_output/
│   │   ├── _object_poses_allen_key.csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── allen_key_0.db3
│   └── metadata.yaml
├── wrench/
│   ├── csv_output/
│   │   ├── _object_poses_wrench.csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── wrench_0.db3
│   └── metadata.yaml
```

### `localizer_bridge.py`

Inside the `LocalizerBridge` class is a set of parameters inside `__init__()`. 

First are `self.cam_offset` for position and quaternion, describing the pose of the camera (focal point) relative to the end-effector (EE). 

Second is a set of initial values for EE pose, to be used if ROS2 fails to update the script with actual values. 

### `merged_localization.py`

`merged_localization.py` is a massive file with many parameters. Listing in current order of appearance:

1. Camera parameters: Image frame height and width, in pixels, plus angular FOV. Distortion coefficient is included. These are specific to the camera hardware used. 
2. `MARKER_SIZE` is the side length of the ArUco marker used for identifying Jenga blocks.
3. `BLOCK_*` are the side measures of the Jenga blocks. 
4. `OBJECT_DICTS` are the side lengths of the triangle formed by the pattern of three blue blobs pertaining to each object. For instance, the allen key is marked with blue tape making a triangle of side lengths 38.8, 102.6, and 129.5 mm. 
5. `TARGET_POSES` are the goal position and orientation for all objects. 
6. `*_range` are the HSV ranges (using cv2 standards, so 0-179 for hue, 0-255 for Sat. and Value) for blue, green, and yellow. Sensitive to lighting conditions (particularly yellow) so tune accordingly.
7. `pusher_distance_max` is the maximum distance (base frame, meters) that a pusher can be from an object's contour while still being recorded. There is no need to recognize pushers too far away to touch an object.

### `object_frame_definitions.py`

The functions `define_body_frame_*` require careful changes for customization or extension to other objects. To summarize, each function does the following:

1. Extracts points `p1, p2, p3` and finds the distances between them. Sorts to find the short, medium, and long side. 
2. Assigns points `A, B, C` in relation to them being intersections of particular side lengths. For instance, `A` might be the point connecting the short and medium sides. 
3. Defines an origin, typically as one of the above points. 
4. Defines a coordinate frame, typically in relation to the above points. 
5. Extracts the object's pose.
6. [defunct, and to be removed later] defines manually-set contact points as points and normals. 
7. References a corresponding `CONTOUR_*` dictionary, transforms the contour's points and normals according to the object pose, and returns a current contour dictionary.

Other functions complete a subset of the above list. If adding objects, be sure to add it where "allen_key" and "wrench" are mentioned.

### `process_stl.py`

The first section of this script extracts trimeshes from given STL files (already installed locally in `./STL/`) and orients them according to the functions in `object_frame_definitions.py`. These are practically manual transformations.

Splines are made of 1000 points, as specified in `n_points`. Additional notable parameters are inside of `splprep()`, which uses an s-value of 1.

# Demonstration
The vision-based localizer, with a camera mounted on a robot's end-effector, can accurately yield the position and orientation of different objects through two primary methods. The first, as seen with the Jenga block, uses CV2's ArUco functions to give a full 6D pose in the base frame. The second, as seen with the allen key and wrench, use color detection to spot patterns of three blue blobs. 

On the top of all following GIFs is a readout of robot end-effector pose, which is required to properly orient the camera in the base frame, which is in turn required to properly orient all objects that the camera sees. When objects come into view, a second readout appears, showing its estimated 6D pose. 

Objects needing blue blob-based detection are assumed to lie flat on the table, thus a constant 10mm z coordinate for the allen key and wrench. 

Multiple objects may be detected at the same time. For best results, however, the working space should be kept as clean as possible. For example, there is a possibility of confusion when the allen key and wrench are placed in such a way that a second allen key pattern can be detected between the two objects.

![Objects Only](./media/Localizer%20Clip%201.gif)

The blue blob method was then extended to the yellow and green ranges, to allow for localization of unique point pushers. When these colors are brought close to the object, the pushers' base frame positions are estimated and compared with the contour of the object. Object contours are calculated from given CAD models, not the image. 

This function is useful for capturing data from human demonstrations. In the GIF below, a human uses a pair of colored chopsticks to non-prehensibly manipulate an allen key to a goal position. Here, contour numbers are recorded for further examination.

![Objects and Pusher](./media/Localizer%20Clip%202.gif)

With sufficient human data, we can at last make recommendations for where on an object to push, given its relative displacement from a target. From there, this recommendation serves as a warm-start to a non-prehensile manipulation process. 

![Object and Recommendation](./media/Localizer%20Clip%203.gif)
