# Object Localization for Camera mounted on UR End-Effector

This is a ROS 2 Humble Package that outputs aruco tag position and orientation *in the robot base frame* using camera inputs and ROS 2 communication with the UR driver. 

# Setup
Usage of this package will require the UR ROS 2 driver and cv2. 

To install, place this repository in the src directory of your workspace. Also install the repository `risl_camera_msgs` in the same location. The repository is [here](https://github.com/MaxlGao/risl_camera_msgs). Then, build your workspace. 

To run, first make sure a UR driver is running. Then, run `ros2 run risl_camera_localizer localize` inside your workspace. 

A preview window will appear, picking one camera accessible to the system. Press ESC to select this camera, or press any other key to move onto the next camera. Once you select your camera, a more detailed window will appear marking detected objects and annotating details in the top left corner. If you know your camera ID and would like to skip this process, pass the additional arg `--camera-id [YOUR_ID]`. Be warned; USB cameras can change IDs between uses.  

While running, this script will publish estimated camera pose (in the UR frame) in `/camera_pose` as well as poses of any detected objects in `/object_poses/[object_name]`. Poses are given as `PoseStamped`.  

This script will doubly print its findings to console. If you wish, you may pass the argument `--suppress-prints` to not see this.

This script assumes a fixed translation from the end effector frame origin and the focal point of the camera. If the camera is ever repositioned, please measure or calibrate, and update the translation or rotation in `localization_bridge.py`

## Example Commands:
The following commands assume your camera is ID 8. Please remove this argument before first running the command. 

Run the following to display objects and any detected (yellow and green) pushers:

`ros2 run risl_camera_localizer localize --camera-id 8 --suppress-prints`

Run the following to display objects without pushers:

`ros2 run risl_camera_localizer localize --camera-id 8 --suppress-prints --no-pushers`

Run the following to display objects without pushers, while recommending pushes:

`ros2 run risl_camera_localizer localize --camera-id 8 --suppress-prints --no-pushers --recommend-push`

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
