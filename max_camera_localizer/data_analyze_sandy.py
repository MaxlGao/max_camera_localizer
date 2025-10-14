import sqlite3
import os
import csv
import yaml
import rclpy
from bisect import bisect_left
import copy
import numpy as np
import rclpy.serialization
from rosidl_runtime_py.utilities import get_message

from geometry_msgs.msg import PoseStamped
from max_camera_msgs.msg import PusherInfo
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.merged_localization import TARGET_POSES

# Config
parent_folder = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 3/"
object_names = ["jenga", "wrench", "allen_key"]
pose_topics = ["/object_poses/jenga_3", "/object_poses/wrench", "/object_poses/allen_key"]

TOLERANCE = 0.002  # seconds
JUMP_THRESHOLD = 20
WINDOW_SIZE = 5
KEPT_PROPORTION = 0.80
BANNED_WRENCH_INDICES = range(220, 550)

# Helpers
def load_csv_with_timestamps(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            data[ts] = row
    return data

def load_pose_data(path):
    poses = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            poses.append((t, row))
    poses.sort(key=lambda x: x[0])
    return poses

def find_closest_pose(ts, poses, pose_times):
    # Because object pose publishes don't exactly line up with contact point publishes, 
    # there needs to be a heuristic to align the two measures to the same frame. 
    # Luckily the publishes tend to be within about a millisecond of each other. 
    idx = bisect_left(pose_times, ts)
    best, best_dt = None, float("inf")
    for offset in [-1, 0, 1]:
        i = idx + offset
        if 0 <= i < len(poses):
            dt = abs(poses[i][0] - ts)
            if dt < best_dt:
                best_dt = dt
                best = poses[i][1]
    return best if best_dt <= TOLERANCE else None

def clean_contour_data(rows, jump_threshold=JUMP_THRESHOLD, window_size=WINDOW_SIZE):
    # Many times throughout the demonstration process, short mistakes happen. 
    # The pushers slip, the pushers hover over the object, or yellow things that look like 
    # pushers get real close to the object. 
    # All this function tries to do is check if there are a bunch of outlying indices that shouldn't be there. 
    # This is agnostic to frame number. ANY four or fewer contigual unusual numbers are deleted.
    
    def get_index(row, color):
        # Gives the index number from a given row, for a corresponding color (yellow or green)
        ind = row.get(color, None)
        if ind is not None:
            ind = int(ind)
        return ind

    def is_different(ind1, ind2):
        # Compares current and subsequent index numbers... (assuming for one single color)
        if ind1 is None and ind2 is not None:
            # If an index number suddenly appears, that's a change
            return True
        elif ind2 is None and ind1 is not None:
            # If an index number suddenly disappears, that's a change
            return True
        elif ind1 is None and ind2 is None:
            # If an index number stays absent, no change.
            return False
        elif abs(ind1 - ind2) > jump_threshold:
            # If an index number stays present, but changes, that's a change.
            return True
        else:
            # The only other case is the index number not changing a lot.
            return False
    
    # Process green and yellow separately.
    for color in ['green_index', 'yellow_index']:
        i = 0
        deleted = 0
        while i < len(rows) - (window_size + 1):
            cur = get_index(rows[i], color)
            # print(f"CURRENT {cur}")
            nxt = get_index(rows[i+1], color)
            # print(f"NEXT {nxt}")
            if cur is None and nxt is None:
                # Keep going until your window has something in the first two rows.
                i += 1
                continue 
                
            if is_different(cur, nxt):
                # Check if nxt+1 is different from nxt. Pop nxt if so. Don't care if it's different from cur.
                # If not, check if nxt+2 is different from nxt. Pop nxt if so (nxt+1 will follow in the next loop). 
                # and so on until window end.

                for j in range(2, window_size):
                    nxtnxt = get_index(rows[i+j], color)
                    # print(f"NEXTNEXT {nxtnxt}")
                    if is_different(nxt, nxtnxt):
                        # st = rows[i+1]["unified_frame"]
                        # print(f"popped {st}")
                        deleted += 1
                        del rows[i+1]
                        i -= 1 # retract (i increments automatically)
                        break
                # If all your nxt's are similar to nxt (even if different from cur), we're good to move on.
            i += 1
        print(f"Deleted {deleted} outliers from class {color}")
    return rows

def group_and_trim(rows, threshold=JUMP_THRESHOLD):
    # If ALL present indices jump by a threshold, define as a new push. 
    original_length = len(rows)
    grouped = []
    group = [rows[0]]
    group_id = 0
    rows[0]["group"] = group_id
    for i in range(1, len(rows)):
        try:
            prev = int(rows[i - 1]["green_index"] or rows[i - 1]["yellow_index"])
            curr = int(rows[i]["green_index"] or rows[i]["yellow_index"])
        except:
            group.append(rows[i])
            rows[i]["group"] = group_id
            continue
        if abs(curr - prev) <= threshold:
            group.append(rows[i])
            rows[i]["group"] = group_id
        else:
            grouped.append(group)
            group_id += 1
            group = [rows[i]]
            rows[i]["group"] = group_id
    if group:
        grouped.append(group)

    trimmed = []
    for g in grouped:
        cutoff = max(1, int(len(g) * KEPT_PROPORTION))
        trimmed.extend(g[:cutoff])
    print(f"Trimmed groups; reduced rows from {original_length} to {len(trimmed)}")
    return trimmed

def remove_bad_pushes(rows):
    # Extra utility to pop any rows that lead to a worse score across the board.
    i = 0
    good = 0
    bad = 0
    while i < len(rows) - 1:
        try:
            cur = float(rows[i]["distance"])  , float(rows[i]["ori_y"])
            # print(f"CURRENT: {cur}")
            nxt = float(rows[i+1]["distance"]), float(rows[i+1]["ori_y"])
            # print(f"NEXT: {nxt}")
            if nxt[0] - cur[0] > 0 and abs(nxt[1] - cur[1]) > 0:
                fr = rows[i]["unified_frame"]
                bad += 1
                # print(f"popped worsening row (frame {fr})")
                del rows[i]
            else:
                good += 1
                i += 1
        except:
            rw = rows[i]["unified_frame"]
            print(f"That row (Frame {rw}) had no data!")
            # sometimes there be no data
            i += 1

    print(f"Popped {bad} bad rows, kept {good} ones")
    return rows

for object_name, pose_topic in zip(object_names, pose_topics):
    bag_dir = parent_folder + object_name
    target_pose = TARGET_POSES[object_name] # (position mm), (orientation degrees)
    (targ_x, targ_y, _), (_, _, targ_yw) = target_pose
    targ_x, targ_y = targ_x*.001, targ_y*.001
    targ_yw = np.deg2rad(targ_yw)

    db_files = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
    assert len(db_files) == 1, f"Expected 1 .db3 file in {bag_dir}, found {len(db_files)}"
    bag_name = db_files[0]
    green_topic = "/pusher_data_green"
    yellow_topic = "/pusher_data_yellow"
    topics_to_export = [pose_topic, green_topic, yellow_topic]
    output_dir = bag_dir + "/csv_output"

    # Paths
    pose_file = os.path.join(output_dir, pose_topic.replace("/", "_") + ".csv")
    green_file = os.path.join(output_dir, green_topic.replace("/", "_") + ".csv")
    yellow_file = os.path.join(output_dir, yellow_topic.replace("/", "_") + ".csv")
    output_file_1 = os.path.join(output_dir, "test_1.csv")
    output_file_2 = os.path.join(output_dir, "test_2.csv")
    output_file_3 = os.path.join(output_dir, "test_3.csv")
    output_file = os.path.join(output_dir, "merged_pushers_with_pose.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Load topic metadata
    topic_map = {
        entry['topic_metadata']['name']: entry['topic_metadata']['type']
        for entry in yaml.safe_load(open(os.path.join(bag_dir, "metadata.yaml")))
        ['rosbag2_bagfile_information']['topics_with_message_count']
    }

    # Connect and query SQLite
    conn = sqlite3.connect(os.path.join(bag_dir, bag_name))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, type FROM topics")
    topic_id_map = {name: (id, type_) for id, name, type_ in cursor.fetchall()}

    # Export each topic
    for topic in topics_to_export:
        if topic not in topic_id_map:
            print(f"Topic {topic} not found.")
            continue

        topic_id, type_str = topic_id_map[topic]
        try:
            msg_type = get_message(type_str)
        except Exception as e:
            print(f"Could not load message type for {type_str}: {e}")
            continue

        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (topic_id,))
        messages = cursor.fetchall()

        out_path = os.path.join(output_dir, topic.replace("/", "_") + ".csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            if type_str == "geometry_msgs/msg/PoseStamped":
                writer.writerow(["sec", "nanosec", "position_x", "position_y", "position_z", "orientation_r", "orientation_p", "orientation_y"])
                for ts, raw in messages:
                    msg = rclpy.serialization.deserialize_message(raw, PoseStamped)
                    p, o = msg.pose.position, msg.pose.orientation
                    euler = R.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
                    writer.writerow([msg.header.stamp.sec, msg.header.stamp.nanosec, p.x - targ_x, p.y - targ_y, p.z, euler[0], euler[1], euler[2] - targ_yw])

            elif type_str.endswith("PusherInfo"):
                writer.writerow(["sec", "nanosec", "frame_num", "pusher_name", "color_r", "color_g", "color_b", "color_a", "loc_x", "loc_y", "loc_z", "nearest_x", "nearest_y", "nearest_z", "kappa", "object_index", "local_contour_index"])
                for ts, raw in messages:
                    msg = rclpy.serialization.deserialize_message(raw, PusherInfo)
                    writer.writerow([
                        msg.header.stamp.sec, msg.header.stamp.nanosec, msg.frame_num, msg.pusher_name,
                        msg.color.r, msg.color.g, msg.color.b, msg.color.a,
                        msg.pusher_location.x, msg.pusher_location.y, msg.pusher_location.z,
                        msg.nearest_point.x, msg.nearest_point.y, msg.nearest_point.z,
                        msg.kappa, msg.object_index, msg.local_contour_index
                    ])
            else:
                print(f"Unsupported message type: {type_str}")
    conn.close()


    # Load data from individual files
    green_data = load_csv_with_timestamps(green_file)
    yellow_data = load_csv_with_timestamps(yellow_file)
    poses = load_pose_data(pose_file)
    pose_times = [p[0] for p in poses]

    data_times = sorted(set(green_data.keys()) | set(yellow_data.keys()))
    rows = []
    for ts in data_times: # Make the rows object
        if not (green_data.get(ts) or yellow_data.get(ts)):
            continue
        row = {"timestamp": f"{ts:.9f}"}
        if (g := green_data.get(ts)):
            row.update({"unified_frame": g["frame_num"]})
            row.update({"green_index": g["local_contour_index"]})
        if (y := yellow_data.get(ts)):
            row.update({"unified_frame": y["frame_num"]})
            row.update({"yellow_index": y["local_contour_index"]})
        if (p := find_closest_pose(ts, poses, pose_times)):
            row.update({"pose_x": p["position_x"], "pose_y": p["position_y"], "ori_y": p["orientation_y"]})
            row.update({"distance": str((float(p["position_x"])**2 + float(p["position_y"])**2)**0.5), 
                        "disp_angle": str(np.arctan2(float(p["position_y"]), float(p["position_x"])))})
        rows.append(row)

    with open(output_file_1, "w", newline="") as f_out:
        fieldnames = ["timestamp", "unified_frame", "green_index", "yellow_index", "pose_x", "pose_y", "ori_y", "distance", "disp_angle", "group"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rows = clean_contour_data(rows)
    # rows = remove_bad_pushes(rows)

    with open(output_file_2, "w", newline="") as f_out:
        fieldnames = ["timestamp", "unified_frame", "green_index", "yellow_index", "pose_x", "pose_y", "ori_y", "distance", "disp_angle", "group"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rows = group_and_trim(rows)

    with open(output_file_3, "w", newline="") as f_out:
        fieldnames = ["timestamp", "unified_frame", "green_index", "yellow_index", "pose_x", "pose_y", "ori_y", "distance", "disp_angle", "group"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Deletion of specific indices on the wrench
    if "wrench" in pose_topic:
        deleted = 0
        for row in rows:
            if "green_index" in row:
                if int(row["green_index"]) in BANNED_WRENCH_INDICES:
                    ind = row["green_index"]
                    deleted += 1
                    # print(f"Deleted illegal index {ind}")
                    del row
                    continue
            if "yellow_index" in row:
                if int(row["yellow_index"]) in BANNED_WRENCH_INDICES:
                    ind = row["yellow_index"]
                    deleted += 1
                    # print(f"Deleted illegal index {ind}")
                    del row
        print(f"Deleted {deleted} wrench rows for having impossible indices.")

    if "jenga" in pose_topic:
        print(f"There are {len(rows)} rows in the jenga table")

    # Duplication of yaw angles to extend range to -2pi, 2pi. 
    duplicated = []
    for row in rows:
        if "ori_y" not in row:
            continue
        angle = float(row["ori_y"])
        if -np.pi <= angle < 0:
            new_row = copy.deepcopy(row)
            new_row["ori_y"] = str(angle + 2*np.pi)
            duplicated.append(new_row)
        elif 0 <= angle < np.pi:
            new_row = copy.deepcopy(row)
            new_row["ori_y"] = str(angle - 2*np.pi)
            duplicated.append(new_row)
    rows.extend(duplicated)

    # Additional Jenga-specific duplication
    if "jenga" in pose_topic:
        print(f"There are {len(rows)} rows in the jenga table")
        duplicated = []
        for row in rows:
            if "ori_y" not in row:
                continue
            angle = float(row["ori_y"])
            if -np.pi <= angle <= np.pi:
                new_row = copy.deepcopy(row)
                # print(f"Original Row {new_row}")
                if "green_index" in new_row: # 180 degree rotation = add/subtract 500 from contour
                    new_row["green_index"] = str((int(new_row["green_index"]) + 500) % 1000) 
                if "yellow_index" in new_row:
                    new_row["yellow_index"] = str((int(new_row["yellow_index"]) + 500) % 1000) 
                new_row["ori_y"] = str(angle + np.pi)
                duplicated.append(new_row)
                # print(f"First Row {new_row}")
                newnew_row = copy.deepcopy(new_row)
                newnew_row["ori_y"] = str(angle - np.pi)
                # print(f"Second Row {newnew_row}")
                duplicated.append(newnew_row)
                # exit()
        rows.extend(duplicated)
        print(f"There are now {len(rows)} rows in the jenga table")



    with open(output_file, "w", newline="") as f_out:
        fieldnames = ["timestamp", "unified_frame", "green_index", "yellow_index", "pose_x", "pose_y", "ori_y", "distance", "disp_angle", "group"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\u2705 Merged alignment saved to: {output_file}")
