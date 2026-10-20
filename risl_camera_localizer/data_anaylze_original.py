import sqlite3
import os
import csv
import yaml
import rclpy
from bisect import bisect_left

import rclpy.serialization
from rosidl_runtime_py.utilities import get_message

from geometry_msgs.msg import PoseStamped
from risl_camera_msgs.msg import PusherInfo
from scipy.spatial.transform import Rotation as R


# change these two
bag_dir = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 2 Sandy/sandy_jenga"
pose_topic = "/object_poses/jenga_3"


db_files = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
assert len(db_files) == 1, f"Expected 1 .db3 file in {bag_dir}, found {len(db_files)}"
bag_name = db_files[0]
green_topic = "/pusher_data_green"
yellow_topic = "/pusher_data_yellow"
topics_to_export = [pose_topic, green_topic, yellow_topic]
output_dir = bag_dir + "/csv_output"
TOLERANCE = 0.002  # seconds

# Input files
def topic_to_filename(topic: str):
    return topic.replace("/", "_") + ".csv"
pose_file = os.path.join(output_dir, topic_to_filename(pose_topic))
green_file = os.path.join(output_dir, topic_to_filename(green_topic))
yellow_file = os.path.join(output_dir, topic_to_filename(yellow_topic))
output_file = os.path.join(output_dir, "merged_pushers_with_pose.csv")

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Read metadata to get topic ID mappings
with open(os.path.join(bag_dir, "metadata.yaml"), 'r') as f:
    metadata = yaml.safe_load(f)

topic_map = {entry['topic_metadata']['name']: entry['topic_metadata']['type']
             for entry in metadata['rosbag2_bagfile_information']['topics_with_message_count']}

# Connect to DB3
conn = sqlite3.connect(os.path.join(bag_dir, bag_name))
cursor = conn.cursor()

# Load topic info
cursor.execute("SELECT id, name, type FROM topics")
topic_info = cursor.fetchall()
topic_id_map = {name: (id, type_) for id, name, type_ in topic_info}

# Deserialize and export
for topic in topics_to_export:
    if topic not in topic_id_map:
        print(f"Topic {topic} not found.")
        continue

    topic_id, type_str = topic_id_map[topic]

    # Get message class
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

        # Dynamically handle message structure
        if type_str == "geometry_msgs/msg/PoseStamped":
            writer.writerow(["sec", "nanosec",
                             "position_x", "position_y", "position_z",
                             "orientation_r", "orientation_p", "orientation_y"])
            for ts, raw in messages:
                msg = rclpy.serialization.deserialize_message(raw, PoseStamped)
                p = msg.pose.position
                o = msg.pose.orientation
                quat = [o.x, o.y, o.z, o.w]
                euler = R.from_quat(quat).as_euler('xyz')
                writer.writerow([
                    msg.header.stamp.sec, msg.header.stamp.nanosec,
                    p.x, p.y, p.z, euler[0], euler[1], euler[2]
                ])

        elif type_str.endswith("PusherInfo"):
            writer.writerow([
                "sec", "nanosec", "frame_num", "pusher_name",
                "color_r", "color_g", "color_b", "color_a",
                "loc_x", "loc_y", "loc_z",
                "nearest_x", "nearest_y", "nearest_z",
                "kappa", "object_index", "local_contour_index"
            ])
            for ts, raw in messages:
                msg = rclpy.serialization.deserialize_message(raw, PusherInfo)
                writer.writerow([
                    msg.header.stamp.sec, msg.header.stamp.nanosec,
                    msg.frame_num, msg.pusher_name,
                    msg.color.r, msg.color.g, msg.color.b, msg.color.a,
                    msg.pusher_location.x, msg.pusher_location.y, msg.pusher_location.z,
                    msg.nearest_point.x, msg.nearest_point.y, msg.nearest_point.z,
                    msg.kappa, msg.object_index, msg.local_contour_index
                ])

        else:
            print(f"Unsupported message type: {type_str}")

    print(f"Exported decoded messages from {topic} to {out_path}")

conn.close()

def load_csv_with_timestamps(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            data[ts] = row
    return data


def load_pose_data(path):
    poses = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            poses.append((t, row))
    poses.sort(key=lambda x: x[0])
    return poses


def find_closest_pose(ts, poses, pose_times):
    idx = bisect_left(pose_times, ts)
    best = None
    best_dt = float("inf")
    for offset in [-1, 0, 1]:
        i = idx + offset
        if 0 <= i < len(poses):
            dt = abs(poses[i][0] - ts)
            if dt < best_dt:
                best_dt = dt
                best = poses[i][1]
    return best if best_dt <= TOLERANCE else None


# Load data
green_data = load_csv_with_timestamps(green_file)
yellow_data = load_csv_with_timestamps(yellow_file)
poses = load_pose_data(pose_file)
pose_times = [p[0] for p in poses]

# Merge timestamps (union of green and yellow)
merged_times = sorted(set(green_data.keys()) | set(yellow_data.keys()))


# Write output
with open(output_file, "w", newline="") as f_out:
    fieldnames = [
        "timestamp",
        "green_frame", 
        "green_index",
        "yellow_frame",
        "yellow_index",
        "pose_x", "pose_y", 
        "ori_y"
    ]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for ts in merged_times:
        row = {"timestamp": f"{ts:.9f}"}

        g = green_data.get(ts)
        y = yellow_data.get(ts)
        p = find_closest_pose(ts, poses, pose_times)

        # Skip if no pushers present
        if not g and not y:
            continue

        # Fill green
        if g:
            row.update({
                "green_frame": g["frame_num"],
                "green_index": g["local_contour_index"],
            })

        # Fill yellow
        if y:
            row.update({
                "yellow_frame": y["frame_num"],
                "yellow_index": y["local_contour_index"],
            })

        # Fill pose
        if p:
            row.update({
                "pose_x": p["position_x"],
                "pose_y": p["position_y"],
                "ori_y": p["orientation_y"],
            })

        writer.writerow(row)

print(f"✅ Merged alignment saved to: {output_file}")