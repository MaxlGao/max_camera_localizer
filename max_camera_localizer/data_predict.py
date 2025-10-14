from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from ament_index_python.packages import get_package_share_directory

file_folder = get_package_share_directory("max_camera_localizer")
file_folder = os.path.join(file_folder, "pusher_data")

# Load the uploaded CSV file
# file_folder = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 3/"
data_files = {
    "jenga": "jenga/csv_output/merged_pushers_with_pose.csv",
    "allen_key": "allen_key/csv_output/merged_pushers_with_pose.csv",
    "wrench": "wrench/csv_output/merged_pushers_with_pose.csv",
}

# Input feature columns
input_cols = ['distance', 'disp_angle', 'ori_y']
models = {}

for object_name, file_name in data_files.items():
    print(f"Processing {object_name} data...")
    path = os.path.join(file_folder, file_name)
    df = pd.read_csv(path)

    # Create 'pusher_class': 1 if both present, 0 if only one
    df['pusher_class'] = ((df['green_index'].notna()) & (df['yellow_index'].notna())).astype(int)

    # Train classifier
    X = df[input_cols]
    y_class = df['pusher_class']
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y_class)

    # One-pusher regressor
    one_pusher = df[df['pusher_class'] == 0].copy()
    one_pusher['index_value'] = one_pusher['green_index'].combine_first(one_pusher['yellow_index'])
    clf_one = RandomForestClassifier(random_state=42)
    clf_one.fit(one_pusher[input_cols], one_pusher['index_value']) # this takes a while
    print("Done with Single Pusher...")

    # Two-pusher classifier
    two_pushers = df[df['pusher_class'] == 1].copy()

    # Assign a unique ID to each valid (green, yellow) pair
    two_pushers['pair_label'] = two_pushers[['green_index', 'yellow_index']].apply(tuple, axis=1)
    pair_label_map = {pair: i for i, pair in enumerate(two_pushers['pair_label'].unique())}
    print(f"Made pair label map of length {len(pair_label_map)}")
    inverse_pair_label_map = {v: k for k, v in pair_label_map.items()}
    two_pushers['pair_class'] = two_pushers['pair_label'].map(pair_label_map)

    clf_two = RandomForestClassifier(random_state=42)
    clf_two.fit(two_pushers[input_cols], two_pushers['pair_class'])
    print("Done with Double Pusher.")

    # Store models for object
    models[object_name] = {
        'classifier': clf,
        'clf_one': clf_one,
        'clf_two': clf_two,
        'inverse_pair_label_map': inverse_pair_label_map,
    }

# Function to make predictions on new input samples
def predict_pusher_outputs(name, pose_x, pose_y, ori_y, target_pose):
    if "jenga" in name:
        # Cut yaw to 0-180 degree range, predict, 
        pass
    (targ_x, targ_y, _), (_, _, targ_yw) = target_pose # position mm and orientation degrees
    targ_x, targ_y = targ_x*.001, targ_y*.001
    targ_yw = np.deg2rad(targ_yw)
    d_x = pose_x-targ_x
    d_y = pose_y-targ_y
    d_yw = ori_y - targ_yw
    distance = (d_x**2+d_y**2)**0.5
    disp_ang = np.arctan2(d_y, d_x)
    print(f"dist: {distance:.3f}, th: {disp_ang:.3f}, yw: {d_yw:.3f}")
    model = models[name]
    sample = pd.DataFrame([[distance, disp_ang, d_yw]], columns=input_cols)

    pusher_type = model['classifier'].predict(sample)[0]
    print(pusher_type)
    
    if pusher_type == 0:
        index_value = model['clf_one'].predict(sample)[0]
        return {
            'object_id': name,
            'pusher_class': 'one_pusher',
            'predicted_index': [int(index_value)]
        }
    else:
        pair_class = model['clf_two'].predict(sample)[0]
        green_index, yellow_index = model['inverse_pair_label_map'][pair_class]
        return {
            'object_id': name,
            'pusher_class': 'two_pusher',
            'predicted_index': [int(green_index), int(yellow_index)]
        }
