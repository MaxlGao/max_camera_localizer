import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_file = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 3/jenga/csv_output/merged_pushers_with_pose.csv"

pose_x, pose_y, ori_y = [], [], []
single_x, single_y, single_z = [], [], []
double_x, double_y, double_z = [], [], []

with open(input_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row.get("pose_x") or not row.get("pose_y") or not row.get("ori_y"):
            continue

        x = float(row["pose_x"])
        y = float(row["pose_y"])
        z = float(row["ori_y"])

        green_present = row.get("green_index") not in ("", None)
        yellow_present = row.get("yellow_index") not in ("", None)

        if green_present and yellow_present:
            double_x.append(x)
            double_y.append(y)
            double_z.append(z)
        else:
            single_x.append(x)
            single_y.append(y)
            single_z.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(single_x, single_y, single_z, c='blue', label='Single Contact')
ax.scatter(double_x, double_y, double_z, c='orange', label='Double Contact')

ax.set_xlabel('pose_x')
ax.set_ylabel('pose_y')
ax.set_zlabel('ori_y')
ax.set_title('3D Visualization of Pose & Orientation')
ax.legend()
plt.tight_layout()
plt.show()