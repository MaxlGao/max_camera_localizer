import os

from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd
from max_camera_localizer.process_stl import CONTOUR_LIBRARY
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_F2V = np.array([0.001691, 0.001691, np.deg2rad(5.234536)], dtype=float)
AX_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18

def _style_axes_labels_and_legend(ax: plt.Axes, legend=None, log=False):
    """Apply shared axis-label and legend font sizes."""
    if log:
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(axis="y", labelsize=AX_LABEL_FONT_SIZE)
    if ax.xaxis.label is not None:
        ax.xaxis.label.set_size(AX_LABEL_FONT_SIZE)
    if ax.yaxis.label is not None:
        ax.yaxis.label.set_size(AX_LABEL_FONT_SIZE)
    if hasattr(ax, 'zaxis') and ax.zaxis.label is not None:
        ax.zaxis.label.set_size(AX_LABEL_FONT_SIZE)

    legend_obj = legend if legend is not None else ax.get_legend()
    if legend_obj is not None:
        for text in legend_obj.get_texts():
            text.set_fontsize(LEGEND_FONT_SIZE)

def characteristic_radius(contour_dict: Dict, k: float = 75) -> float:
    """
    Finds characteristic radius in meters, using the kth percentile of distance from the origin.
    That is, k% of all points are closer to the origin than the yielded 'characteristic radius'
    This is to bias towards points further away from the center of mass.
    """
    body_xyz = contour_dict['xyz'] / 1000.0  # mm -> m
    contour_xy = body_xyz[:, :2]
    dists = np.linalg.norm(contour_xy, axis=1)
    r_char = float(np.percentile(dists, k))
    return r_char

def _get_desired_wrenches(data: pd.DataFrame):
    """Return the unique desired wrench vectors if present."""
    for cols in (
        ['wrench_echo_fx', 'wrench_echo_fy', 'wrench_echo_tz'],
        ['wrench_fx', 'wrench_fy', 'wrench_tz'],
    ):
        if set(cols).issubset(data.columns):
            unique_wrenches = data[cols].drop_duplicates().to_numpy(dtype=float)
            return [row for row in unique_wrenches]
    return []

def _get_surface_type(data: pd.DataFrame):
    """Return normalized surface type when available."""
    for col in ('surface_type', 'surface', 'surface_name'):
        if col in data.columns and not data[col].empty:
            values = {
                str(v).strip().lower()
                for v in data[col].dropna().unique()
                if str(v).strip()
            }
            if 'sticky' in values and 'smooth' in values:
                return 'mixed'
            if 'sticky' in values:
                return 'sticky'
            if 'smooth' in values:
                return 'smooth'
            if values:
                return sorted(values)[0]

    for attr_name in ('source_path', 'source_paths'):
        value = data.attrs.get(attr_name)
        if value is None:
            continue
        paths = value if isinstance(value, (list, tuple)) else [value]
        found = set()
        for path in paths:
            path_lower = str(path).lower()
            if 'sticky' in path_lower:
                found.add('sticky')
            if 'smooth' in path_lower:
                found.add('smooth')
        if 'sticky' in found and 'smooth' in found:
            return 'mixed'
        if 'sticky' in found:
            return 'sticky'
        if 'smooth' in found:
            return 'smooth'
    return 'unknown'

def _set_figure_title(fig: plt.Figure, base_title: str, data: pd.DataFrame):
    """Apply a consistent figure-level title with surface and desired wrenches."""
    parts = [f"{base_title} | Surface: {_get_surface_type(data)}"]
    desired_wrenches = _get_desired_wrenches(data)
    if desired_wrenches:
        if len(desired_wrenches) == 1:
            wrench_str = ", ".join(f"{v:.2f}" for v in desired_wrenches[0])
            parts.append(f"Desired wrench: [{wrench_str}]")
        else:
            wrench_strs = [", ".join(f"{v:.2f}" for v in w) for w in desired_wrenches]
            parts.append(f"Desired wrenches: {', '.join(f'[{s}]' for s in wrench_strs)}")
    fig.suptitle("\n".join(parts))

def _norm_to_length(vec, length=0.05, eps=1e-12):
    """Normalize a vector to a given length (safe for zero vectors)."""
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n * length

def _filter_pushes(df: pd.DataFrame, push_selection):
    """Filter a DataFrame of push data by label/index selection."""
    if push_selection is None:
        return df
    if isinstance(push_selection, slice):
        return df.iloc[push_selection]
    if isinstance(push_selection, (int, np.integer, str)):
        keys = [push_selection]
    else:
        keys = list(push_selection)
    if not keys:
        return df.iloc[0:0]

    mask = pd.Series(False, index=df.index)
    if 'push_label' in df.columns:
        mask |= df['push_label'].isin(keys)
    if 'push_index' in df.columns:
        mask |= df['push_index'].isin(keys)
    return df.loc[mask]

def add_rot_arrow(ax: plt.Axes, center: Tuple[float, float], radius: float, magnitude: float, color = "black"):
    """
    Draw a small circular rotation indicator (arc + arrow head) at `center`. 
    Magnitude indicates rotation direction and strength. A Magnitude of 1 implies a full circle and CCW rotation. (capped between -1 and 1)
    """
    if magnitude == 0:
        return
    magnitude = min(1, max(-1, magnitude))
    magnituderad = magnitude*2*np.pi # normed to radians
    magnitudedeg = magnitude*360 # normed to deg
    cx, cy = center
    if magnitude > 0:
        arc = patches.Arc((cx, cy), width=2*radius, height=2*radius, angle=0, theta1=0, theta2=magnitudedeg, linewidth=1.2, color=color)
    else:
        arc = patches.Arc((cx, cy), width=2*radius, height=2*radius, angle=0, theta1=magnitudedeg, theta2=0, linewidth=1.2, color=color)
    ax.add_patch(arc)

    # compute arrow head location at the end of arc
    arrow_angle = magnituderad -0.5*np.pi + np.sign(magnituderad)*0.5*np.pi # rotate quarter turn depending on direction.
    hx = cx + radius * np.cos(magnituderad)
    hy = cy + radius * np.sin(magnituderad)
    # small arrow head as a triangle
    head = patches.RegularPolygon((hx, hy), numVertices=3, radius=radius * 0.12, orientation=arrow_angle, color=color)
    ax.add_patch(head)

def plot_position_and_orientation(data, contour_name="wrench"):
    """
    Plot object position and orientation on XY plane with contour outline.
    
    Args:
        data: pandas DataFrame with columns 'object_x', 'object_y', 'object_yaw_deg'
        contour_name: name of contour to overlay
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get contour
    contour = CONTOUR_LIBRARY.get(contour_name)
    if contour is None:
        print(f"Warning: contour '{contour_name}' not found")
        contour_xyz = None
    else:
        contour_xyz = contour['xyz']*0.001  # convert mm to m
    
    # Plot trajectories normalized to each push's initial pose (0,0,0).
    if 'push_label' in data.columns:
        grouped = list(data.groupby('push_label', sort=False))
    elif 'push_index' in data.columns:
        grouped = list(data.groupby('push_index', sort=False))
    else:
        grouped = [(0, data)]

    num_pushes = len(grouped)
    push_keys = [k for k, _ in grouped]
    numeric_push_keys = np.array(push_keys, dtype=float) if push_keys and np.issubdtype(np.array(push_keys).dtype, np.number) else None
    push_positions = np.linspace(0.1, 0.9, max(1, num_pushes))
    push_colors = plt.cm.viridis(push_positions)
    if numeric_push_keys is not None and numeric_push_keys.size > 1:
        push_norm = plt.Normalize(vmin=float(np.min(numeric_push_keys)), vmax=float(np.max(numeric_push_keys)))
    else:
        push_norm = plt.Normalize(vmin=0, vmax=max(1, num_pushes - 1))

    normalized_endpoint_poses = []
    desired_endpoint_poses = []
    contour_samples_actual = []
    contour_samples_desired = []
    start_plotted = False
    for i, (_, group) in enumerate(grouped):
        push_color = push_colors[i]
        x = group['object_x'].to_numpy(copy=False)
        y = group['object_y'].to_numpy(copy=False)
        yaw_deg = group['object_yaw_deg'].to_numpy(copy=False)
        if x.size == 0:
            continue

        x0, y0, yaw0_deg = x[0], y[0], yaw_deg[0]
        dx = x - x0
        dy = y - y0
        yaw0 = np.radians(yaw0_deg)
        cos0, sin0 = np.cos(yaw0), np.sin(yaw0)

        # Rotate world displacement by -yaw0 into the push-local frame.
        x_norm = cos0 * dx + sin0 * dy
        y_norm = -sin0 * dx + cos0 * dy
        yaw_norm_deg = (yaw_deg - yaw0_deg + 180.0) % 360.0 - 180.0

        contour_samples_actual.append((x_norm, y_norm, yaw_norm_deg, push_color))
        normalized_endpoint_poses.append((x_norm[0], y_norm[0], yaw_norm_deg[0], push_color))
        normalized_endpoint_poses.append((x_norm[-1], y_norm[-1], yaw_norm_deg[-1], push_color))

        ax.plot(
            x_norm, y_norm, '-', color=push_color, linewidth=1.2, markersize=3, alpha=0.2,
            label='Actual (per-push color)' if i == 0 else None
        )
        if not start_plotted:
            ax.plot(0.0, 0.0, 'o', color='k', markersize=8, label='Push Start')
            start_plotted = True
        ax.plot(x_norm[-1], y_norm[-1], 'o', color=push_color, markersize=6, alpha=0.95,
                label='Push End' if i == 0 else None)

        if {'elapsed_s', 'v_hat_body_vx', 'v_hat_body_vy', 'v_hat_body_wz'}.issubset(group.columns):
            t = group['elapsed_s'].to_numpy(copy=False)
            dt = np.diff(t, prepend=t[0])
            v_hat_vx = group['v_hat_body_vx'].to_numpy(copy=False)
            v_hat_vy = group['v_hat_body_vy'].to_numpy(copy=False)
            v_hat_wz = group['v_hat_body_wz'].to_numpy(copy=False)
            yaw_des = np.cumsum(v_hat_wz * dt)
            yaw_prev = yaw_des - (v_hat_wz * dt)
            cos_yaw = np.cos(yaw_prev)
            sin_yaw = np.sin(yaw_prev)

            x_des = np.cumsum((v_hat_vx * cos_yaw - v_hat_vy * sin_yaw) * dt)
            y_des = np.cumsum((v_hat_vx * sin_yaw + v_hat_vy * cos_yaw) * dt)
            yaw_des_deg = np.degrees(yaw_des)
            yaw_des_deg = (yaw_des_deg + 180.0) % 360.0 - 180.0

            contour_samples_desired.append((x_des, y_des, yaw_des_deg, push_color))
            desired_endpoint_poses.append((x_des[0], y_des[0], yaw_des_deg[0], push_color)) # Start
            desired_endpoint_poses.append((x_des[-1], y_des[-1], yaw_des_deg[-1], push_color)) # End

            ax.plot(
                x_des, y_des, color=push_color, linestyle='--', linewidth=1.2,
                alpha=0.3, label='Desired (same push color)' if i == 0 else None
            )

            # desired endpoints
            ax.plot(x_des[-1], y_des[-1], 'o', color=push_color, markersize=3, alpha=0.95)

    if contour_xyz is not None:
        contour_xy_real = contour_xyz[:, :2] / 50.0  # shrink to fit - real contours larger than intended contours
        contour_xy_desired = contour_xyz[:, :2] / 100.0  # shrink to fit - real contours larger than intended contours
        c_x_real = contour_xy_real[:, 0]
        c_y_real = contour_xy_real[:, 1]
        c_x_desired = contour_xy_desired[:, 0]
        c_y_desired = contour_xy_desired[:, 1]

        def _plot_contours_sampled(c_x, c_y, x_vals, y_vals, yaw_deg_vals, color, alpha, linewidth, max_samples):
            n = x_vals.size
            if n == 0:
                return
            yaw_rad_vals = np.radians(yaw_deg_vals)
            sample_count = min(max_samples, n)
            sample_idx = np.unique(np.linspace(0, n - 1, sample_count, dtype=int))
            for idx in sample_idx:
                cos_y = np.cos(yaw_rad_vals[idx])
                sin_y = np.sin(yaw_rad_vals[idx])
                rot_x = cos_y * c_x - sin_y * c_y + x_vals[idx]
                rot_y = sin_y * c_x + cos_y * c_y + y_vals[idx]
                ax.plot(rot_x, rot_y, color=color, alpha=alpha, linewidth=linewidth)

        def _plot_endpoint_contours(c_x, c_y, endpoint_poses, color, alpha, linewidth):
            for x_val, y_val, yaw_deg, pose_color in endpoint_poses:
                yaw_rad = np.radians(yaw_deg)
                cos_y = np.cos(yaw_rad)
                sin_y = np.sin(yaw_rad)
                rot_x = cos_y * c_x - sin_y * c_y + x_val
                rot_y = sin_y * c_x + cos_y * c_y + y_val
                line_color = pose_color if color is None else color
                ax.plot(rot_x, rot_y, color=line_color, alpha=alpha, linewidth=linewidth)

        for x_vals, y_vals, yaw_vals, color in contour_samples_actual:
            _plot_contours_sampled(c_x_real, c_y_real, x_vals, y_vals, yaw_vals, color=color, alpha=0.22, linewidth=0.55, max_samples=0)
        _plot_endpoint_contours(c_x_real, c_y_real, normalized_endpoint_poses, color=None, alpha=0.5, linewidth=0.9)

        # for x_vals, y_vals, yaw_vals, color in contour_samples_desired:
        #     _plot_contours_sampled(x_vals, y_vals, yaw_vals, color=color, alpha=0.15, linewidth=0.5, max_samples=12)
        _plot_endpoint_contours(c_x_desired, c_y_desired, desired_endpoint_poses, color=None, alpha=0.3, linewidth=0.8)

        # Dotted line from actual to desired endpoints for each push
        # label_made = False
        # for (x_a, y_a, _, color), (x_d, y_d, _, _) in zip(normalized_endpoint_poses, desired_endpoint_poses):
        #     ax.plot([x_a, x_d], [y_a, y_d], color=color, linestyle=':', linewidth=0.9, alpha=0.6, label='Actual to Desired Endpoint' if not label_made else None)
        #     label_made = True

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    _set_figure_title(fig, f'Object Position and Orientation ({contour_name})', data)

    sm = plt.cm.ScalarMappable(norm=push_norm, cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Push Number')
    legend = ax.legend()
    _style_axes_labels_and_legend(ax, legend)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax

def plot_push_contacts_on_contour(
    data,
    contour_name="wrench",
    push_selection=None,
    emphasize_first_last=True,
    show_wrench=True,
    show_w_hat=True,
):
    """Plot push contact segments on a contour with desired/estimated wrenches.

    The data is expected to contain per-push summary information (one row per push)
    with at least the following columns:
      - contact_p1_x, contact_p1_y, contact_p2_x, contact_p2_y: contact points in meters
      - wrench_fx/fy/tz or wrench_echo_fx/fy/tz: desired wrench in body frame
      - object_twist_body_vx/vy/wz and mu_x/mu_y/mu_rot (optional) for estimating w_hat
      - push_label or push_index to distinguish pushes

    Args:
        data: pandas DataFrame containing push summaries.
        contour_name: name of the contour to draw (must exist in CONTOUR_LIBRARY).
        push_selection: optional (int/str/list/slice) subset of pushes to include.
        emphasize_first_last: draw first/last pushes with thicker/highlighted styles.
        show_wrench: whether to draw desired wrench arrows.
        show_w_hat: whether to draw estimated wrench arrows (requires twist + mu columns).

    Returns:
        (fig, ax) tuple.
    """

    data = _filter_pushes(data, push_selection)

    contour = CONTOUR_LIBRARY.get(contour_name)
    if contour is None:
        raise ValueError(f"Unknown contour '{contour_name}'")
    contour_xy = contour['xyz'][:, :2] * 0.001  # mm -> m

    # Set appropriate figsize and arrow scales for different objects
    if contour_name == "wrench":
        figsize = (6, 10)
        arrow_width = 0.010
        norm_length = 0.02
    elif contour_name == "allen_key":
        # both jenga and allen key are wider than they are tall
        figsize = (10, 6)
        arrow_width = 0.010
        norm_length = 0.02
    elif contour_name == "jenga":
        figsize = (10, 6)
        arrow_width = 0.005
        norm_length = 0.01

    show_wrench_subplot = show_wrench or show_w_hat
    if show_wrench_subplot:
        fig, (ax, ax_wrench) = plt.subplots(
            1, 2, figsize=(figsize[0] + 4, figsize[1]),
            gridspec_kw={'width_ratios': [4, 1.6]}
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_wrench = None
    ax.plot(contour_xy[:, 0], contour_xy[:, 1], '-', color='k', alpha=0.4)

    # Determine desired wrench once (assumed constant across pushes)
    desired_wrench = None
    for cols in (
        ['wrench_echo_fx', 'wrench_echo_fy', 'wrench_echo_tz'],
        ['wrench_fx', 'wrench_fy', 'wrench_tz'],
    ):
        if set(cols).issubset(data.columns):
            desired_wrench = data[cols].iloc[0].to_numpy(dtype=float)
            break

    if show_wrench and desired_wrench is not None:
        ax_wrench.quiver(0.0, 0.0, desired_wrench[0], desired_wrench[1], angles='xy', scale_units='xy', scale=1,
                         color='tab:red', width=arrow_width*4.5, alpha=0.85, label='$\mathbf{w}_d$')

    # Determine push ordering
    if 'push_label' in data.columns:
        push_keys = pd.unique(data['push_label'])
        key_col = 'push_label'
    elif 'push_index' in data.columns:
        push_keys = np.sort(data['push_index'].unique())
        key_col = 'push_index'
    else:
        push_keys = [0]
        key_col = None

    num_pushes = len(push_keys)
    push_positions = np.linspace(0.1, 0.9, max(1, num_pushes))
    colors = plt.cm.viridis(push_positions)
    
    # Set up push normalization for colorbar
    numeric_push_keys = None
    try:
        numeric_push_keys = np.array(push_keys, dtype=float) if push_keys and np.issubdtype(np.array(push_keys).dtype, np.number) else None
    except (ValueError, TypeError):
        pass
    if numeric_push_keys is not None and numeric_push_keys.size > 1:
        push_norm = plt.Normalize(vmin=float(np.min(numeric_push_keys)), vmax=float(np.max(numeric_push_keys)))
    else:
        push_norm = plt.Normalize(vmin=0, vmax=max(1, num_pushes - 1))

    # Plot pushes and contact segments
    w_hat_torques = []  # Collect (color, torque) for rotation arrows
    for i, key in enumerate(push_keys):
        group = data[data[key_col] == key] if key_col is not None else data
        if group.empty:
            continue

        pose_row = group.iloc[0]
        row = group.iloc[-1]

        p1 = np.asarray([row.get('contact_p1_x', np.nan), row.get('contact_p1_y', np.nan)], dtype=float)
        p2 = np.asarray([row.get('contact_p2_x', np.nan), row.get('contact_p2_y', np.nan)], dtype=float)
        if np.isnan(p1).any() or np.isnan(p2).any():
            continue

        # Convert contact points from world -> body frame, using the start-of-push object pose.
        if set(['object_x', 'object_y', 'object_yaw_deg']).issubset(pose_row.index):
            x0, y0 = float(pose_row['object_x']), float(pose_row['object_y'])
            yaw0 = np.radians(float(pose_row['object_yaw_deg']))
            cos0, sin0 = np.cos(yaw0), np.sin(yaw0)
            def to_body(pt):
                dx, dy = pt[0] - x0, pt[1] - y0
                return np.array([cos0 * dx + sin0 * dy, -sin0 * dx + cos0 * dy])
            p1 = to_body(p1)
            p2 = to_body(p2)

        color = colors[i]
        is_edge = emphasize_first_last and (i == 0 or i == num_pushes - 1)
        lw = 2.5 if is_edge else 1.2
        alpha = 1.0 if is_edge else 0.7


        # Mark closest points on the contour for each contact point.
        idx1 = np.argmin(np.linalg.norm(contour_xy - p1, axis=1))
        idx2 = np.argmin(np.linalg.norm(contour_xy - p2, axis=1))
        ax.scatter(contour_xy[[idx1, idx2], 0], contour_xy[[idx1, idx2], 1],
                   c=[color, color], s=60, marker='x', linewidths=2, alpha=alpha, label='Push points' if i == 0 else None)
        pcontour1 = contour_xy[idx1]
        pcontour2 = contour_xy[idx2]
        ax.plot([pcontour1[0], pcontour2[0]], [pcontour1[1], pcontour2[1]], '-', color=color, linewidth=lw, alpha=alpha,
                label='Push chord' if i == 0 else None)
        mid = 0.5 * (pcontour1 + pcontour2)
        cmd_twist = None
        if set(['gripper_cmd_twist_body_vx', 'gripper_cmd_twist_body_vy']).issubset(row.index):
            cmd_twist = np.array([row['gripper_cmd_twist_body_vx'], row['gripper_cmd_twist_body_vy']], dtype=float)
        # elif set(['avg_object_twist_body_vx', 'avg_object_twist_body_vy']).issubset(row.index):
        #     cmd_twist = np.array([row['avg_object_twist_body_vx'], row['avg_object_twist_body_vy']], dtype=float)

        seg_dir = cmd_twist
        arrow_dir = seg_dir
        if cmd_twist is not None and np.all(np.isfinite(cmd_twist)) and np.linalg.norm(cmd_twist) > 1e-9:
            if np.dot(cmd_twist, seg_dir) < 0.0:
                arrow_dir = -seg_dir

        arrow = arrow_dir * 6.0
        ax.quiver(mid[0], mid[1], arrow[0], arrow[1], angles='xy', scale_units='xy', scale=1,
                    color=color, width=arrow_width, alpha=alpha,
                    headwidth=4.0, headlength=5.0, headaxislength=4.5, label='$\\mathbf{v}_{gripper}$' if i == 0 else None)


        # Draw estimated wrench (w_hat) from object twist + mu if available
        if show_w_hat and set(['v_hat_body_vx', 'v_hat_body_vy', 'v_hat_body_wz',
                               'mu_x', 'mu_y', 'mu_rot']).issubset(row.index):
            mu_vals = np.array([row['mu_x'], row['mu_y'], row['mu_rot']], dtype=float)
            v_meas = np.array([row['v_hat_body_vx'], row['v_hat_body_vy'],
                               row['v_hat_body_wz']], dtype=float)
            denom = np.maximum(BASE_F2V / mu_vals, 1e-12)
            w_hat = v_meas / denom
            ax_wrench.quiver(0.0, 0.0, w_hat[0], w_hat[1], angles='xy', scale_units='xy', scale=1,
                             color=color, width=arrow_width*2, alpha=0.5,
                             label='$\hat{{\mathbf{w}}}$' if i == 0 else None)
            # Collect torque for rotation arrow if above threshold
            if abs(w_hat[2]) > 0.005:
                w_hat_torques.append((color, w_hat[2]))

    _set_figure_title(fig, f"Push Contacts on Contour ({contour_name})", data)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()
    # x_step = np.min(np.diff(x_ticks)) if x_ticks.size > 1 else np.nan
    # y_step = np.min(np.diff(y_ticks)) if y_ticks.size > 1 else np.nan
    tick_step = 0.020
    if np.isfinite(tick_step) and tick_step > 0.0:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.xaxis.set_major_locator(FixedLocator(np.arange(np.floor(x_min / tick_step) * tick_step, x_max + 0.5 * tick_step, tick_step)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(np.floor(y_min / tick_step) * tick_step, y_max + 0.5 * tick_step, tick_step)))
    legend = ax.legend(loc='best')
    _style_axes_labels_and_legend(ax, legend)
    if ax_wrench is not None:
        ax_wrench.set_title('Wrenches')
        ax_wrench.text(0.97, 0.03, 'X (N)', transform=ax_wrench.transAxes, ha='right', va='bottom', fontsize=AX_LABEL_FONT_SIZE)
        ax_wrench.text(0.03, 0.97, 'Y (N)', transform=ax_wrench.transAxes, ha='left', va='top', fontsize=AX_LABEL_FONT_SIZE)

        ax_wrench.grid(True, alpha=0.3)
        ax_wrench.axhline(0.0, color='k', linewidth=0.8, alpha=0.2)
        ax_wrench.axvline(0.0, color='k', linewidth=0.8, alpha=0.2)
        wrench_legend = ax_wrench.legend(bbox_to_anchor=(0.2, 1.01), loc='lower right')
        _style_axes_labels_and_legend(ax_wrench, wrench_legend)
        # Set limits to show both w_d and w_hat clearly, with some margin
        all_wrench_vectors = []
        if desired_wrench is not None:
            all_wrench_vectors.append(desired_wrench[:2])
        if show_w_hat: 
            for _, row in data.iterrows():
                if set(['v_hat_body_vx', 'v_hat_body_vy', 'v_hat_body_wz',
                        'mu_x', 'mu_y', 'mu_rot']).issubset(row.index):
                    mu_vals = np.array([row['mu_x'], row['mu_y'], row['mu_rot']], dtype=float)
                    v_hat = np.array([row['v_hat_body_vx'], row['v_hat_body_vy'],
                                       row['v_hat_body_wz']], dtype=float)
                    denom = np.maximum(BASE_F2V / mu_vals, 1e-12)
                    w_hat = v_hat / denom
                    all_wrench_vectors.append(w_hat[:2])
        if all_wrench_vectors:
            all_wrench_vectors = np.array(all_wrench_vectors)
            max_val = np.max(np.abs(all_wrench_vectors)) * 1.1
            ax_wrench.set_xlim(-max_val, max_val)
            ax_wrench.set_ylim(-max_val, max_val)
            ax_wrench.set_aspect('equal')
            ax_wrench.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)
        # Draw rotation arrows for w_hat torques on the wrench subplot
        if w_hat_torques:
            base_radius = 1.0
            radius_increment = 0.40
            max_torque = 0.4
            for idx, (color, torque) in enumerate(w_hat_torques):
                radius = base_radius + idx * radius_increment
                # Normalize torque to [-1, 1] where 1 means max_torque
                magnitude = np.clip(torque / max_torque, -1.0, 1.0)
                add_rot_arrow(ax_wrench, (0.0, 0.0), radius, magnitude, color=color)
            # ax_wrench.text(-max_val*.95, -max_val*.95, f"Full Rotation = {max_torque} Nm", fontsize=AX_LABEL_FONT_SIZE)
        tick_step = 2.0
        if np.isfinite(tick_step) and tick_step > 0.0:
            x_min, x_max = ax_wrench.get_xlim()
            y_min, y_max = ax_wrench.get_ylim()
            ax_wrench.xaxis.set_major_locator(FixedLocator(np.arange(np.floor(x_min / tick_step) * tick_step, x_max + 0.5 * tick_step, tick_step)))
            ax_wrench.yaxis.set_major_locator(FixedLocator(np.arange(np.floor(y_min / tick_step) * tick_step, y_max + 0.5 * tick_step, tick_step)))


    # Add colorbar for push ordering
    sm = plt.cm.ScalarMappable(norm=push_norm, cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top')
    cbar.set_label('Push Number', size=AX_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=AX_LABEL_FONT_SIZE)
    # cbar.set_ticks(cbar.get_ticks(), textsize=AX_LABEL_FONT_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Adjust subplot positions to be roughly level
    if ax_wrench is not None:
        ax_pos_wrench = ax_wrench.get_position()
        ax_pos = ax.get_position()
        ax.set_position([ax_pos.x0, ax_pos.y0 +0.45, ax_pos.width, ax_pos.height])
        ax_wrench.set_position([ax_pos_wrench.x0, ax_pos_wrench.y0 -0.10, ax_pos_wrench.width, ax_pos_wrench.height])
    return fig, ax

def plot_mu_alpha_over_pushes(data):
    """
    Plot mu_x, mu_y, mu_rot, and alpha_ub over push indices.
    NaN values are replaced with 1.0.

    Args:
        data: pandas DataFrame with columns 'push_index', 'mu_x', 'mu_y', 'mu_rot', 'alpha_ub'
              optionally 'alpha_ub_closed'
    """
    # Replace NaN with 1.0
    mu_x = data['mu_x'].fillna(1.0)
    mu_y = data['mu_y'].fillna(1.0)
    mu_rot = data['mu_rot'].fillna(1.0)
    alpha_ub = data['alpha_ub'].fillna(1.0)
    # alpha_ub_closed = data['alpha_ub_closed'].fillna(1.0) if 'alpha_ub_closed' in data.columns else alpha_ub

    # Get unique push indices and take last value for each push
    push_indices = data['push_index'].unique()
    push_indices.sort()

    mu_x_by_push = [mu_x[data['push_index'] == p].iloc[-1] if len(mu_x[data['push_index'] == p]) > 0 else 1.0 for p in push_indices]
    mu_y_by_push = [mu_y[data['push_index'] == p].iloc[-1] if len(mu_y[data['push_index'] == p]) > 0 else 1.0 for p in push_indices]
    mu_rot_by_push = [mu_rot[data['push_index'] == p].iloc[-1] if len(mu_rot[data['push_index'] == p]) > 0 else 1.0 for p in push_indices]
    alpha_by_push = [alpha_ub[data['push_index'] == p].iloc[-1] if len(alpha_ub[data['push_index'] == p]) > 0 else 1.0 for p in push_indices]
    # alpha_closed_by_push = [alpha_ub_closed[data['push_index'] == p].iloc[-1] if len(alpha_ub_closed[data['push_index'] == p]) > 0 else 1.0 for p in push_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Mu values
    ax1.plot(push_indices, mu_x_by_push, 'o-', label='mu_x', markersize=4)
    ax1.plot(push_indices, mu_y_by_push, 's-', label='mu_y', markersize=4)
    ax1.plot(push_indices, mu_rot_by_push, '^-', label='mu_rot', markersize=4)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_ylabel('Coefficient of Friction (mu)')
    ax1_legend = ax1.legend()
    _style_axes_labels_and_legend(ax1, ax1_legend)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)

    # Alpha values
    ax2.plot(push_indices, alpha_by_push, 'D-', color='purple', markersize=4, label='$\alpha_{ub}$')
    # if 'alpha_ub_closed' in data.columns:
        # ax2.plot(push_indices, alpha_closed_by_push, 'x-', color='tab:orange', markersize=4, label='alpha_ub_closed')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Push Index')
    ax2.set_ylabel('Alpha Upper Bound')
    ax2_legend = ax2.legend()
    _style_axes_labels_and_legend(ax2, ax2_legend)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)

    _set_figure_title(fig, 'Friction Coefficients and Alpha Upper Bound Over Pushes', data)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, (ax1, ax2)

def plot_twist_error_with_mu_alpha_over_time(
    data,
    show_twist_error=False,
    show_mu_alpha=True,
    replace_alpha_with_mu_theta=True,
    show_wrenches=True,
    show_weighted_errors=True,
    wrench_subplot_layout='x_y_theta', # xy_theta, merged, or x_y_theta
):
    """
    Plot twist error and per-push mu/alpha values aligned on the same sequential time axis.
    """
    wrench_subplot_count = 0
    if show_wrenches:
        if wrench_subplot_layout == 'xy_theta':
            wrench_subplot_count = 2
        elif wrench_subplot_layout == 'merged':
            wrench_subplot_count = 1
        elif wrench_subplot_layout == 'x_y_theta':
            wrench_subplot_count = 3

    num_plots = int(show_twist_error) + int(show_weighted_errors) + (2 if show_mu_alpha else 0) + wrench_subplot_count

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 1.6*num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    # Match the sequential time construction used in plot_twist_error_over_time.
    sequential_t = pd.Series(index=data.index, dtype=float)
    push_boundaries = []
    group_keys = []
    offset = 0.0
    for key, group in data.groupby('push_label', sort=False):
        t = group['elapsed_s'].to_numpy(copy=False)
        t0 = t[0]
        local_t = t - t0
        sequential_t.loc[group.index] = local_t + offset
        offset += local_t[-1]
        push_boundaries.append(offset)
        group_keys.append(key)

    seq_t_values = sequential_t.to_numpy(copy=False)

    # Build per-push mu/alpha (last sample in each push) and align to push time segments.
    push_key_to_idx = data.groupby('push_label', sort=False)['push_index'].last()
    push_indices = push_key_to_idx.reindex(group_keys).to_numpy()
    per_push_columns = ['push_index', 'mu_x', 'mu_y', 'mu_rot', 'alpha_ub']
    # if 'alpha_ub_closed' in data.columns:
    #     per_push_columns.append('alpha_ub_closed')
    per_push = (
        data[per_push_columns]
        .groupby('push_index', sort=True)
        .last()
        .fillna(1.0)
    )
    aligned = per_push.reindex(push_indices).fillna(1.0)

    # Step-plot constants per push over each push's [start, end] interval.
    segment_edges = np.concatenate(([0.0], np.asarray(push_boundaries, dtype=float)))
    mu_x_vals = aligned['mu_x'].to_numpy(copy=False)
    mu_y_vals = aligned['mu_y'].to_numpy(copy=False)
    mu_rot_vals = aligned['mu_rot'].to_numpy(copy=False)
    alpha_vals = aligned['alpha_ub'].to_numpy(copy=False)
    # alpha_closed_vals = aligned['alpha_ub_closed'].to_numpy(copy=False) if 'alpha_ub_closed' in aligned.columns else None

    # Bottom: reconstructed wrench from measured/estimated twist and constant desired wrench.
    mu_vals = data[['mu_x', 'mu_y', 'mu_rot']].fillna(1.0).to_numpy(copy=False)
    denom = np.maximum(BASE_F2V.reshape(1, 3) / mu_vals, 1e-12)
    v_meas = data[['v_tilde_body_vx', 'v_tilde_body_vy', 'v_tilde_body_wz']].to_numpy(copy=False)
    v_meas_avg = data[['v_tilde_avg_body_vx', 'v_tilde_avg_body_vy', 'v_tilde_avg_body_wz']].to_numpy(copy=False)
    v_hat = data[['v_hat_body_vx', 'v_hat_body_vy', 'v_hat_body_wz']].to_numpy(copy=False)
    w_meas = v_meas / denom
    w_meas_avg = v_meas_avg / denom
    w_hat = v_hat / denom

    w_des = data[['wrench_fx', 'wrench_fy', 'wrench_tz']].to_numpy(copy=False)

    # Compute characteristic radius
    contour_name = _infer_contour_name_from_path(data.attrs.get('source_path', ''))
    if contour_name in CONTOUR_LIBRARY:
        char_rad = characteristic_radius(CONTOUR_LIBRARY[contour_name])
    else:
        char_rad = 0.01  # default 1 cm
    W = np.diag([1.0, 1.0, 1.0 / char_rad])

    # Compute weighted errors
    diff_meas_hat = w_hat - w_meas
    weighted_diff_meas_hat = (W @ diff_meas_hat.T).T
    error_meas_hat = np.linalg.norm(weighted_diff_meas_hat, axis=1)

    diff_meas_avg_hat = w_hat - w_meas_avg
    weighted_diff_meas_avg_hat = (W @ diff_meas_avg_hat.T).T
    error_meas_avg_hat = np.linalg.norm(weighted_diff_meas_avg_hat, axis=1)

    diff_des_hat = w_des - w_hat
    weighted_diff_des_hat = (W @ diff_des_hat.T).T
    error_des_hat = np.linalg.norm(weighted_diff_des_hat, axis=1)

    # Start plotting
    thick_line_thickness = 1.8

    ax_idx = 0

    # Top: twist error over sequential time.
    if show_twist_error:
        ax_top = axes[ax_idx]
        ax_top.plot(seq_t_values, data['error_twist_body_vx'], label='error_vx', linewidth=1)
        ax_top.plot(seq_t_values, data['error_twist_body_vy'], label='error_vy', linewidth=1)
        ax_top.plot(seq_t_values, 0.025 * data['error_twist_body_wz'], label='error_wz/40', linewidth=1, color='tab:red')
        for boundary in push_boundaries[:-1]:
            ax_top.axvline(boundary, color='k', linestyle='--', linewidth=1.2, alpha=0.25)
        ax_top.set_ylabel('Twist Error')
        top_legend = ax_top.legend(loc='upper right')
        _style_axes_labels_and_legend(ax_top, top_legend)
        ax_top.grid(True, alpha=0.3)
        ax_top.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)
        ax_idx += 1

    # Weighted wrench errors
    if show_weighted_errors:
        ax_weighted = axes[ax_idx]
        # ax_weighted.plot(seq_t_values, error_meas_hat, label='||W*(w_hat - w_meas)||', linewidth=1)
        ax_weighted.plot(seq_t_values, error_meas_avg_hat, label='$||W(\hat{{\mathbf{{w}}}} - \\tilde{{\mathbf{{w}}}}_{avg})||$', linewidth=thick_line_thickness)
        ax_weighted.plot(seq_t_values, error_des_hat, label='$||W(\mathbf{{w}}_d - \hat{{\mathbf{{w}}}})||$', linewidth=thick_line_thickness)
        for boundary in push_boundaries[:-1]:
            ax_weighted.axvline(boundary, color='k', linestyle='--', linewidth=1.2, alpha=0.25)
        ax_weighted.set_ylabel('Wrench Error')
        ax_weighted.set_yscale("log")
        weighted_legend = ax_weighted.legend(loc='best', ncol=2)
        _style_axes_labels_and_legend(ax_weighted, weighted_legend, log=True)
        ax_weighted.grid(True, alpha=0.3)
        ax_idx += 1

    if show_mu_alpha:
        mu_ax_idx = ax_idx
        alpha_ax_idx = ax_idx + 1
        ax_mu = axes[mu_ax_idx]
        ax_alpha = axes[alpha_ax_idx]

        # replace_alpha_with_mu_theta = False

        # Duplicate the final value so the last plateau reaches the final push boundary.
        ax_mu.step(segment_edges, np.append(mu_x_vals, mu_x_vals[-1]), where='post', label='$\mu_x$', linewidth=thick_line_thickness)
        ax_mu.step(segment_edges, np.append(mu_y_vals, mu_y_vals[-1]), where='post', label='$\mu_y$', linewidth=thick_line_thickness)
        if replace_alpha_with_mu_theta:
            ax_alpha.step(segment_edges, np.append(mu_rot_vals, mu_rot_vals[-1]), where='post', label='$\mu_{\\theta}$', linewidth=thick_line_thickness, color='tab:green')
        else:
            ax_alpha.step(segment_edges, np.append(alpha_vals, alpha_vals[-1]), where='post', label='$\\alpha_{ub}$', linewidth=thick_line_thickness, color='purple')
            ax_mu.step(segment_edges, np.append(mu_rot_vals, mu_rot_vals[-1]), where='post', label='$\mu_{\\theta}$', linewidth=thick_line_thickness)
        # if alpha_closed_vals is not None:
            # ax_alpha.step(segment_edges, np.append(alpha_closed_vals, alpha_closed_vals[-1]), where='post', label='alpha_ub_closed', linewidth=step_thickness, color='tab:orange')
        for boundary in push_boundaries[:-1]:
            ax_mu.axvline(boundary, color='k', linestyle='--', linewidth=1.2, alpha=0.25)
            ax_alpha.axvline(boundary, color='k', linestyle='--', linewidth=1.2, alpha=0.25)
        ax_mu.axhline(y=1.0, color='k', linestyle='--', alpha=0.25, linewidth=1)
        ax_alpha.axhline(y=1.0, color='k', linestyle='--', alpha=0.25, linewidth=1)
        if replace_alpha_with_mu_theta:
            ax_mu.set_ylabel('$\mu_{x, y}$')
            ax_alpha.set_ylabel('$\mu_\\theta$')
        else:
            ax_mu.set_ylabel('$\mu$')
            ax_alpha.set_ylabel('$\\alpha_{ub}$')
        ax_mu.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)
        ax_alpha.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)
        mu_legend = ax_mu.legend(loc='upper left')
        alpha_legend = ax_alpha.legend(loc='upper left')
        _style_axes_labels_and_legend(ax_mu, mu_legend)
        _style_axes_labels_and_legend(ax_alpha, alpha_legend)
        ax_mu.grid(True, alpha=0.3)
        ax_alpha.grid(True, alpha=0.3)

    ax_idx += 2 if show_mu_alpha else 0

    if show_wrenches:
        colors = ('tab:blue', 'tab:orange', 'tab:green')
        labels = ('x', 'y', '\\theta')
        if wrench_subplot_layout == 'x_y_theta':
            wrench_axes = axes[ax_idx : ax_idx + 3]
            component_groups = ((0,), (1,), (2,))
            ylabels = ('$\mathbf{w}_x$ (N)', '$\mathbf{w}_y$ (N)', '$\mathbf{w}_\\theta$ (Nm)')
            legend_cols = (3, 3, 3)
        elif wrench_subplot_layout == 'xy_theta':
            wrench_axes = axes[ax_idx : ax_idx + 2]
            component_groups = ((0, 1), (2,))
            ylabels = ('Force XY (N)', 'Torque (Nm)')
            legend_cols = (3, 3)
        elif wrench_subplot_layout == 'merged':
            wrench_axes = (axes[ax_idx],)
            component_groups = ((0, 1, 2),)
            ylabels = ('Wrench (N, N, Nm)',)
            legend_cols = (3,)
        else:
            print("Something went wrong")

        for ax_wrench, component_indices, ylabel, ncol in zip(wrench_axes, component_groups, ylabels, legend_cols):
            for i in component_indices:
                ax_wrench.plot(seq_t_values, w_meas[:, i], color=colors[i], linewidth=1.0, alpha=0.9, label=f'$\\tilde{{\mathbf{{w}}}}_{labels[i]}$')
                ax_wrench.plot(seq_t_values, w_hat[:, i], color=colors[i], linewidth=thick_line_thickness, linestyle='--', alpha=0.9, label=f'$\hat{{\mathbf{{w}}}}_{labels[i]}$')
                # ax_wrench.plot(seq_t_values, w_des[:, i], color=colors[i], linewidth=thick_line_thickness, linestyle=':', alpha=0.8, label=f'$\mathbf{{w}}_{{d{labels[i]}}}$')
            for boundary in push_boundaries[:-1]:
                ax_wrench.axvline(boundary, color='k', linestyle='--', linewidth=1.2, alpha=0.25)
            ax_wrench.set_ylabel(ylabel)
            wrench_legend = ax_wrench.legend(loc='best', ncol=ncol)
            _style_axes_labels_and_legend(ax_wrench, wrench_legend)
            ax_wrench.grid(True, alpha=0.3)
            ax_wrench.tick_params(axis='both', labelsize=AX_LABEL_FONT_SIZE)

        wrench_axes[-1].set_xlabel('Sequential Time Across Pushes (s)')
        _style_axes_labels_and_legend(wrench_axes[-1])
    if len(push_boundaries) > 0:
        total_t = push_boundaries[-1]
        axes[-1].set_xlim(0, total_t * 1.05)
        axes[-1].set_xticks(np.arange(0, total_t * 1.05, 6.0))

    # _set_figure_title(fig, 'Mu/Alpha and Wrenches Aligned to Sequential Time', data)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

def _load_and_merge_summary(timeseries_path, summary_path=None):
    """Load timeseries and (optionally) summary CSVs and merge summary columns into timeseries.

    If summary_path is not provided, it will attempt to infer it by replacing
    "_timeseries" with "_summary" in the filename.
    """
    ts = pd.read_csv(timeseries_path)
    ts.attrs['source_path'] = timeseries_path
    path_lower = os.path.basename(timeseries_path).lower()
    if 'surface_type' not in ts.columns:
        if 'sticky' in path_lower:
            ts['surface_type'] = 'sticky'
        elif 'smooth' in path_lower:
            ts['surface_type'] = 'smooth'

    if summary_path is None:
        base, ext = os.path.splitext(timeseries_path)
        summary_path_guess = base.replace("_timeseries", "_summary") + ext
        if os.path.exists(summary_path_guess):
            summary_path = summary_path_guess

    if summary_path is None or not os.path.exists(summary_path):
        return ts, None

    summary = pd.read_csv(summary_path)

    # Prefer push_label for joining when available, else use push_index.
    join_keys = []
    if 'push_label' in ts.columns and 'push_label' in summary.columns:
        join_keys.append('push_label')
    if 'push_index' in ts.columns and 'push_index' in summary.columns:
        join_keys.append('push_index')

    if not join_keys:
        # If there is no common key, return both unmerged.
        return ts, summary

    merged = ts.merge(summary, how='left', on=join_keys, suffixes=('', '_summary'))
    merged.attrs['source_path'] = timeseries_path
    return merged, summary

def _infer_contour_name_from_path(path):
    """Infer contour name from a CSV filename."""
    name = os.path.basename(path).lower()
    if "allen_key" in name:
        return "allen_key"
    if "wrench" in name:
        return "wrench"
    if "jenga" in name:
        return "jenga"
    return "unknown"

def save_plot_all_batch(csv_paths, export_prefix, summary_paths=None, push_selection=None):
    """Run plot_all sequentially for many experiments and save five figures per experiment.

    Args:
        csv_paths: sequence of timeseries CSV paths, one experiment per path.
        export_prefix: shared output prefix for saved image files.
        summary_paths: optional summary CSV path or list aligned with csv_paths.
        push_selection: optional push selector forwarded to plot_all.
    """
    csv_paths = [csv_paths] if isinstance(csv_paths, (str, bytes)) else list(csv_paths)
    if not csv_paths:
        raise ValueError("csv_paths must contain at least one CSV path")

    if summary_paths is None:
        summary_paths = [None] * len(csv_paths)
    else:
        summary_paths = [summary_paths] if isinstance(summary_paths, (str, bytes)) else list(summary_paths)
        if len(summary_paths) == 1 and len(csv_paths) > 1:
            summary_paths = summary_paths * len(csv_paths)
        if len(summary_paths) != len(csv_paths):
            raise ValueError("summary_paths must be length 1 or match csv_paths")

    figure_suffixes = (
        # "position_orientation",
        # "mu_alpha", # unused
        "twist_error_mu_alpha",
        "push_contacts_all",
        # "push_contacts_first_last",
    )

    saved_paths = []
    for csv_path, summary_path in zip(csv_paths, summary_paths):
        contour_name = _infer_contour_name_from_path(csv_path)
        figures = plot_all(
            csv_path,
            contour_name=contour_name,
            push_selection=push_selection,
            summary_path=summary_path,
        )

        experiment_name = os.path.splitext(os.path.basename(csv_path))[0]
        for fig, suffix in zip(figures, figure_suffixes):
            output_path = f"{export_prefix}{experiment_name}_{suffix}.png"
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            saved_paths.append(output_path)
            plt.close(fig)
        print(f"Saved figures for {csv_path}")
    return saved_paths

def plot_all(csv_path, contour_name="wrench", push_selection=None, separate_files=False, example_file_index=None, summary_path=None):
    """Create all plots from CSV data.

    Args:
        csv_path: path or list of paths to timeseries CSV file(s)
        contour_name: name of contour to use
        push_selection: optional push selector (int, str, list, tuple, set, or slice)
        separate_files: if True and multiple files are provided, produce one figure set per file
        example_file_index: optional file index to select one CSV from a list
        summary_path: optional path (or list of paths) to corresponding summary CSV(s). When provided,
            will be merged into the timeseries data to pull contact_point/wrench data.
    """
    csv_paths = [csv_path] if isinstance(csv_path, (str, bytes)) else list(csv_path)
    if not csv_paths:
        raise ValueError("csv_path must contain at least one CSV path")

    if example_file_index is not None:
        csv_paths = [csv_paths[int(example_file_index)]]

    if summary_path is None:
        summary_paths = [None] * len(csv_paths)
    else:
        summary_paths = [summary_path] if isinstance(summary_path, (str, bytes)) else list(summary_path)
        if len(summary_paths) == 1 and len(csv_paths) > 1:
            summary_paths = summary_paths * len(csv_paths)

    # Use shared filtering logic to support push selection by label/index.
    def _filter_pushes_local(df):
        return _filter_pushes(df, push_selection)

    if separate_files and len(csv_paths) > 1:
        outputs = []
        for path, summ_path in zip(csv_paths, summary_paths):
            data, _summary = _load_and_merge_summary(path, summ_path)
            data = _filter_pushes_local(data)
            data.attrs['source_path'] = path
            # fig1, _ = plot_position_and_orientation(data, contour_name)
            fig4, _ = plot_twist_error_with_mu_alpha_over_time(data)
            fig5, _ = plot_push_contacts_on_contour(data, contour_name, push_selection)
            # outputs.append((fig1, fig4, fig5))
            outputs.append((fig4, fig5))
        plt.tight_layout()
        return outputs

    frames = []
    for file_idx, (path, summ_path) in enumerate(zip(csv_paths, summary_paths)):
        df, _ = _load_and_merge_summary(path, summ_path)
        path_lower = os.path.basename(path).lower()
        if 'surface_type' not in df.columns:
            if 'sticky' in path_lower:
                df['surface_type'] = 'sticky'
            elif 'smooth' in path_lower:
                df['surface_type'] = 'smooth'
        if len(csv_paths) > 1:
            if 'push_label' in df.columns:
                df['push_label'] = str(file_idx) + ":" + df['push_label'].astype(str)
            elif 'push_index' in df.columns:
                df['push_label'] = str(file_idx) + ":" + df['push_index'].astype(str)
            else:
                df['push_label'] = str(file_idx) + ":0"
        frames.append(df)

    data = _filter_pushes_local(pd.concat(frames, ignore_index=True, copy=False))
    data.attrs['source_paths'] = list(csv_paths)

    # fig1, _ = plot_position_and_orientation(data, contour_name)
    # fig2, _ = plot_mu_alpha_over_pushes(data)
    fig3, _ = plot_twist_error_with_mu_alpha_over_time(data)
    fig4, _ = plot_push_contacts_on_contour(data, contour_name, push_selection)
    # fig5, _ = plot_push_contacts_on_contour(data, contour_name, [0, 7])

    plt.tight_layout()
    # return fig1, fig3, fig4, fig5
    return fig3, fig4

if __name__ == "__main__":
    prefix = "/home/maxlg/0320/CSV/"
    csv_files = [
        # Uncomment files to include in the plot:
        # "smooth_wrench_w1_timeseries.csv",
        # "smooth_wrench_w2_timeseries.csv",
        # "smooth_wrench_w3_timeseries.csv",
        # "smooth_wrench_custom_123123_timeseries.csv",
        # "smooth_allen_key_w1_timeseries.csv",
        # "smooth_allen_key_w2_timeseries.csv",
        # "smooth_allen_key_w3_timeseries.csv",
        # "smooth_allen_key_custom_123123_timeseries.csv",
        # "smooth_jenga_6_w1_timeseries.csv",
        # "smooth_jenga_6_w2_timeseries.csv",
        # "smooth_jenga_6_w3_timeseries.csv",
        # "smooth_jenga_6_custom_123123_timeseries.csv",
        # "sticky_wrench_w1_timeseries.csv",
        # "sticky_wrench_w2_timeseries.csv",
        # "sticky_wrench_w3_timeseries.csv",
        # "sticky_wrench_custom_123123_timeseries.csv",
        # "sticky_allen_key_w1_timeseries.csv",
        "sticky_allen_key_w2_timeseries.csv",
        # "sticky_allen_key_w3_timeseries.csv",
        # "sticky_allen_key_custom_123123_timeseries.csv",
        # "sticky_jenga_6_w1_timeseries.csv",
        "sticky_jenga_6_w2_timeseries.csv",
        # "sticky_jenga_6_w3_timeseries.csv",
        # "sticky_jenga_6_custom_123123_timeseries.csv",
    ]
    # apply prefix to files
    csv_files = [prefix + f for f in csv_files]

    # Example usage:
    # - Plot all pushes from all CSV files together:
    # plot_all(csv_files, contour_name="wrench")
    # - Plot only selected pushes by push index/label:
    # plot_all(csv_files, contour_name="wrench", push_selection=[0, 1, 2])
    # - Plot each CSV file separately:
    # plot_all(csv_files, contour_name="wrench", separate_files=True)
    # - Plot one CSV as an example sequence:
    contour_name = _infer_contour_name_from_path(csv_files[0])
    plot_all(csv_files, contour_name=contour_name, example_file_index=0)
    # - Export one figure set per experiment with a shared prefix:
    # save_plot_all_batch(csv_files, export_prefix=prefix + "../Plots/")
    plt.show()
