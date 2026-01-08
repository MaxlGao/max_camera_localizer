import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
from scipy.interpolate import splprep, splev, interp1d
from ament_index_python.packages import get_package_share_directory
import csv

def trirotmat(angledeg, direction, point):
    return trimesh.transformations.rotation_matrix(angle=np.radians(angledeg), direction=direction, point=point)

pkg_dir = get_package_share_directory("max_camera_localizer")
stl_path_key = os.path.join(pkg_dir, "STL", "Allen Key.STL")
stl_path_wrench = os.path.join(pkg_dir, "STL", "Wrench.STL")

mesh_key = trimesh.load_mesh(stl_path_key)
origin_key = np.array([28.33, 0, 37]) # STL file doesn't preserve origin... probably should refactor other origin-setting.
mesh_key.apply_translation(-origin_key)
mesh_key.apply_transform(trirotmat(90, [1,0,0], [0,0,0]))

mesh_wrench = trimesh.load_mesh(stl_path_wrench)
origin_wrench = np.array([49, 0, 26])
mesh_wrench.apply_translation(-origin_wrench)
mesh_wrench.apply_transform(trirotmat(90, [1,0,0], [0,0,0]))
mesh_wrench.apply_transform(trirotmat(-90, [0,0,-1], [0,0,0])) # Rotate to match origin definition in object_frame_definitions. Much ad hoc.

# No STL needed for the jenga block
mesh_jenga = trimesh.creation.box(extents=[72, 24, 14])
mesh_jenga.apply_translation([0, 0, -7])


def get_path(mesh):
    direction = np.array([0, 0, 1])  # z up/down
    direction = direction / np.linalg.norm(direction)

    # Get the centroid of the mesh
    plane_z = mesh.center_mass[2]
    plane_origin = [0, 0, plane_z]

    # Intersect mesh with all planes
    all_paths, to3D, _ = trimesh.intersections.mesh_multiplane(mesh, 
                                                    plane_origin=plane_origin,
                                                    plane_normal=direction,
                                                    heights=[0])
    return all_paths[0], to3D[0], plane_z

def group_segments_into_loops(segments, tol=1e-5):
    used = np.zeros(len(segments), dtype=bool)
    loops = []

    def next_loop():
        for i, seg in enumerate(segments):
            if not used[i]:
                return i
        return None

    while (start := next_loop()) is not None:
        loop_segments = []
        queue = [start]
        used[start] = True
        loop_segments.append(segments[start])

        # build connected loop
        end_pt = segments[start][1]
        while True:
            found = False
            for i, seg in enumerate(segments):
                if used[i]:
                    continue
                if np.allclose(seg[0], end_pt, atol=tol):
                    loop_segments.append(seg)
                    end_pt = seg[1]
                    used[i] = True
                    found = True
                    break
                elif np.allclose(seg[1], end_pt, atol=tol):
                    loop_segments.append(seg[::-1])  # reverse
                    end_pt = seg[0]
                    used[i] = True
                    found = True
                    break
            if not found:
                break
            if np.allclose(loop_segments[0][0], end_pt, atol=tol):
                break  # closed
        loops.append(np.array(loop_segments))
    return loops

def segments_to_polyline(loop_segments):
    polyline = [loop_segments[0][0], loop_segments[0][1]]
    for seg in loop_segments[1:]:
        if np.allclose(seg[0], polyline[-1]):
            polyline.append(seg[1])
        elif np.allclose(seg[1], polyline[-1]):
            polyline.append(seg[0])
        else:
            raise ValueError("Segment discontinuity")
    return np.array(polyline)

def resample_polyline_equally(polyline, n_points=200):
    # Step 1: Compute arc lengths
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)

    # Step 2: Interpolation functions
    fx = interp1d(cumulative, polyline[:, 0])
    fy = interp1d(cumulative, polyline[:, 1])

    # Step 3: Evenly spaced distances
    total_length = cumulative[-1]
    new_lengths = np.linspace(0, total_length, n_points)

    # Step 4: Sample new points
    x_new = fx(new_lengths)
    y_new = fy(new_lengths)
    return np.stack((x_new, y_new), axis=1)

def polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def is_contained(inner, outer):
    # Use center point of inner
    centroid = np.mean(inner, axis=0)
    return Path(outer).contains_point(centroid)

def fit_spline_arclength_weighted(polyline, s=1.0, per=1, pow=1):
    # Compute arc-length-based parameterization
    dists = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    u = np.zeros(len(polyline))
    u[1:] = np.cumsum(dists**pow)
    # u[1:] = dists
    u /= u[-1]  # Normalize to [0, 1]

    # Fit spline using manual parameterization
    x, y = polyline[:, 0], polyline[:, 1]
    tck, _ = splprep([x, y], u=u, s=s, per=per)
    return tck

def resample_spline_by_arclength(tck, n_points=1000):
    u_dense = np.linspace(0, 1, 5000)
    x_dense, y_dense = splev(u_dense, tck)
    coords_dense = np.stack((x_dense, y_dense), axis=1)

    deltas = np.diff(coords_dense, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.insert(np.cumsum(dists), 0, 0)
    total_length = arc_lengths[-1]
    arc_lengths /= total_length

    u_interp = interp1d(arc_lengths, u_dense)
    u_equal = u_interp(np.linspace(0, 1, n_points))

    x_eq, y_eq = splev(u_equal, tck)
    return np.stack((x_eq, y_eq), axis=1), u_equal, total_length

def compute_curvature(tck, u_eval):
    """
    Compute curvature κ over the spline defined by tck and u_eval.
    Units: curvature is in 1/mm (inverse millimeters), assuming (x, y) in mm.
    """
    dx, dy = splev(u_eval, tck, der=1)
    ddx, ddy = splev(u_eval, tck, der=2)

    # Curvature κ = (x'·y'' - y'·x'') / (x'^2 + y'^2)^(3/2)
    num = dx * ddy - dy * ddx
    denom = (dx**2 + dy**2)**1.5
    kappa = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-8)

    # Arc length (normalized)
    delta = np.sqrt(dx**2 + dy**2)
    arc_length = np.cumsum(delta)
    arc_length /= arc_length[-1]

    return arc_length, kappa

def compute_normals(tck, u_eval, polygon_for_orientation_check=None):
    """
    Compute outward-pointing normal vectors along the spline.
    
    Args:
        tck: The spline representation.
        u_eval: Parameter values at which to evaluate the spline.
        polygon_for_orientation_check: Optional (n,2) array of polygon points for winding check.

    Returns:
        normals: Array of shape (n_points, 2) with outward-pointing unit normal vectors.
    """
    dx, dy = splev(u_eval, tck, der=1)
    tangents = np.stack((dx, dy), axis=1)
    
    # Rotate tangent vector 90 degrees counter-clockwise to get normal
    normals = np.stack((-dy, dx), axis=1)

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Determine winding: positive area -> CCW, negative -> CW
    if polygon_for_orientation_check is not None:
        area = np.sum(
            polygon_for_orientation_check[:, 0] * np.roll(polygon_for_orientation_check[:, 1], -1)
            - polygon_for_orientation_check[:, 1] * np.roll(polygon_for_orientation_check[:, 0], -1)
        ) * 0.5
        if area > 0:
            normals *= -1  # flip normals if polygon is CW

    return normals

def get_curve(mesh, ax_geom=None, ax_curv=None, reverse=False):
    path, to3D, height = get_path(mesh)
    all_loops = group_segments_into_loops(path)
    polylines = [segments_to_polyline(loop) for loop in all_loops if len(loop) > 2]

    # Filter only closed and valid loops
    polylines = [pline for pline in polylines if np.allclose(pline[0], pline[-1], atol=1e-5)]

    # Compute containment relationships
    contained_by = [set() for _ in polylines]
    for i, outer in enumerate(polylines):
        for j, inner in enumerate(polylines):
            if i != j and is_contained(inner, outer):
                contained_by[j].add(i)

    outermost_indices = [i for i, container in enumerate(contained_by) if len(container) == 0]
    outermost_loop = max(outermost_indices, key=lambda i: polygon_area(polylines[i]))

    # Resample and fit
    resampled = resample_polyline_equally(polylines[outermost_loop], n_points=1000)
    if reverse:
        resampled = resampled[::-1]

    # Fit spline
    tck = fit_spline_arclength_weighted(resampled, s=50.0, per=1)
    resampled_spline, u_equal, total_length = resample_spline_by_arclength(tck, n_points=1000)
    x_smooth, y_smooth = resampled_spline[:, 0], resampled_spline[:, 1]
    normals = compute_normals(tck, u_equal, resampled)

    # Verify total length and segment length agree
    # print(total_length)
    # coords = np.stack((x_smooth, y_smooth), axis=1)
    # segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    # plt.figure()
    # plt.plot(segment_lengths, '.-')
    # plt.title("Distance Between Consecutive Points")
    # plt.xlabel("Segment Index")
    # plt.ylabel("Length")
    # plt.grid(True)
    # plt.show()

    arrow_dx = x_smooth[1] - x_smooth[0]
    arrow_dy = y_smooth[1] - y_smooth[0]

    if ax_geom is not None:
        ax_geom.scatter(resampled[:, 0], resampled[:, 1], alpha=0.3)
        ax_geom.plot(x_smooth, y_smooth, 'r.')
        ax_geom.plot(x_smooth[0], y_smooth[0], 'o', color='green', markersize=8, label='Start')
        ax_geom.arrow(x_smooth[0], y_smooth[0], 50*arrow_dx, 50*arrow_dy,
                    head_width=2.0, head_length=3.0, fc='green', ec='green', length_includes_head=True)
        # Draw original polylines
        for pline in polylines:
            ax_geom.plot(pline[:, 0], pline[:, 1], 'k--', alpha=0.3)
        
        skip = 25  # spacing for arrows
        ax_geom.quiver(
            x_smooth[::skip], y_smooth[::skip],
            normals[::skip, 0], normals[::skip, 1],
            angles='xy', scale_units='xy', color='blue'#, width=0.1, scale=1
        )

        ax_geom.set_aspect('equal')
        ax_geom.set_title("Smoothed Contour")

    arc_length, kappa = compute_curvature(tck, u_equal)

    if ax_curv is not None:
        # Curvature plot
        ax_curv.plot(arc_length, kappa, label='Curvature (1/m)')
        ax_curv.set_xlabel("Normalized Arc Length")
        ax_curv.set_ylabel("Curvature κ")
        ax_curv.set_title("Curvature vs Arc Length")
        ax_curv.grid(True)
    
    return x_smooth, y_smooth, kappa, height, total_length, normals

# Contours are:
    # 'xyz': np.array of shape (N, 3), rows in mm
    # 'normals': np.array of shape (N, 3), unit vector
    # 'kappa': np.array of shape (N, ), curvature in 1/m
    # 'length': float, total length of contour in mm

x_key, y_key, k_key, z_key, l_key, n_key = get_curve(mesh_key)
CONTOUR_ALLEN_KEY = {
    'xyz': np.column_stack((x_key, y_key, [z_key]*len(x_key))),
    'normals': np.column_stack((n_key, np.zeros(len(n_key)))),  # pad Z = 0
    'kappa': k_key,
    'length': l_key
}

x_wrench, y_wrench, k_wrench, z_wrench, l_wrench, n_wrench = get_curve(mesh_wrench)
CONTOUR_WRENCH = {
    'xyz': np.column_stack((x_wrench, y_wrench, [z_wrench]*len(x_wrench))),
    'normals': np.column_stack((n_wrench, np.zeros(len(n_wrench)))),
    'kappa': k_wrench,
    'length': l_wrench
}

x_jenga, y_jenga, k_jenga, z_jenga, l_jenga, n_jenga = get_curve(mesh_jenga, reverse=True)
CONTOUR_JENGA = {
    'xyz': np.column_stack((x_jenga, y_jenga, [z_jenga]*len(x_jenga))),
    'normals': np.column_stack((n_jenga, np.zeros(len(n_jenga)))),
    'kappa': k_jenga,
    'length': l_jenga
}

def export_contour_wrench_to_csv(contour_wrench, filename):
    xyz = contour_wrench['xyz']       # shape: (N, 3)
    normals = contour_wrench['normals']  # shape: (N, 3)
    kappa = contour_wrench['kappa']   # shape: (N,)

    # Check that all arrays have the same length
    N = len(kappa)
    assert all(len(arr) == N for arr in [xyz, normals, kappa]), "Mismatched array lengths"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['x', 'y', 'z', 'nx', 'ny', 'nz', 'kappa'])

        # Write data rows
        for i in range(N):
            row = list(xyz[i]) + list(normals[i]) + [kappa[i]]
            writer.writerow(row)

if __name__ == "__main__":
    # If running this script directly, show plots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    get_curve(mesh_key, axs[0, 1], axs[0, 0])
    get_curve(mesh_wrench, axs[1, 1], axs[1, 0])
    get_curve(mesh_jenga, axs[2, 1], axs[2, 0], reverse=True)

    plt.tight_layout()
    plt.show()
    export_contour_wrench_to_csv(CONTOUR_WRENCH, "wrench_contour.csv")