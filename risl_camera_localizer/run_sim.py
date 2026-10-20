#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
import pyswarms.single.global_best as gbest
from shapely.geometry import LineString, Polygon

from risl_camera_localizer.process_stl import CONTOUR_JENGA, CONTOUR_WRENCH, CONTOUR_ALLEN_KEY

# region Math / geometry helpers
def rot2(angle: float) -> np.ndarray:
    "Angle in radians"
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def wrap01(s: np.ndarray | float) -> np.ndarray | float:
    return np.mod(s, 1.0)

def norm_to_length(v: np.ndarray, len = 1.0, eps = 1e-12) -> np.ndarray:
    "Enlarges/reduces vector to size"
    norm = np.linalg.norm(v)
    return v / (norm + eps) * len

def cross2d_z(p: np.ndarray, f: np.ndarray) -> float:
    return float(p[0] * f[1] - p[1] * f[0])

def periodic_interp(points_xy: np.ndarray) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 boundary points.")
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    N = pts.shape[0] - 1

    def p_of_s(s_in):
        s_arr = np.asarray(s_in)
        s_wrapped = wrap01(s_arr)
        t = s_wrapped * N
        i0 = np.floor(t).astype(int) % N
        i1 = (i0 + 1) % N
        frac = t - np.floor(t)
        p0 = pts[i0]
        p1 = pts[i1]
        return p0 + np.expand_dims(frac, -1) * (p1 - p0)

    bx = lambda s: p_of_s(s)[..., 0]
    by = lambda s: p_of_s(s)[..., 1]
    return bx, by

def periodic_interp_normals(points_xy: np.ndarray) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 boundary points.")
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    N = pts.shape[0] - 1

    edges = pts[1:] - pts[:-1]
    edge_norms = np.linalg.norm(edges, axis=1, keepdims=True)
    tangents = edges / np.maximum(edge_norms, 1e-12)
    signed_area2 = np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    if signed_area2 >= 0.0:
        normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    else:
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    def n_of_s(s_in):
        s_arr = np.asarray(s_in)
        s_wrapped = wrap01(s_arr)
        t = s_wrapped * N
        i0 = np.floor(t).astype(int) % N
        return normals[i0]

    nx = lambda s: n_of_s(s)[..., 0]
    ny = lambda s: n_of_s(s)[..., 1]
    return nx, ny

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

def fmt4(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return fmt4(value.item())
        if np.issubdtype(value.dtype, np.floating):
            return np.array2string(
                value,
                formatter={"float_kind": lambda x: format(float(x), ".4g")},
                separator="  "
            )
        return np.array2string(value)
    if isinstance(value, (list, tuple)):
        open_bracket, close_bracket = ("[", "]") if isinstance(value, list) else ("(", ")")
        return open_bracket + ", ".join(fmt4(v) for v in value) + close_bracket
    if isinstance(value, np.generic):
        return fmt4(value.item())
    if isinstance(value, float):
        return format(value, ".4g")
    return str(value)

def check_approach_clear(p1, p2, polygon: Polygon, approach_length=0.1):
    """
    At each contact point, cast a ray perpendicular to the span p1->p2
    in the outward-normal direction, and check it doesn't re-enter the boundary.
    
    approach_length: how far out to check in m. Verify that boundary_pts are in m for this to be consistent.
    """
    span = p2 - p1
    span_len = np.linalg.norm(span)
    if span_len < 1e-9:
        return False
    
    # Unit perpendicular to span — two candidates, pick outward one
    perp = np.array([-span[1], span[0]]) / span_len  # 90deg CCW from span

    # Build boundary polygon once (caller should cache this)

    for point, normal_sign in [(p1, -1), (p2, -1)]:
        # The outward normal at each point may be +perp or -perp depending
        # on which side of the span the body is. Flip if needed.
        # Here we assume n1 faces +perp and n2 faces +perp — adjust to your convention.
        ray_dir = normal_sign * perp
        ray_end = point + approach_length * ray_dir
        ray = LineString([point+0.002*ray_dir, ray_end])
        
        # Ray must not pass through the interior
        if ray.intersects(polygon) and not polygon.exterior.contains(ray):
            return False

    return True
# endregion

# region Wrench optimisation
def pso_wrench_cost_python(s: float, delta: float,
                           bx: Callable[[float], float], by: Callable[[float], float], poly: Polygon, params: Params) -> Tuple[float, Dict]:
    """
    Compute cost and return debug info. Returns (cost, debug_dict). Intended for body frame.

    Weights torque cost by characteristic radius. The smaller the object, the more impactful torque is.
    """

    # unpack params
    w_d, theta_friction, f_max = params.desired_wrench, params.theta_friction, params.f_max
    char_rad, span_min, span_max = params.char_rad, params.span_min, params.span_max
    alpha_ub = params.alpha_ub
    contour = params.contour
    name = contour['name']

    debug_info = {}
    s = float(s)
    s2 = wrap01(s + float(delta))
    p1 = np.array([float(bx(s)), float(by(s))])
    p2 = np.array([float(bx(s2)), float(by(s2))])
    n1 = np.array([float(params.nx(s)), float(params.ny(s))])
    n2 = np.array([float(params.nx(s2)), float(params.ny(s2))])
    dvec = p2 - p1
    dist = np.linalg.norm(dvec)

    cost = 0.0
    normal_dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    # exceptions
    exceptions = (
        # ("wrench" in params.contour['name'] and ((0.22 < s < 0.56) or (0.22 < s2 < 0.56))),
        normal_dot < np.cos(np.deg2rad(150.0)), # High slip likelihood if normals differ by more than 150 degrees 
        not check_approach_clear(p1, p2, poly, approach_length=0.1),
        dist < span_min or dist > span_max,
        # Temporary exception for wrench - if both points are between 0.86 and 0.91, localization drops. Exclude for now.
        ("wrench" in name) and (0.85 < s < 0.92) and (0.85 < s2 < 0.92), 
        # Temporary exception for allen_key - if both points are between 0.80 and 0.15 (wrapped around), localization drops. Exclude for now.
        ("allen_key" in name) and (0.80 < s or s < 0.15) and (0.80 < s2 or s2 < 0.15), 
    )
    if any(exceptions):
        cost += 100.0

    debug_info['s'] = s
    # debug_info['delta'] = delta
    debug_info['s2'] = s2
    debug_info['p1'] = p1
    debug_info['p2'] = p2
    debug_info['n1'] = n1
    debug_info['n2'] = n2
    debug_info['normal_difference_deg'] = np.rad2deg(np.arccos(normal_dot))
    debug_info['span_dist'] = dist
    debug_info['contour_normal_1'] = n1
    debug_info['contour_normal_2'] = n2
    debug_info['contour_normal_dot'] = normal_dot

    d_unit = dvec / (dist + 1e-12)
    n_g = np.array([-d_unit[1], d_unit[0]])
    debug_info['normal_direction'] = n_g

    # two opposing friction directions per contact (symmetric)
    f1_1 = f_max * (rot2(+theta_friction) @ n_g)
    f1_2 = f_max * (rot2(-theta_friction) @ n_g)
    f2_1 = f1_1.copy() # parallel rays due to current chord-based modeling
    f2_2 = f1_2.copy()

    w11 = np.array([f1_1[0], f1_1[1], cross2d_z(p1, f1_1)]) # Ray 1, Point 1
    w12 = np.array([f1_2[0], f1_2[1], cross2d_z(p1, f1_2)]) # Ray 2, Point 1
    w21 = np.array([f2_1[0], f2_1[1], cross2d_z(p2, f2_1)]) # Ray 1, Point 2
    w22 = np.array([f2_2[0], f2_2[1], cross2d_z(p2, f2_2)]) # Ray 2, Point 2
    W = np.column_stack([w11, w12, w21, w22])  # 3x4

    debug_info['W'] = W.copy()
    debug_info['w_d'] = w_d.copy()

    # Smaller characteristic radius -> larger penalty on torque residual
    torque_weight = 1.0 / max(char_rad, 1e-6)
    W_weighted = W.copy()
    W_weighted[2, :] *= torque_weight
    w_d_vec = w_d.reshape(-1, 1).astype(float) # columnize it
    w_d_weighted = w_d_vec.copy()
    w_d_weighted[2, 0] *= torque_weight

    debug_info['torque_weight'] = float(torque_weight)
    debug_info['W_weighted'] = W_weighted.copy()
    debug_info['w_d_weighted'] = w_d_weighted.copy().reshape(1, -1) # uncolumnize for readibility

    a_pair = cp.Variable(2, nonneg=True)
    a = cp.vstack([a_pair[0], a_pair[1], a_pair[0], a_pair[1]]) # rays are parallel and co-linear with friction cone
    W_pair_weighted = np.column_stack([
        W_weighted[:, 0] + W_weighted[:, 2],
        W_weighted[:, 1] + W_weighted[:, 3],
    ]) # combine parallel rays for optimization, since they will have the same alpha due to symmetry
    obj = cp.Minimize(cp.sum_squares(W_pair_weighted @ a_pair - w_d_weighted[:, 0]))
    prob = cp.Problem(obj, [cp.sum(a_pair) <= alpha_ub])

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    except Exception as exc:
        return 1e6, {}

    if a_pair.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return 1e6, {}

    a_pair_val = a_pair.value.copy()
    a_val = np.array([[a_pair_val[0]], [a_pair_val[1]], [a_pair_val[0]], [a_pair_val[1]]])
    w_hat = W @ a_val
    w_hat_weighted = W_weighted @ a_val
    resid = w_hat_weighted - w_d_weighted.reshape(-1, 1)
    cost += float(np.sum((w_hat_weighted - w_d_weighted.reshape(-1, 1))**2))

    debug_info['a_value'] = a_val
    debug_info['w_hat_weighted'] = w_hat_weighted
    debug_info['w_hat'] = w_hat
    debug_info['residual'] = resid
    debug_info['cost'] = cost

    return cost, debug_info


# Convert STL contour (by default body frame) to world frame
def world_frame_contour(contour_dict: Dict, body_pos: np.ndarray, body_theta: float) -> np.ndarray:
    body_xyz = contour_dict['xyz'] / 1000.0  # mm -> m
    body_xy = body_xyz[:, :2]
    body_xy = body_xy + np.asarray(body_pos)[:2]
    c, s = np.cos(body_theta), np.sin(body_theta)
    rot = np.array([[c, -s], [s, c]])
    world_xy = (rot @ body_xy.T).T
    return world_xy

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

def v_hat(f2v_vec: np.ndarray, w_hat: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Model from control_allocation_node: v_hat = (f2v .* w_hat) / mu."""
    return (f2v_vec.reshape(3,) * w_hat.reshape(3,)) / mu.reshape(3,)

def adapt_friction(params: Params, w_hat: np.ndarray, vhat: np.ndarray, v_meas: np.ndarray) -> None:
    """Update mu and theta_friction based on measured twist v_meas."""
    e = vhat - v_meas
    if params.use_sigma_adaptation:
        sigma = 1.0 / np.clip(params.mu, 1e-12, None)
        J = np.zeros((3, 3))
        for i in range(3):
            sigma_f = sigma.copy(); sigma_f[i] += params.fd_eps
            sigma_b = sigma.copy(); sigma_b[i] -= params.fd_eps
            v_f = v_hat(params.mu_base, w_hat, 1.0 / sigma_f)
            v_b = v_hat(params.mu_base, w_hat, 1.0 / sigma_b)
            J[:, i] = (v_f - v_b) / (2.0 * params.fd_eps)
            # print(f"Finite diff for mu[{i}]: v_f={v_f}, v_b={v_b}, J[:, {i}]={J[:, i]}")
    else:
        J = np.zeros((3, 3))
        for i in range(3):
            mu_f = params.mu.copy(); mu_f[i] += params.fd_eps
            mu_b = params.mu.copy(); mu_b[i] -= params.fd_eps
            v_f = v_hat(params.mu_base, w_hat, mu_f)
            v_b = v_hat(params.mu_base, w_hat, mu_b)
            J[:, i] = (v_f - v_b) / (2.0 * params.fd_eps)
            # print(f"Finite diff for mu[{i}]: v_f={v_f}, v_b={v_b}, J[:, {i}]={J[:, i]}")

    print(f"Error e={fmt4(e)}, Jacobian J=\n{fmt4(J)}")
    grad = 2.0 * (J.T @ e)
    # Weight rotation component according to characteristic radius (squared for effect in e, v_hat)
    eta_vec = params.eta * np.array([1.0, 1.0, params.char_rad**2])
    print(f"Adaptation step: grad={fmt4(grad)}, eta={fmt4(eta_vec)}")
    if params.use_sigma_adaptation:
        sigma = np.clip(sigma - grad * eta_vec, 1e-1, 1e2)
        params.mu = 1.0 / sigma # recover mu from sigma. 
    else:
        # Jacobian is diagonal and negative, so this gradient step will increase mu if
        # vhat is too high, and decrease mu if vhat is too low.
        params.mu = np.clip(params.mu - grad * eta_vec, 1e-2, 10.0)
    params.theta_friction = float(np.arctan(np.mean(params.mu)))

    vhat_weighted = vhat.copy(); vhat_weighted[2] *= params.char_rad
    vmeas_weighted = v_meas.copy(); vmeas_weighted[2] *= params.char_rad
    denom = float(np.dot(vhat_weighted, vhat_weighted)) + 1e-12
    closed_alpha_ub = float(np.dot(vhat_weighted, vmeas_weighted) / denom)
    closed_alpha_ub = max(0.1, min(10.0, closed_alpha_ub))
    effective_alpha_ub = 0.5 * params.alpha_ub + 0.5 * closed_alpha_ub
    effective_alpha_ub = max(0.1, min(10.0, effective_alpha_ub))

    params.alpha_ub_closed = closed_alpha_ub
    params.alpha_ub = effective_alpha_ub

def solve_for_wrench(params: Params) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run PSO and return best solution components."""
    world_xy = world_frame_contour(params.contour, params.body_pos, params.body_theta)
    bx_world, by_world = periodic_interp(world_xy)
    params.nx, params.ny = periodic_interp_normals(world_xy)
    poly = Polygon(params.contour['xyz'][:, :2]*0.001) # for approach checking

    def objective_func(swarm):
        costs = np.array([
            pso_wrench_cost_python(s.item(), d.item(), bx_world, by_world, poly, params)[0]
            for s, d in swarm
        ])
        return costs

    optimizer = gbest.GlobalBestPSO(
        n_particles=params.pso_swarm,
        dimensions=2,
        options={"c1": 1.3, "c2": 1.3, "w": 0.6},
        bounds=(params.lb, params.ub),
    )

    best_cost, (s_opt, d_opt) = optimizer.optimize(objective_func, iters=params.pso_iters, verbose=False)
    _, debug = pso_wrench_cost_python(s_opt, d_opt, bx_world, by_world, poly, params)
    w_hat = debug.get("w_hat", np.zeros((3,1))).reshape(3,)
    vhat = v_hat(params.mu_base, w_hat, params.mu)

    # world coordinates for contact points for visualization/plotting
    p1_world = np.array([bx_world(s_opt), by_world(s_opt)])
    p2_world = np.array([bx_world((s_opt + d_opt) % 1.0), by_world((s_opt + d_opt) % 1.0)])

    return best_cost, s_opt, d_opt, w_hat, vhat, p1_world, p2_world

# Data container for params, because 12+ arguments is a lot
@dataclass
class Params:
    contour: Dict
    char_rad: float
    body_pos: np.ndarray
    theta_friction: float
    desired_wrench: np.ndarray
    f_max: float
    span_min: float
    span_max: float
    mu_base: np.ndarray
    mu: np.ndarray
    eta: float
    fd_eps: float
    use_sigma_adaptation: bool
    alpha_ub: float
    alpha_ub_closed: float
    pso_swarm: int
    pso_iters: int
    grid_n: int
    lb: np.ndarray
    ub: np.ndarray
    body_theta: float = 0.0
    nx: Callable[[float], float] | None = None
    ny: Callable[[float], float] | None = None

def run(params: Params, interactive: bool = True, save_path: str | None = None):
    # prepare contour / interpolation
    world_xy = world_frame_contour(params.contour, params.body_pos, params.body_theta)
    bx_world, by_world = periodic_interp(world_xy)
    poly = Polygon(params.contour['xyz'][:, :2]*0.001) # for approach checking
    params.nx, params.ny = periodic_interp_normals(world_xy)

    best_cost, s_opt, d_opt, w_hat, vhat, p1_world, p2_world = solve_for_wrench(params)
    print(f"Initial PSO: best_cost={best_cost:.6f}, s_opt={s_opt:.4f}, d_opt={d_opt:.4f}")

    # build cost grid for visualization
    grid_n = params.grid_n
    s_grid = np.linspace(params.lb[0], params.ub[0], grid_n)
    d_grid = np.linspace(params.lb[1], params.ub[1], grid_n)
    S, D = np.meshgrid(s_grid, d_grid, indexing='ij')
    cost_grid = np.empty_like(S)
    k = 0
    for i in range(grid_n):
        for j in range(grid_n):
            cost_grid[i, j], _ = pso_wrench_cost_python(
                S[i, j], D[i, j], bx_world, by_world, poly, params)
            k += 1
            print(f"Calculated {k} out of {grid_n**2} Entries", end='\r')

    # region plotting
    # Cost Map
    fig, (ax_cost, ax_object) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    S, D = np.meshgrid(s_grid, d_grid, indexing='ij')
    cs = ax_cost.contourf(S, D, np.log10(cost_grid), levels=50, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax_cost)
    cbar.set_label("Log-10 Cost (squared wrench error)")
    ax_cost.plot(s_opt, d_opt, "ro", markersize=10, label="best")
    ax_cost.set_xlabel("$s$  (first contact fraction)")
    ax_cost.set_ylabel("$\delta$  (span along contour)")
    ax_cost.set_title("PSO cost landscape")
    ax_cost.legend(loc="upper right")

    # Contour panel
    # Scale distances to forces
    # scale = 1.0/params.mu_base[0]
    scale = 80.0

    body_patch = patches.Polygon(scale*world_xy, closed=True, facecolor="lightgray", edgecolor="k", alpha=0.6)
    ax_object.add_patch(body_patch)

    p1_world = scale*np.array([bx_world(s_opt), by_world(s_opt)])
    p2_world = scale*np.array([bx_world((s_opt + d_opt) % 1.0), by_world((s_opt + d_opt) % 1.0)])
    p1_plot, = ax_object.plot(p1_world[0], p1_world[1], "ro", zorder=5) #, label="p1")
    p2_plot, = ax_object.plot(p2_world[0], p2_world[1], "go", zorder=5) #, label="p2")

    w_d_body = params.desired_wrench # Body frame

    # w_d_body_norm = norm_to_length(w_d_body[:2], len=0.05)
    w_d_body_norm = w_d_body[:2]
    ax_object.quiver(0, 0, w_d_body_norm[0], w_d_body_norm[1], angles="xy", scale_units="xy", scale=1, color="blue",
                   label="$\mathbf{w}_d$", width=0.015)

    _, dbg = pso_wrench_cost_python(s_opt, d_opt, bx_world, by_world, poly, params)
    w_hat_world = dbg.get("w_hat", np.zeros(3))
    w_hat_norm = norm_to_length(w_hat_world[:2], len=0.05)
    # w_hat_quiver = ax_object.quiver(0, 0, w_hat_norm[0], w_hat_norm[1], angles="xy", scale_units="xy", scale=1,
    #                               color="orange", label="produced wrench", width=0.01)

    v_lin = params.mu_base[0:2] * w_hat_world[0:2].flatten()
    omega = params.mu_base[2] * w_hat_world[2].flatten()
    v_g_world = np.array([v_lin[0], v_lin[1], omega[0]])
    v_g_norm = norm_to_length(v_g_world[:2], len=0.05)
    midx, midy = 0.5 * (p1_world[0] + p2_world[0]), 0.5 * (p1_world[1] + p2_world[1])
    # v_gripper_quiver = ax_object.quiver(midx, midy, v_g_norm[0], v_g_norm[1], angles="xy", scale_units="xy", scale=1,
    #                             color="green", label="gripper twist", zorder=5, width=0.01)

    info_box_cost = ax_cost.text(0.02, 0.98, "", transform=ax_cost.transAxes, verticalalignment="top",
                                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    # info_text_body = ax_object.text(0.02, 0.98, "", transform=ax_object.transAxes, verticalalignment="top",
    #                               bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))

    # ax_object.set_xlabel("x (m)")
    # ax_object.set_ylabel("y (m)")
    ax_object.text(0.97, 0.03, 'X (N)', transform=ax_object.transAxes, ha='right', va='bottom', fontsize=16)
    ax_object.text(0.03, 0.97, 'Y (N)', transform=ax_object.transAxes, ha='left', va='top', fontsize=16)
    # ax_object.set_title(f"Visualisation for desired body-frame wrench [{w_d_body[0]:.2f}, {w_d_body[1]:.2f}, {w_d_body[2]:.2f}]")
    ax_object.set_ylim([-4, 4])
    ax_object.set_xlim([-7.5, 9])
    ax_object.set_aspect('equal')
    # ax_object.axis("equal")
    # ax_object.legend(loc="upper right", fontsize=16)
    ax_object.tick_params(axis='both', labelsize=16)
    ax_object.grid(True, alpha=0.3)

    # x_ticks = ax_object.get_xticks()
    # y_ticks = ax_object.get_yticks()
    # x_step = np.min(np.diff(x_ticks)) if x_ticks.size > 1 else np.nan
    # y_step = np.min(np.diff(y_ticks)) if y_ticks.size > 1 else np.nan
    tick_step = 2.0
    if np.isfinite(tick_step) and tick_step > 0.0:
        x_min, x_max = ax_object.get_xlim()
        y_min, y_max = ax_object.get_ylim()
        ax_object.xaxis.set_major_locator(FixedLocator(np.arange(np.floor(x_min / tick_step) * tick_step, x_max + 0.5 * tick_step, tick_step)))
        ax_object.yaxis.set_major_locator(FixedLocator(np.arange(np.floor(y_min / tick_step) * tick_step, y_max + 0.5 * tick_step, tick_step)))


    clicked_point_plot, = ax_cost.plot([], [], "go", markersize=8, label="selected")


    def update_body_plot(s_val: float, d_val: float):
        p1 = scale*np.array([bx_world(s_val), by_world(s_val)])
        p2 = scale*np.array([bx_world((s_val + d_val) % 1.0), by_world((s_val + d_val) % 1.0)])
        _, debug = pso_wrench_cost_python(s_val, d_val, bx_world, by_world, poly, params)
        w_hat_wrld = debug["w_hat"]
        v_lin_local = params.mu_base[0:2] * w_hat_wrld[0:2].flatten()
        omega_local = params.mu_base[2] * w_hat_wrld[2].flatten()
        v_gripper_wrld = np.array([v_lin_local[0], v_lin_local[1], omega_local[0]])

        p1_plot.set_data([p1[0]], [p1[1]])
        p2_plot.set_data([p2[0]], [p2[1]])
        # w_hat_quiver.set_UVC(*norm_to_length(w_hat_wrld[:2], len=0.05))
        # v_gripper_quiver.set_UVC(*norm_to_length(v_gripper_wrld[:2], len=0.05))
        # v_gripper_quiver.set_offsets([0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])])
        midx, midy = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        for art in list(ax_object.patches):
            if isinstance(art, mpl.patches.Arc) or isinstance(art, mpl.patches.RegularPolygon):
                art.remove()
        # Remove old friction ray lines if any
        for line in list(getattr(ax_object, '_friction_ray_lines', [])):
            line.remove()
        ax_object._friction_ray_lines = []

        # 0.20 N is a full rotation
        add_rot_arrow(ax_object, (0, 0), 2, 5*w_d_body[2], color="blue")
        # add_rot_arrow(ax_object, (0, 0), 0.02, 10*w_hat_wrld[2][0], color="orange")
        # add_rot_arrow(ax_object, (midx, midy), 0.01, 100*v_gripper_wrld[2], color="green")

        # Draw decomposed friction rays from p1 and p2
        a_val = np.asarray(debug.get('a_value', np.zeros((4, 1)))).flatten()
        if a_val.size >= 4:
            d_span = p2 - p1
            d_span_norm = np.linalg.norm(d_span)
            if d_span_norm > 1e-9:
                d_unit = d_span / d_span_norm
                # normal to span is 90deg CCW
                n_span = np.array([-d_unit[1], d_unit[0]])
                ray_dir_plus = rot2(params.theta_friction) @ n_span
                ray_dir_minus = rot2(-params.theta_friction) @ n_span
                base_scale = params.f_max

                for pt, a_idx in [(p1, 0), (p2, 2)]:
                    for sign, ray_dir, a_idx_offset, color in [
                        (+1, ray_dir_plus, 0, 'magenta'),
                        (-1, ray_dir_minus, 1, 'darkorange')
                    ]:
                        magnitude = a_val[a_idx + a_idx_offset]
                        length = base_scale * float(np.abs(magnitude))
                        end_pt = pt + (ray_dir if sign > 0 else ray_dir) * length
                        line, = ax_object.plot([pt[0], end_pt[0]], [pt[1], end_pt[1]], color=color, linewidth=2, alpha=0.8)
                        ax_object._friction_ray_lines.append(line)

        # info_text_body.set_text(f"$s$={fmt4(s_val)}, $\delta$={fmt4(d_val)}\n$\hat{{w}}$={fmt4(w_hat_wrld.T)}\n$v_{{g}}$={fmt4(v_gripper_wrld.T)}\nBest Cost={fmt4(best_cost)}")
        fig.canvas.draw_idle()
        print(f"\n\nSOLUTION STATS:                     ")
        for dbk, dbv in debug.items():
            if dbk in ["a_value", "w_hat", "w_hat_weighted", "residual"]:
                print(dbk, fmt4(dbv.T)) # make into row vectors for readability
            else:
                print(dbk, fmt4(dbv))
    # endregion

    def handle_interaction(event):
        if event.inaxes != ax_cost:
            return
        sx, dx = event.xdata, event.ydata
        if sx is None or dx is None:
            return
        clicked_point_plot.set_data([sx], [dx])
        update_body_plot(sx, dx)
        cost_clicked, _ = pso_wrench_cost_python(sx, dx, bx_world, by_world, poly, params)
        info_box_cost.set_text(
            f"Clicked (s,d) = ({fmt4(sx)}, {fmt4(dx)})\nCost at click = {fmt4(cost_clicked)}\n"
            f"Best cost = {fmt4(best_cost)}\nBest (s,d) = ({fmt4(s_opt)}, {fmt4(d_opt)})"
        )
        fig.canvas.draw_idle()

    def onclick(event):
        handle_interaction(event)
    
    def onmove(event):
        if event.button is None: # Mouse button must be held down
            return
        handle_interaction(event)

    if interactive:
        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("motion_notify_event", onmove)
        update_body_plot(s_opt, d_opt)
        plt.show()
    else:
        update_body_plot(s_opt, d_opt)
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

def run_adaptation_loop(params: Params):
    """Run the friction adaptation loop using user-input v_meas."""
    print("\n--- Entering friction adaptation loop. Type 'q' to quit. ---")

    while True:
        best_cost, s_opt, d_opt, w_hat, vhat, p1_world, p2_world = solve_for_wrench(params)
        print(f"PSO result: cost={best_cost:.6f}, s={s_opt:.4f}, d={d_opt:.4f}")
        print(f"  w_hat={fmt4(w_hat)}, vhat={fmt4(vhat)}")

        raw = input("Enter v_meas in body frame as 'vx vy wz' (or 'q'): ").strip()
        if raw.lower() in ["q", "quit", "exit"]:
            print("Exiting adaptation loop.")
            break

        parts = raw.split()
        if len(parts) != 3:
            print("Invalid input, please enter three numbers.")
            continue

        try:
            v_meas = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
        except ValueError:
            print("Invalid numeric values; try again.")
            continue

        adapt_friction(params, w_hat, vhat, v_meas)
        print(f"Updated mu={fmt4(params.mu)}, theta_friction={fmt4(params.theta_friction)}, alpha_ub={fmt4(params.alpha_ub)}, alpha_ub_closed={fmt4(params.alpha_ub_closed)}")

    print("Adaptation loop finished.")

# --- Entrypoint -------------------------------------------------------
def main():
    sequence_number = 9 # 0, 1, 2 for different wrench sequences (see below)
    contour_dict = CONTOUR_ALLEN_KEY
    # contour_dict = CONTOUR_WRENCH
    # contour_dict = CONTOUR_JENGA
    name = contour_dict['name']
    char_rad = characteristic_radius(contour_dict)
    print(f"Characteristic radius: {char_rad:.4f} m")

    if name == "wrench":
        base_wrench_sequence = [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.2), (3.0, 1.0, 0.0)]
    elif name == "allen_key":
        base_wrench_sequence = [(0.0, 4.0, 0.0), (0.0, 4.0, -0.2), (4.0, 1.0, -0.05), # 0-2
                                (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 0.2), #3-6
                                (2.0, 4.0, 0.2), (-4.0, -2.0, 0.15), (1.0, 3.0, -0.1)] # 7-9
    elif "jenga" in name:
        # Torques reduced in accordance to smaller body size
        base_wrench_sequence = [(3.0, 1.0, 0.0), (0.0, 4.0, 0.1), (1.0, -4.0, 0.1)]

    # Replicating special cases found in experiments (uncommented to use)
    # Wrench sequence 0 finding optimum in weird place
    # theta_friction = 0.9645
    # mu = np.array([2.33, 1.0, 1.0], dtype=float)
    # alpha_ub = 0.4196
    # alpha_ub_closed = 0.4196

    # Allen key sequence 0 appearing to work well
    # theta_friction = 0.9159
    # mu = np.array([1.0, 1.44, 1.458], dtype=float)
    # alpha_ub = 0.988
    # alpha_ub_closed = 0.988

    # Standard
    theta_friction = 0.5
    mu = np.array([1.0, 1.0, 1.0], dtype=float)
    alpha_ub = 1.0
    alpha_ub_closed = 1.0

    # Everything in body frame; no need to transform
    params = Params(
        contour=contour_dict,
        char_rad=char_rad,
        body_pos=np.array([0.0, 0.0, 0.0]),
        theta_friction=theta_friction,
        desired_wrench=np.array([base_wrench_sequence[sequence_number]]).flatten(),
        f_max=5.0,
        span_min=0.01,
        span_max=0.10,
        mu_base=np.array([0.001691, 0.001691, np.deg2rad(5.234536)]),
        mu=mu.copy(),
        eta=1e4,
        fd_eps=1e-4,
        use_sigma_adaptation=True,
        alpha_ub=alpha_ub,
        alpha_ub_closed=alpha_ub_closed,
        pso_swarm=20,
        pso_iters=20,
        grid_n=31, # cosmetic choice for resolution of cost grid; higher is more expensive (grid_n^2 cost evaluations)
        lb=np.array([0.0, 0.0]), # [first point, span]
        ub=np.array([1.0, 0.2]), # [first point, span]
    )

    # mode = input("Select mode: [1] interactive plot (default), [2] adaptation loop, [3] batch save: ").strip()
    mode = "1"
    if mode == "2":
        run_adaptation_loop(params)
    elif mode == "3":
        n = int(input("Enter number of runs: "))
        for i in range(n):
            print(f"Running batch {i+1}/{n}")
            run(params, interactive=False, save_path=f"run_{i}.png")
        print(f"Saved {n} figures as run_0.png to run_{n-1}.png")
    else:
        run(params)


if __name__ == "__main__":
    main()
