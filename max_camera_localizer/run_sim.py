#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
import pyswarms.single.global_best as gbest  # type: ignore

# External constants imported from your project (unchanged)
from max_camera_localizer.process_stl import CONTOUR_WRENCH  # type: ignore

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

# endregion

# region Wrench optimisation
def pso_wrench_cost_python(s: float, delta: float,
                           bx: Callable[[float], float], by: Callable[[float], float],
                           w_d: np.ndarray, theta_friction: float,
                           f_max: float, torque_weight: float,
                           span_min: float, span_max: float) -> Tuple[float, Dict]:
    """
    Compute cost and return debug info. Returns (cost, debug_dict).
    Works with body frame or world frame optimization, but make sure that all of these are consistent:
    - The object's contours
    - The desired wrench
    """
    debug_info = {}
    s = float(s)
    s2 = wrap01(s + float(delta))
    p1 = np.array([float(bx(s)), float(by(s))])
    p2 = np.array([float(bx(s2)), float(by(s2))])
    dvec = p2 - p1
    dist = np.linalg.norm(dvec)
    # if dist < 1e-8:
    #     return 1e6, {} # This tends to mess up the cost landscape plot

    debug_info['s'] = s
    # debug_info['delta'] = delta
    debug_info['s2'] = s2
    debug_info['p1'] = p1
    debug_info['p2'] = p2
    debug_info['span_dist'] = dist

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

    a_unit = cp.Variable(nonneg=True)
    a = cp.vstack([a_unit, a_unit, a_unit, a_unit])
    obj = cp.Minimize(cp.sum_squares(W @ a - w_d.reshape(-1, 1)))
    A = np.array([[1., 1., 0., 0.],
                  [0., 0., 1., 1.]])
    b = np.array([1., 1.])
    prob = cp.Problem(obj, [A @ a <= b]) # First two 'a' sum up to first 'b', vise versa for second set

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    except Exception as exc:
        return 1e6, {}

    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return 1e6, {}

    a_val = a.value.copy()
    w_hat = W @ a.value
    resid = w_hat - w_d.reshape(-1, 1)
    cost = float(np.sum((w_hat - w_d.reshape(-1, 1))**2))

    debug_info['a_value'] = a_val
    debug_info['w_hat'] = w_hat
    debug_info['residual'] = resid
    debug_info['cost'] = cost


    return cost, debug_info


# Convert STL contour to world frame
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

# Data container for params, because 12+ arguments is a lot
@dataclass
class Params:
    contour: Dict
    char_rad: float
    body_pos: np.ndarray
    body_theta: float
    theta_friction: float
    desired_wrench_world: np.ndarray
    f_max: float
    lam: float
    lam_min: float
    lam_max: float
    mu_base: np.ndarray
    pso_swarm: int
    pso_iters: int
    grid_n: int
    lb: np.ndarray
    ub: np.ndarray

def run(params: Params):
    # prepare contour / interpolation
    world_xy = world_frame_contour(params.contour, params.body_pos, params.body_theta)
    bx_world, by_world = periodic_interp(world_xy)

    # small objective wrapper for pyswarms
    def objective_func(swarm):
        costs = np.array([
            pso_wrench_cost_python(s.item(), d.item(), bx_world, by_world,
                                   params.desired_wrench_world, params.theta_friction,
                                   params.f_max, params.lam, params.lam_min, params.lam_max)[0]
            for s, d in swarm
        ])
        return costs


    optimizer = gbest.GlobalBestPSO(
        n_particles=params.pso_swarm,
        dimensions=2,
        options={"c1": 1.3, "c2": 1.3, "w": 0.6},
        bounds=(params.lb, params.ub),
    )

    best_cost, (s_opt, d_opt) = optimizer.optimize(objective_func, iters=params.pso_iters, verbose=True)

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
                S[i, j], D[i, j], bx_world, by_world,
                params.desired_wrench_world, params.theta_friction,
                params.f_max, params.lam, params.lam_min, params.lam_max)
            k += 1
            print(f"Calculated {k} out of {grid_n**2} Entries", end='\r')

    # region plotting
    # Cost Map
    fig, (ax_cost, ax_object) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    S, D = np.meshgrid(s_grid, d_grid, indexing='ij')
    cs = ax_cost.contourf(S, D, cost_grid, levels=50, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax_cost)
    cbar.set_label("Cost (squared wrench error)")
    ax_cost.plot(s_opt, d_opt, "ro", markersize=10, label="best")
    ax_cost.set_xlabel("s  (first contact fraction)")
    ax_cost.set_ylabel("d  (span along contour)")
    ax_cost.set_title("PSO cost landscape")
    ax_cost.legend(loc="upper right")

    # Contour panel
    body_patch = patches.Polygon(world_xy, closed=True, facecolor="lightgray", edgecolor="k", alpha=0.6)
    ax_object.add_patch(body_patch)


    p1_world = np.array([bx_world(s_opt), by_world(s_opt)])
    p2_world = np.array([bx_world((s_opt + d_opt) % 1.0), by_world((s_opt + d_opt) % 1.0)])
    p1_plot, = ax_object.plot(p1_world[0], p1_world[1], "ro", label="p1")
    p2_plot, = ax_object.plot(p2_world[0], p2_world[1], "go", label="p2")

    w_d_world = params.desired_wrench_world
    # w_d_body = (R.from_euler("xyz", [0, 0, params.body_theta], degrees=False).as_matrix()
    #             @ params.desired_wrench_world).reshape(3,)
    w_d_world_norm = norm_to_length(w_d_world[:2], len=0.05)
    ax_object.quiver(0, 0, w_d_world_norm[0], w_d_world_norm[1], angles="xy", scale_units="xy", scale=1, color="blue",
                   label="desired wrench")

    _, dbg = pso_wrench_cost_python(s_opt, d_opt, bx_world, by_world, params.desired_wrench_world,
                                    params.theta_friction, params.f_max, params.lam, params.lam_min, params.lam_max)
    w_hat_world = dbg.get("w_hat", np.zeros(3))
    w_hat_norm = norm_to_length(w_hat_world[:2], len=0.05)
    w_hat_quiver = ax_object.quiver(0, 0, w_hat_norm[0], w_hat_norm[1], angles="xy", scale_units="xy", scale=1,
                                  color="orange", label="produced wrench")

    v_lin = params.mu_base[0:2] * w_hat_world[0:2].flatten()
    omega = params.mu_base[2] * w_hat_world[2].flatten()
    v_g_world = np.array([v_lin[0], v_lin[1], omega[0]])
    v_g_norm = norm_to_length(v_g_world[:2], len=0.05)
    midx, midy = 0.5 * (p1_world[0] + p2_world[0]), 0.5 * (p1_world[1] + p2_world[1])
    v_gripper_quiver = ax_object.quiver(midx, midy, v_g_norm[0], v_g_norm[1], angles="xy", scale_units="xy", scale=1,
                                color="green", label="gripper twist")

    info_box_cost = ax_cost.text(0.02, 0.98, "", transform=ax_cost.transAxes, verticalalignment="top",
                                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    info_text_body = ax_object.text(0.02, 0.98, "", transform=ax_object.transAxes, verticalalignment="top",
                                  bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))

    ax_object.set_xlabel("world x (m)")
    ax_object.set_ylabel("world y (m)")
    ax_object.set_title(f"World frame visualisation for desired wrench [{w_d_world[0]:.2f}, {w_d_world[1]:.2f}, {w_d_world[2]:.2f}]")
    ax_object.axis("equal")
    ax_object.legend(loc="upper right")

    clicked_point_plot, = ax_cost.plot([], [], "go", markersize=8, label="selected")


    def update_body_plot(s_val: float, d_val: float):
        p1 = np.array([bx_world(s_val), by_world(s_val)])
        p2 = np.array([bx_world((s_val + d_val) % 1.0), by_world((s_val + d_val) % 1.0)])
        _, debug = pso_wrench_cost_python(s_val, d_val, bx_world, by_world, params.desired_wrench_world,
                                          params.theta_friction, params.f_max, params.lam, params.lam_min, params.lam_max)
        w_hat_wrld = debug["w_hat"]
        v_lin_local = params.mu_base[0:2] * w_hat_wrld[0:2].flatten()
        omega_local = params.mu_base[2] * w_hat_wrld[2].flatten()
        v_gripper_wrld = np.array([v_lin_local[0], v_lin_local[1], omega_local[0]])

        p1_plot.set_data([p1[0]], [p1[1]])
        p2_plot.set_data([p2[0]], [p2[1]])
        w_hat_quiver.set_UVC(*norm_to_length(w_hat_wrld[:2], len=0.05))
        v_gripper_quiver.set_UVC(*norm_to_length(v_gripper_wrld[:2], len=0.05))
        v_gripper_quiver.set_offsets([0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])])
        midx, midy = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        for art in list(ax_object.patches):
            if isinstance(art, mpl.patches.Arc) or isinstance(art, mpl.patches.RegularPolygon):
                art.remove()
        add_rot_arrow(ax_object, (0, 0), 0.04, 10*w_d_world[2], color="blue")
        add_rot_arrow(ax_object, (0, 0), 0.02, 10*w_hat_wrld[2][0], color="orange")
        add_rot_arrow(ax_object, (midx, midy), 0.01, 100*v_gripper_wrld[2], color="green")
        info_text_body.set_text(f"s={s_val:.3f}, d={d_val:.3f}\nw_hat={w_hat_wrld.T}\nv_g={v_gripper_wrld}")
        fig.canvas.draw_idle()
        print(f"\n\nSOLUTION STATS:                     ")
        for dbk, dbv in debug.items():
            if dbk in ["a_value", "w_hat", "residual"]:
                print(dbk, dbv.T) # make into row vectors for readability
            else:
                print(dbk, dbv)
    # endregion

    def handle_interaction(event):
        if event.inaxes != ax_cost:
            return
        sx, dx = event.xdata, event.ydata
        if sx is None or dx is None:
            return
        clicked_point_plot.set_data([sx], [dx])
        update_body_plot(sx, dx)
        cost_clicked, _ = pso_wrench_cost_python(sx, dx, bx_world, by_world, params.desired_wrench_world, params.theta_friction,
                                                 params.f_max, params.lam, params.lam_min, params.lam_max)
        info_box_cost.set_text(
            f"Clicked (s,d) = ({sx:.3f}, {dx:.3f})\nCost at click = {cost_clicked:.3f}\n"
            f"Best cost = {best_cost:.3f}\nBest (s,d) = ({s_opt:.3f}, {d_opt:.3f})"
        )
        fig.canvas.draw_idle()

    def onclick(event):
        handle_interaction(event)
    
    def onmove(event):
        if event.button is None: # Mouse button must be held down
            return
        handle_interaction(event)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("motion_notify_event", onmove)
    update_body_plot(s_opt, d_opt)

    plt.show()

# endregion

# --- Entrypoint -------------------------------------------------------
def main():
    contour_dict = CONTOUR_WRENCH
    char_rad = characteristic_radius(contour_dict)
    params = Params(
        contour=contour_dict,
        char_rad=char_rad,
        body_pos=np.array([0.0, 0.0, 0.0]),
        body_theta=4.4,
        theta_friction=0.0,
        desired_wrench_world=np.array([3.0, 0.0, 0.05]),
        f_max=5.0,
        lam=1.0,
        lam_min=0.01,
        lam_max=0.10,
        mu_base=np.array([0.001691, 0.001691, np.deg2rad(5.234536)]),
        pso_swarm=10,
        pso_iters=50,
        grid_n=21,
        lb=np.array([0.0, 0.0]),
        ub=np.array([1.0, 0.2]),
    )
    run(params)


if __name__ == "__main__":
    main()