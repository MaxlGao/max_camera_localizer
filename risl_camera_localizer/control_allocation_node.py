#!/usr/bin/env python3
# control_allocation_node.py

import math
from typing import Tuple
import numpy as np
import rclpy
import warnings
warnings.filterwarnings('ignore')
from rclpy.node import Callable, Node
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import TwistStamped, PoseStamped, WrenchStamped
from scipy.spatial.transform import Rotation as R
from functools import partial
import time
from shapely.geometry import LineString, Polygon

import cvxpy as cp
from pyswarms.single.global_best import GlobalBestPSO

# region Helpers
def wrap01(s):
    return np.mod(s, 1.0)

def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])

def cross2d_z(p, f):
    return p[0]*f[1] - p[1]*f[0]

def fmt4(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return fmt4(value.item())
        if np.issubdtype(value.dtype, np.floating):
            return np.array2string(
                value,
                formatter={"float_kind": lambda x: format(float(x), ".4g")},
                separator=", "
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

def periodic_interp(points_xy: np.ndarray):
    """
    Lightweight periodic, piecewise-linear interpolator for a closed 2D boundary.
    Returns bx(s), by(s) for s in [0,1).
    """
    pts = np.array(points_xy, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 boundary points.")
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    N = pts.shape[0] - 1

    def p_of_s(s):
        s = wrap01(np.asarray(s))
        t = s * N
        i0 = np.floor(t).astype(int)
        i1 = (i0 + 1) % (N)
        frac = (t - i0)[...]
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

def characteristic_radius(contour, k: float = 75) -> float:
    """
    Finds characteristic radius in meters, using the kth percentile of distance from the origin.
    That is, k% of all points are closer to the origin than the yielded 'characteristic radius'
    This is to bias towards points further away from the center of mass.

    Doesn't care about frame. Contour assumed as [x,y; x,y; ...]
    """
    dists = np.linalg.norm(contour, axis=1)
    r_char = float(np.percentile(dists, k))
    return r_char

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

# region Cost Function
def pso_wrench_cost_python(s, delta, bx, by, nx, ny, w_d: np.ndarray, alpha_ub,
                           theta_friction, f_max: float, char_rad, span_min, span_max, name, polygon):
    """
    Finds best PSO cost with cvxpy's OSQP. All in body frame

    Weights torque cost by characteristic radius. The smaller the object, the more impactful torque is.
    """
    # Mostly the same from run_sim.py, but with more care to world/body frame

    debug_info = {}
    s = float(s)
    s2 = wrap01(s + float(delta))
    p1 = np.array([float(bx(s)), float(by(s))])
    p2 = np.array([float(bx(s2)), float(by(s2))])
    n1 = np.array([float(nx(s)), float(ny(s))])
    n2 = np.array([float(nx(s2)), float(ny(s2))])
    dvec = p2 - p1
    dist = np.linalg.norm(dvec)

    # list of exceptions
    normal_dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    exceptions = (
        normal_dot < np.cos(np.deg2rad(150.0)), # High slip likelihood if normals differ by more than 150 degrees 
        not check_approach_clear(p1, p2, polygon, approach_length=0.1),
        dist < span_min or dist > span_max,
        # Temporary exception for wrench - if both points are between 0.86 and 0.91, localization drops. Exclude for now.
        ("wrench" in name) and (0.85 < s < 0.92) and (0.85 < s2 < 0.92), 
        # Temporary exception for allen_key - if both points are between 0.80 and 0.15 (wrapped around), localization drops. Exclude for now.
        ("allen_key" in name) and (0.80 < s or s < 0.15) and (0.80 < s2 or s2 < 0.15), 
    )
    if any(exceptions):
        return 1e6, debug_info

    debug_info['s1_and_s2'] = (s, s2)
    debug_info['n1'] = n1
    debug_info['n2'] = n2
    debug_info['normal_difference_deg'] = np.rad2deg(np.arccos(normal_dot))
    debug_info['p1_body'] = p1
    debug_info['p2_body'] = p2

    d_unit = dvec / (dist + 1e-12)
    n_g = np.array([-d_unit[1], d_unit[0]])

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

    debug_info['theta_friction'] = theta_friction
    debug_info['W'] = W.copy()
    debug_info['w_d_body'] = w_d.copy()

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
    except Exception:
        return 1e6, debug_info

    if a_pair.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return 1e6, debug_info

    a_pair_val = a_pair.value.copy()
    a_val = np.array([[a_pair_val[0]], [a_pair_val[1]], [a_pair_val[0]], [a_pair_val[1]]])
    w_hat = W @ a_val
    w_hat_weighted = W_weighted @ a_val
    resid = w_hat_weighted - w_d_weighted.reshape(-1, 1)
    cost = float(np.sum((w_hat_weighted - w_d_weighted.reshape(-1, 1))**2))

    debug_info['a_value'] = a_val
    debug_info['w_hat_weighted'] = w_hat_weighted
    debug_info['w_hat'] = w_hat
    debug_info['residual'] = resid
    debug_info['cost'] = cost

    return cost, debug_info

class PSOCostWrapper:
    def __init__(self, bx, by, nx, ny, w_d, alpha_ub, theta, f_mag, lam, lam_min, lam_max, n_iters, name, polygon):
        self.bx, self.by = bx, by
        self.nx, self.ny = nx, ny
        self.w_d = np.asarray(w_d, dtype=float).reshape(3,)
        self.alpha_ub = float(alpha_ub)
        self.theta = float(theta)
        self.f_mag = float(f_mag)
        self.lam = float(lam)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.iter = 0
        self.n_pso_iters = n_iters
        self.name = name
        self.polygon = polygon

    def __call__(self, swarm):
        # swarm: (n_particles, 2) -> costs (n_particles,)
        out = np.empty(swarm.shape[0], dtype=float)
        for i, (s, d) in enumerate(swarm):
            cost, debug = pso_wrench_cost_python(
                s, d, self.bx, self.by, self.nx, self.ny, self.w_d, self.alpha_ub,
                self.theta, self.f_mag, self.lam,
                self.lam_min, self.lam_max, self.name, self.polygon
            )
            out[i] = cost
            self.iter += 1
            print(f"Processed Particle {self.iter} / {swarm.shape[0]*self.n_pso_iters}", end= "\r")
        return out

# endregion

class ControlAllocationNode(Node):
    def __init__(self):
        super().__init__("control_allocation_node")
        #region Parameters
        # Body
        self.declare_parameter("desired_wrench", [2.0, 2.0, -0.2]) # Default; to be updated by push_with_allocation
        self.declare_parameter("com_xy", [0.0, 0.0])               # Body center of mass, wrt body origin (pls make it 0,0)
        self.declare_parameter("char_rad", 1.0)                    # General size of the body, measured as a characteristic radius
        self.w_d_body = np.array(self.get_parameter("desired_wrench").value, dtype=float).reshape(3,1)
        self.com =      np.array(self.get_parameter("com_xy").value, dtype=float)
        self.char_rad =    float(self.get_parameter("char_rad").value)

        # Friction / span
        self.declare_parameter("f_max", 5.0)                     # maximum allowed force
        self.declare_parameter("span_min", 0.01)
        self.declare_parameter("span_max", 0.10)
        self.declare_parameter("f2v_lin", 0.001691)              # v_{x, y} = a_lin * F_{x, y}
        self.declare_parameter("f2v_rot", np.deg2rad(5.234536))  # omega_z = a_rot * T_z
        self.f_max =        float(self.get_parameter("f_max").value)
        self.span_min =     float(self.get_parameter("span_min").value)
        self.span_max =     float(self.get_parameter("span_max").value)
        self.f2v_base = np.array([self.get_parameter("f2v_lin").value, 
                                  self.get_parameter("f2v_lin").value, 
                                  self.get_parameter("f2v_rot").value])

        # PSO - Total particles is swarm * iters, and takes 200 particles per second.
        self.declare_parameter("pso_swarm", 20)                  # 100 nominal, 20 quick
        self.declare_parameter("pso_iters", 20)                  # 50 nominal, 20 quick
        self.declare_parameter("lb", [0.00, 0.00])               # case1 defaults
        self.declare_parameter("ub", [1.00, 0.20])               # first point anywhere from 0 to 1, second point must be within 0.2.
                                                                 # second column of lb ub isn't necessarily *span* limits, but keeping for now.
        self.pso_swarm = int(self.get_parameter("pso_swarm").value)
        self.pso_iters = int(self.get_parameter("pso_iters").value)
        self.lb =   np.array(self.get_parameter("lb").value, dtype=float)
        self.ub =   np.array(self.get_parameter("ub").value, dtype=float)

        # Adaptation
        self.declare_parameter("mu_init", [1.0, 1.0, 1.0])       # static friction in x, y, and rotational, initial
        self.declare_parameter("alpha_ub", 1.0)
        self.declare_parameter("eta", 1e4)                       # Learning Rate for Friction Adaptation; how big of a step to take in the direction of the gradient
        self.declare_parameter("fd_eps", 1e-3)
        # self.declare_parameter("update_rate_hz", 10.0)
        self.mu =   np.array(self.get_parameter("mu_init").value, dtype=float)
        self.alpha_ub =float(self.get_parameter("alpha_ub").value)
        self.alpha_ub_closed = self.alpha_ub
        self.eta =     float(self.get_parameter("eta").value)
        self.fd_eps =  float(self.get_parameter("fd_eps").value)
        # self.timer = self.create_timer(1.0 / float(self.get_parameter("update_rate_hz").value), self.control_step)
        
        self.theta_friction = math.atan(np.mean(self.mu))

        self.target_name = None
        self.boundary = None     # (N,2)
        self.bx = None
        self.by = None
        self.polygon = None
        self.nx = None
        self.ny = None
        self.vhat = None
        self.v_meas = np.zeros(3) # body frame
        self.s1_opt = None
        self.s2_opt = None
        self.w_hat = None
        #endregion

        #region ROS2 IO
        self.sub_target_name =        self.create_subscription(String,            "/robot/target_name", self.target_name_cb, 10)
        self.sub_desired_wrench =     self.create_subscription(Float32MultiArray, "/robot/wrench_desired", self.wrench_cb, 10)
        self.sub_push_status =        self.create_subscription(Bool,              "/robot/currently_pushing", self.pushing_cb, 10)
        self.sub_boundary =           self.create_subscription(Float32MultiArray, "/vision/boundary_points_body", self.boundary_cb, 10)
        self.sub_boundary_normals =   self.create_subscription(Float32MultiArray, "/vision/boundary_normals_body", self.boundary_normals_cb, 10)
        self.sub_obj_vels = {} # averaged over push
        for name in ("wrench", "jenga_6", "allen_key"):
            vel_topic = f"/vision/object_twists_push_avg/{name}"
            vel_cb = partial(self.vel_cb, name)
            self.sub_obj_vels[name] =  self.create_subscription(TwistStamped, vel_topic, vel_cb, 10)

        self.pub_push =               self.create_publisher(Float32MultiArray, "/allocation/push_points", 10)
        self.pub_mu =                 self.create_publisher(Float32MultiArray, "/allocation/mu_alpha", 10)
        self.pub_desired_wrench =     self.create_publisher(Float32MultiArray, "/allocation/wrench_desired", 10) # echo as a way to say "done"
        self.pub_gripper_twist =      self.create_publisher(TwistStamped,      "/allocation/recommended_gripper_twist_body", 10)
        self.pub_v_hat =              self.create_publisher(TwistStamped,      "/allocation/v_hat", 10)
        self.pub_w_hat =              self.create_publisher(WrenchStamped,     "/allocation/w_hat", 10)
        #endregion

        self.get_logger().info("control_allocation_node ready.")

    #region Callbacks
    def target_name_cb(self, msg: String):
        self.target_name = msg.data

    def boundary_cb(self, msg: Float32MultiArray):
        """Boundary is in body frame. Units should be in meters."""
        data = np.array(msg.data, dtype=float)
        if data.size < 6 or data.size % 2 != 0:
            self.get_logger().warn("Boundary malformed; expected [x1,y1,x2,y2,...].")
            return
        pts_xy = data.reshape(-1, 2)  # [x,y; x,y; ...]
        self.boundary = pts_xy
        self.bx, self.by = periodic_interp(pts_xy)

        # Compute normals consistently from the boundary points (same method as run_sim).
        # This avoids any sign flips that can occur if we interpret a normal vector list as a polygon.
        self.nx, self.ny = periodic_interp_normals(pts_xy)

        # Create a Shapely polygon from the boundary points
        self.polygon = Polygon(pts_xy)

        self.char_rad = characteristic_radius(pts_xy)

    def boundary_normals_cb(self, msg: Float32MultiArray):
        """Boundary normals are in body frame. Units should be in meters."""
        data = np.array(msg.data, dtype=float)
        if data.size < 6 or data.size % 2 != 0:
            self.get_logger().warn("Boundary normals malformed; expected [x1,y1,x2,y2,...].")
            return
        pts_xy = data.reshape(-1, 2)  # [x,y; x,y; ...]
        self.boundary_normals = pts_xy

        # We compute normals from the boundary points instead of re-interpreting the
        # provided normals as a polygon. This avoids sign flips that can arise if the
        # normals list is treated as a boundary point list.
        if self.boundary is not None:
            self.nx, self.ny = periodic_interp_normals(self.boundary)

    def vel_cb(self, name: str, msg: TwistStamped):
        if name != self.target_name: # ignore if object isn't target
            return
        self.v_meas = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z], dtype=float)
        print(f"Twist meas: {fmt4(self.v_meas)}", end="\r")

    def wrench_cb(self, msg: Float32MultiArray):
        """
        Process Wrench Desired Intake, which would be from the OverController module.
        Trust this desired wrench to be in body frame.
        """
        self.w_d_body = np.array(msg.data, dtype=float).reshape(3,1)
        # An incoming desired wrench is grounds to get on the control step.
        self.control_step()
        # flag to ensure one at a time? nah, adaptation's fast.

    def pushing_cb(self, msg: Bool):
        if msg.data == False:
            # if it declared no push, start on the adaptation section
            self.adapt_friction()

    # endregion

    # region Point Selection
    def control_step(self):
        print("\n\n---------- Starting Control Step ----------")
        print(f"Using W_d_body = {fmt4(self.w_d_body.T)} on Target {self.target_name}")
        while self.boundary is None or self.nx is None or self.ny is None:
            print(f" Boundary: {'Yes' if self.boundary is not None else 'No'}, Normals: {'Yes' if self.nx is not None and self.ny is not None else 'No'}", end="\r")
            time.sleep(0.3)

        # Build PSO cost with current boundary
        cost = PSOCostWrapper(
            self.bx, self.by, self.nx, self.ny, self.w_d_body, self.alpha_ub, self.theta_friction, self.f_max,
            self.char_rad, self.span_min, self.span_max, self.pso_iters, self.target_name, self.polygon
        )

        optimizer = GlobalBestPSO(
            n_particles=self.pso_swarm,
            dimensions=2,
            options={"c1": 1.3, "c2": 1.3, "w": 0.6},
            bounds=(self.lb, self.ub),
        )
        b_cost, (s_opt, d_opt) = optimizer.optimize(cost, iters=self.pso_iters, verbose=False)
        # Test the best point
        cost_dbg, dbg = self.debug_solve_at(s_opt, d_opt)
        print(f"SOLUTION STATS:                     ")
        for dbk, dbv in dbg.items():
            if dbk in ["w_d_body", "a_value", "w_hat_weighted", "w_hat", "residual"]:
                print(dbk, fmt4(dbv.T)) # make into row vectors for readability
            else:
                print(dbk, fmt4(dbv))

        # Compute/contact points
        self.s1_opt = float(s_opt)
        self.s2_opt = float((s_opt + d_opt) % 1.0)
        p1 = np.array([float(self.bx(self.s1_opt)), float(self.by(self.s1_opt))])
        p2 = np.array([float(self.bx(self.s2_opt)), float(self.by(self.s2_opt))])

        # Publish push points
        msg_pts = Float32MultiArray()
        msg_pts.data = [p1[0], p1[1], p2[0], p2[1]]
        self.pub_push.publish(msg_pts)
        self.w_hat = dbg["w_hat"]

        msg_w_d_body = Float32MultiArray()
        msg_w_d_body.data = [float(self.w_d_body[0]),float(self.w_d_body[1]),float(self.w_d_body[2])]
        self.pub_desired_wrench.publish(msg_w_d_body) 

        self.vhat = self.v_hat(self.f2v_base, self.w_hat, self.mu)
        # publish v_hat and w_hat for debugging/visualization
        msg_v_hat = TwistStamped()
        msg_v_hat.header.stamp = self.get_clock().now().to_msg()
        msg_v_hat.header.frame_id = "body"
        msg_v_hat.twist.linear.x = float(self.vhat[0])
        msg_v_hat.twist.linear.y = float(self.vhat[1])
        msg_v_hat.twist.linear.z = 0.0
        msg_v_hat.twist.angular.x = 0.0
        msg_v_hat.twist.angular.y = 0.0
        msg_v_hat.twist.angular.z = float(self.vhat[2])
        self.pub_v_hat.publish(msg_v_hat)

        msg_w_hat = WrenchStamped()
        msg_w_hat.header.stamp = self.get_clock().now().to_msg()
        msg_w_hat.header.frame_id = "body"
        msg_w_hat.wrench.force.x = float(self.w_hat[0])
        msg_w_hat.wrench.force.y = float(self.w_hat[1])
        msg_w_hat.wrench.force.z = 0.0
        msg_w_hat.wrench.torque.x = 0.0
        msg_w_hat.wrench.torque.y = 0.0
        msg_w_hat.wrench.torque.z = float(self.w_hat[2])
        self.pub_w_hat.publish(msg_w_hat)

        # Find required gripper twist to track object (body frame)
        v_lin = self.vhat[0:2]
        omega = self.vhat[2]
        v_p1_rot = omega*np.array([-p1[1], p1[0]]) # 90 degree rot
        v_p2_rot = omega*np.array([-p2[1], p2[0]]) # 90 degree rot
        print(f"Got v_p1_rot = {fmt4(v_p1_rot)}")
        print(f"Got v_p2_rot = {fmt4(v_p2_rot)}")
        v_p1 = v_lin + v_p1_rot
        v_p2 = v_lin + v_p2_rot
        v_g_lin = (v_p1 + v_p2) * 0.5
        v_g = np.array([v_g_lin[0], v_g_lin[1], omega])
        print(f"Got v_g (body) = {fmt4(v_g)}")
        # Cap omega at 15 deg/s for safety
        omega_cap = np.deg2rad(15)
        if abs(omega) > omega_cap:
            print(
                f"Omega {fmt4(np.rad2deg(omega))} deg/s exceeds cap of "
                f"{fmt4(np.rad2deg(omega_cap))} deg/s, capping."
            )
            omega = np.sign(omega) * omega_cap
            v_g = np.array([v_g_lin[0], v_g_lin[1], omega])
            print(f"Got v_g (body) (new) = {fmt4(v_g)}")

        msg_twist = TwistStamped()
        msg_twist.header.stamp = self.get_clock().now().to_msg()
        msg_twist.header.frame_id = "body"
        msg_twist.twist.linear.x = v_lin[0]
        msg_twist.twist.linear.y = v_lin[1]
        msg_twist.twist.linear.z = 0.0
        msg_twist.twist.angular.x = 0.0
        msg_twist.twist.angular.y = 0.0
        msg_twist.twist.angular.z = omega
        self.pub_gripper_twist.publish(msg_twist)
    # endregion

    # region Adaptation
    def adapt_friction(self):
        print("\n\n---------- Starting Friction Adaptation ----------")
        print(f"Using W_d_body = {fmt4(self.w_d_body.T)} on Target {self.target_name}")
        # ---- Adapt μ using e = v_hat - ṽ and central diff Jacobian ----
        e = self.vhat - self.v_meas  # e = v̂ - ṽ
        print(f"Got Vhat (body) = {fmt4(self.vhat)}")
        print(f"Got Vmeas (body) = {fmt4(self.v_meas)}")
        print(f"Got Error Twist = {fmt4(e)}")


        # Sigma-based adaptation
        sigma = 1.0 / np.clip(self.mu, 1e-12, None)
        J = np.zeros((3, 3))
        for i in range(3):
            # note: v_hat is linear to sigma, so jacobian is just v_hat with perturbed sigma
            sigma_f = sigma.copy(); sigma_f[i] += self.fd_eps
            sigma_b = sigma.copy(); sigma_b[i] -= self.fd_eps
            v_f = self.v_hat(self.f2v_base, self.w_hat, 1.0 / sigma_f)
            v_b = self.v_hat(self.f2v_base, self.w_hat, 1.0 / sigma_b)
            J[:, i] = (v_f - v_b) / (2.0 * self.fd_eps)

        # old method
        # J = np.zeros((3, 3))
        # for i in range(3):
        #     mu_f = self.mu.copy(); mu_f[i] += self.fd_eps
        #     mu_b = self.mu.copy(); mu_b[i] -= self.fd_eps
        #     J[:, i] = (self.v_hat(self.f2v_base, self.w_hat, mu_f) - self.v_hat(self.f2v_base, self.w_hat, mu_b)) / (2.0 * self.fd_eps)

        print(f"Got Jacobian J = {fmt4(J)}")
        grad = 2.0 * (J.T @ e)  # ∂||e||^2/∂μ
        print(f"Got Gradient 2*de/dsigma*e = {fmt4(grad)}")
        print(f"Got Mu Old = {fmt4(self.mu)}")
        # Weight rotation component according to characteristic radius (squared for effect in e, v_hat)
        eta_vec = self.eta * np.array([1.0, 1.0, self.char_rad**2])
        
        # sigma-based method
        sigma = np.clip(sigma - grad * eta_vec, 1e-1, 1e1)
        self.mu = 1.0 / sigma

        # old method
        # # clamp mu to reasonable values above fd_eps to avoid instability
        # self.mu = np.clip(self.mu - grad * eta_vec, 1e-2, 10.0)
        print(f"Got Mu New = {fmt4(self.mu)}")
        self.theta_friction = math.atan(np.mean(self.mu))

        # weight rotation component of v meas and hat according to characteristic radius, 
        # since smaller objects are more sensitive to torque errors
        v_meas_weighted = self.v_meas.copy()
        v_meas_weighted[2] *= self.char_rad
        vhat_weighted = self.vhat.copy()
        vhat_weighted[2] *= self.char_rad
        print(f"Got weighted Vhat (body) = {fmt4(vhat_weighted)}")
        print(f"Got weighted Vmeas (body) = {fmt4(v_meas_weighted)}")
        print(f"Error Twist weighted = {fmt4(vhat_weighted - v_meas_weighted)}")
        
        # α_ub = (v̂^T ṽ)/||v̂||^2
        denom = float(np.dot(vhat_weighted, vhat_weighted)) + 1e-12
        closed_alpha_ub = float(np.dot(vhat_weighted, v_meas_weighted) / denom)
        # clamp closed-form alpha_ub to [0.1, 10] to avoid extreme values
        closed_alpha_ub = np.clip(closed_alpha_ub, 0.1, 10.0)

        # Blend alpha_ub using previous (effective) value to smooth updates
        #region FLAG FOR ATTENTION
        #endregion
        # effective_alpha_ub = 0.5 * self.alpha_ub + 0.5 * closed_alpha_ub
        effective_alpha_ub = closed_alpha_ub
        effective_alpha_ub = np.clip(effective_alpha_ub, 0.1, 10.0)

        self.alpha_ub_closed = closed_alpha_ub
        self.alpha_ub = effective_alpha_ub
        print(
            f"Got alpha_ub_closed = {fmt4(self.alpha_ub_closed)}, "
            f"alpha_ub_effective = {fmt4(self.alpha_ub)}"
        )

        # Publish μ and α_ub (effective + closed-form)
        msg_mu = Float32MultiArray()
        msg_mu.data = [
            float(self.mu[0]), float(self.mu[1]), float(self.mu[2]),
            float(self.alpha_ub), float(self.alpha_ub_closed)
        ]
        self.pub_mu.publish(msg_mu)
    # endregion

    # Model placeholder for v(f2v): f2v .* w ./ mu
    def v_hat(self, f2v_vec: np.ndarray, w_hat: np.ndarray, mu: np.ndarray):
        return f2v_vec.reshape(3,) * w_hat.reshape(3,) / mu.reshape(3,)

    def debug_solve_at(self, s, d):
        cost, dbg = pso_wrench_cost_python(
            s, d, self.bx, self.by, self.nx, self.ny, self.w_d_body, self.alpha_ub,
            self.theta_friction, self.f_max, self.char_rad,
            self.span_min, self.span_max, self.target_name, self.polygon
        )
        return cost, dbg

def main(args=None):
    rclpy.init(args=args)
    node = ControlAllocationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
