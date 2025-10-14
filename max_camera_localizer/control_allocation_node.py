#!/usr/bin/env python3
# control_allocation_node.py

import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TwistStamped

# ---- Dependencies for optimization ----
import cvxpy as cp
from pyswarms.single.global_best import GlobalBestPSO

# ==================== Helpers ====================

def wrap01(s):
    return np.mod(s, 1.0)

def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])

def cross2d_z(p, f):
    return p[0]*f[1] - p[1]*f[0]

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

# ==================== Cost Function ====================

def pso_wrench_cost_python(s, delta, bx, by, w_d, theta, f_mag, lam, lam_min, lam_max):
    """
    Implements MATLAB pso_wrench_cost in Python with cvxpy (OSQP).
    """
    # Format points, span
    s  = float(s)
    s2 = (s + float(delta)) % 1.0

    p1 = np.array([float(bx(s)),  float(by(s))])
    p2 = np.array([float(bx(s2)), float(by(s2))])
    d  = p2 - p1
    dist = np.linalg.norm(d)
    if dist < lam_min or dist > lam_max:
        print("Lambda Fault!")
        print(dist)
        return 1e6

    # Friction directions from span
    d_unit = d / (dist + 1e-12)
    n_g = np.array([-d_unit[1], d_unit[0]])  # +90° rotation

    f1_1 = f_mag * (rot2(+theta) @ n_g)
    f1_2 = f_mag * (rot2(-theta) @ n_g)
    f2_1 = f_mag * (rot2(+theta) @ n_g)
    f2_2 = f_mag * (rot2(-theta) @ n_g)

    w11 = np.array([f1_1[0], f1_1[1], lam * cross2d_z(p1, f1_1)])
    w12 = np.array([f1_2[0], f1_2[1], lam * cross2d_z(p1, f1_2)])
    w21 = np.array([f2_1[0], f2_1[1], lam * cross2d_z(p2, f2_1)])
    w22 = np.array([f2_2[0], f2_2[1], lam * cross2d_z(p2, f2_2)])
    W   = np.column_stack([w11, w12, w21, w22])  # 3x4

    a = cp.Variable(4, nonneg=True)
    A = np.array([[1., 1., 0., 0.],
                  [0., 0., 1., 1.]])
    b = np.array([1., 1.])
    obj = cp.Minimize(cp.sum_squares(W @ a - w_d))
    prob = cp.Problem(obj, [A @ a <= b])

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    except Exception:
        print("Solver Fault!")
        return 1e6

    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        print("Not Optimal!")
        return 1e6

    w_hat = W @ a.value
    # print("Success!")
    return float(np.sum((w_hat - w_d)**2))

class PSOCostWrapper:
    def __init__(self, bx, by, w_d, theta, f_mag, lam, lam_min, lam_max):
        self.bx, self.by = bx, by
        self.w_d = np.asarray(w_d, dtype=float).reshape(3,)
        self.theta = float(theta)
        self.f_mag = float(f_mag)
        self.lam = float(lam)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)

    def __call__(self, swarm):
        # swarm: (n_particles, 2) -> costs (n_particles,)
        out = np.empty(swarm.shape[0], dtype=float)
        for i, (s, d) in enumerate(swarm):
            out[i] = pso_wrench_cost_python(
                s, d, self.bx, self.by, self.w_d,
                self.theta, self.f_mag, self.lam,
                self.lam_min, self.lam_max
            )
        return out

# ==================== ROS 2 Node ====================

class ControlAllocationNode(Node):
    def __init__(self):
        super().__init__("control_allocation_node")

        # -------- Parameters (tweak from CLI) --------
        self.declare_parameter("boundary_topic", "/vision/boundary_points")
        self.declare_parameter("velocity_topic", "/robot/commanded_twist")
        self.declare_parameter("push_points_topic", "/allocation/push_points")
        self.declare_parameter("mu_alpha_topic", "/allocation/mu_alpha")

        self.declare_parameter("com_xy", [0.0, 0.0])             # center of mass
        self.declare_parameter("desired_wrench", [0.0, 1.0, 0.0])

        # Friction / span
        self.declare_parameter("theta", float(math.atan(0.5)))   # mu = 0.5
        self.declare_parameter("f_mag", 1.0)
        self.declare_parameter("lambda", 0.5)                    # torque scale in your cost
        self.declare_parameter("lambda_min", 0.15)
        self.declare_parameter("lambda_max", 0.50)

        # PSO
        self.declare_parameter("pso_swarm", 50)
        self.declare_parameter("pso_iters", 50)
        self.declare_parameter("lb", [0.10, 0.00])               # case1 defaults
        self.declare_parameter("ub", [0.20, 0.10])

        # Adaptation
        self.declare_parameter("mu_init", [1.0, 1.0, 1.0])
        self.declare_parameter("eta", 0.05)
        self.declare_parameter("fd_eps", 1e-3)
        self.declare_parameter("update_rate_hz", 10.0)

        # -------- State --------
        self.boundary = None     # (N,2)
        self.bx = None
        self.by = None
        self.com = np.array(self.get_parameter("com_xy").value, dtype=float)
        self.w_d = np.array(self.get_parameter("desired_wrench").value, dtype=float).reshape(3,)
        self.theta = float(self.get_parameter("theta").value)
        self.f_mag = float(self.get_parameter("f_mag").value)
        self.lam = float(self.get_parameter("lambda").value)
        self.lam_min = float(self.get_parameter("lambda_min").value)
        self.lam_max = float(self.get_parameter("lambda_max").value)

        self.mu = np.array(self.get_parameter("mu_init").value, dtype=float)  # R^3
        self.alpha_ub = 1.0
        self.v_meas = np.zeros(3)

        # -------- IO --------
        self.sub_boundary = self.create_subscription(
            Float32MultiArray,
            self.get_parameter("boundary_topic").value,
            self.boundary_cb, 10
        )
        self.sub_vel = self.create_subscription(
            TwistStamped,
            self.get_parameter("velocity_topic").value,
            self.vel_cb, 10
        )
        self.pub_push = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("push_points_topic").value, 10
        )
        self.pub_mu = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("mu_alpha_topic").value, 10
        )

        self.timer = self.create_timer(
            1.0 / float(self.get_parameter("update_rate_hz").value),
            self.control_step
        )

        self.get_logger().info("control_allocation_node ready.")

    # -------- Callbacks --------
    def boundary_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.size < 6 or data.size % 2 != 0:
            self.get_logger().warn("Boundary malformed; expected [x1,y1,x2,y2,...].")
            return
        pts = data.reshape(-1, 2)
        self.boundary = pts
        self.bx, self.by = periodic_interp(pts)

    def vel_cb(self, msg: TwistStamped):
        self.v_meas = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z], dtype=float)

    # -------- Main loop --------
    def control_step(self):
        print("Starting Control Step")
        if self.boundary is None:
            return

        # Build PSO cost with current boundary
        cost = PSOCostWrapper(
            self.bx, self.by,
            self.w_d, self.theta, self.f_mag,
            self.lam, self.lam_min, self.lam_max
        )

        lb = np.array(self.get_parameter("lb").value, dtype=float)
        ub = np.array(self.get_parameter("ub").value, dtype=float)

        optimizer = GlobalBestPSO(
            n_particles=int(self.get_parameter("pso_swarm").value),
            dimensions=2,
            options={"c1": 1.3, "c2": 1.3, "w": 0.6},
            bounds=(lb, ub),
        )
        b_cost, (s_opt, d_opt) = optimizer.optimize(cost, iters=int(self.get_parameter("pso_iters").value), verbose=False)

        # Compute/contact points
        s1 = float(s_opt)
        s2 = float((s_opt + d_opt) % 1.0)
        p1 = np.array([float(self.bx(s1)), float(self.by(s1))])
        p2 = np.array([float(self.bx(s2)), float(self.by(s2))])

        # Publish push points
        msg_pts = Float32MultiArray()
        msg_pts.data = [p1[0], p1[1], p2[0], p2[1]]
        print(f"Got Contacts at {1000*p1[0]:.0f}, {1000*p1[1]:.0f} and {1000*p2[0]:.0f}, {1000*p2[1]:.0f}")
        self.pub_push.publish(msg_pts)

        # ---- Adapt μ using e = v_hat - ṽ and central diff Jacobian ----
        # Model placeholder for v_hat(μ): diag(μ) * w_d  (swap with your mapping if needed)
        def v_hat(mu_vec):
            return np.asarray(mu_vec).reshape(3,) * self.w_d

        vhat = v_hat(self.mu)
        e = vhat - self.v_meas  # e = v̂ - ṽ

        eps = float(self.get_parameter("fd_eps").value)
        J = np.zeros((3, 3))
        for i in range(3):
            mu_f = self.mu.copy(); mu_f[i] += eps
            mu_b = self.mu.copy(); mu_b[i] -= eps
            J[:, i] = (v_hat(mu_f) - v_hat(mu_b)) / (2.0 * eps)

        grad = 2.0 * (J.T @ e)  # ∂||e||^2/∂μ
        eta = float(self.get_parameter("eta").value)
        self.mu = self.mu - eta * grad
        print(f"Got Mu = {self.mu}")

        # α_ub = (v̂^T ṽ)/||v̂||^2
        print(f"Got Vhat, Vmeas = {vhat}, {self.v_meas}")
        denom = float(np.dot(vhat, vhat)) + 1e-12
        self.alpha_ub = float(np.dot(vhat, self.v_meas) / denom)
        print(f"Got Alpha_ub = {self.alpha_ub}")

        # Publish μ and α_ub
        msg_mu = Float32MultiArray()
        msg_mu.data = [float(self.mu[0]), float(self.mu[1]), float(self.mu[2]), float(self.alpha_ub)]
        self.pub_mu.publish(msg_mu)

# ==================== Entrypoint ====================

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