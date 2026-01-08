#!/usr/bin/env python3
# control_allocation_node.py

import math
import numpy as np
import rclpy
import warnings
warnings.filterwarnings('ignore')
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TwistStamped, PoseStamped, WrenchStamped
from scipy.spatial.transform import Rotation as R

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
    Finds best PSO cost with cvxpy's OSQP.
    """
    debug_info = {}
    # Format points, span
    s  = float(s)
    s2 = (s + float(delta)) % 1.0

    p1 = np.array([float(bx(s)),  float(by(s))])
    p2 = np.array([float(bx(s2)), float(by(s2))])
    d  = p2 - p1
    dist = np.linalg.norm(d)
    if dist < lam_min or dist > lam_max:
        return 1e6, debug_info

    debug_info['s'] = s
    # debug_info['delta'] = delta
    debug_info['s2'] = s2
    debug_info['p1_body'] = p1.copy()
    debug_info['p2_body'] = p2.copy()
    debug_info['span_dist'] = dist

    # Friction directions from span
    d_unit = d / (dist + 1e-12)
    n_g = np.array([-d_unit[1], d_unit[0]])  # +90° rotation
    debug_info['normal_direction_body'] = n_g.copy()

    f1_1 = f_mag * (rot2(+theta) @ n_g)
    f1_2 = f_mag * (rot2(-theta) @ n_g)
    f2_1 = f_mag * (rot2(+theta) @ n_g)
    f2_2 = f_mag * (rot2(-theta) @ n_g)

    w11 = np.array([f1_1[0], f1_1[1], lam * cross2d_z(p1, f1_1)])
    w12 = np.array([f1_2[0], f1_2[1], lam * cross2d_z(p1, f1_2)])
    w21 = np.array([f2_1[0], f2_1[1], lam * cross2d_z(p2, f2_1)])
    w22 = np.array([f2_2[0], f2_2[1], lam * cross2d_z(p2, f2_2)])
    W   = np.column_stack([w11, w12, w21, w22])  # 3x4 (point 1 ray 1, point 1 ray 2, etc...)


    debug_info['W'] = W.copy()
    debug_info['w_d_body'] = w_d.copy()

    # a = cp.Variable(4, nonneg=True) # 4 independent a
    a_unit = cp.Variable(nonneg=True) # Force all alpha the same === assume frictionless contact.
    a = cp.vstack([a_unit, a_unit, a_unit, a_unit])
    A = np.array([[1., 1., 0., 0.],
                  [0., 0., 1., 1.]])
    b = np.array([1., 1.])
    obj = cp.Minimize(cp.sum_squares(W @ a - w_d))
    prob = cp.Problem(obj, [A @ a <= b])

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    except Exception:
        return 1e6, debug_info

    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return 1e6, debug_info

    a_val = a.value.copy()
    w_hat = W @ a.value
    resid = w_hat - w_d
    cost = float(np.sum((w_hat-w_d)**2))

    debug_info['a_value'] = a_val
    debug_info['w_hat'] = w_hat.copy()
    debug_info['residual'] = resid
    debug_info['cost'] = cost

    return cost, debug_info

class PSOCostWrapper:
    def __init__(self, bx, by, w_d, theta, f_mag, lam, lam_min, lam_max, n_iters):
        self.bx, self.by = bx, by
        self.w_d = np.asarray(w_d, dtype=float).reshape(3,)
        self.theta = float(theta)
        self.f_mag = float(f_mag)
        self.lam = float(lam)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.iter = 0
        self.n_pso_iters = n_iters

    def __call__(self, swarm):
        # swarm: (n_particles, 2) -> costs (n_particles,)
        out = np.empty(swarm.shape[0], dtype=float)
        for i, (s, d) in enumerate(swarm):
            cost, debug = pso_wrench_cost_python(
                s, d, self.bx, self.by, self.w_d,
                self.theta, self.f_mag, self.lam,
                self.lam_min, self.lam_max
            )
            out[i] = cost
            self.iter += 1
            print(f"Processed Particle {self.iter} / {swarm.shape[0]*self.n_pso_iters}", end= "\r")
        return out

# ==================== ROS 2 Node ====================

class ControlAllocationNode(Node):
    def __init__(self):
        super().__init__("control_allocation_node")
        #region Parameters
        self.declare_parameter("boundary_topic", "/vision/boundary_points_body")
        self.declare_parameter("velocity_topic", "/robot/commanded_twist")
        self.declare_parameter("push_points_topic", "/allocation/push_points")
        self.declare_parameter("mu_alpha_topic", "/allocation/mu_alpha")
        self.declare_parameter("wrench_desired_topic", "/allocation/wrench_desired")
        self.declare_parameter("wrench_desired_topic_intake", "/robot/wrench_desired")
        self.declare_parameter("object_pose_topic", "/object_poses/jenga_6") # or some other object
        self.declare_parameter("gripper_twist_topic", "/allocation/gripper_twist")

        self.declare_parameter("com_xy", [0.0, 0.0])             # center of mass
        self.declare_parameter("desired_wrench", [2.0, 2.0, -0.2])

        # Friction / span
        self.declare_parameter("theta", float(math.atan(0.1)))   # mu = 0.1
        self.declare_parameter("f_mag", 5.0)                     # maximum allowed force
        self.declare_parameter("lambda", 1.0)                    # torque scale in your cost
        self.declare_parameter("lambda_min", 0.01)
        self.declare_parameter("lambda_max", 0.10)
        self.declare_parameter("a_lin", 0.001691)               # v_{x, y} = a_lin * F_{x, y}
        self.declare_parameter("a_rot", np.deg2rad(5.234536))    # omega_z = a_rot * T_z

        # PSO - Total particles is swarm * iters, and takes 200 particles per second. 100/50 is nominal
        self.declare_parameter("pso_swarm", 100)
        self.declare_parameter("pso_iters", 50)
        self.declare_parameter("lb", [0.00, 0.00])               # case1 defaults
        self.declare_parameter("ub", [1.00, 0.20])               # first point anywhere from 0 to 1, second point must be within 0.2.

        # Adaptation
        self.declare_parameter("mu_init", [1.0, 1.0, 1.0])
        self.declare_parameter("eta", 0.05)
        self.declare_parameter("fd_eps", 1e-3)
        self.declare_parameter("update_rate_hz", 10.0)

        self.boundary = None     # (N,2)
        self.bx = None
        self.by = None
        self.mu_base = np.array([self.get_parameter("a_lin").value, 
                                 self.get_parameter("a_lin").value, 
                                 self.get_parameter("a_rot").value])
        self.com = np.array([0.0, 0.0]) # Center of Mass
        self.theta_body = 0.0 # Body angle, radians
        self.w_d_world = np.array(self.get_parameter("desired_wrench").value, dtype=float).reshape(3,1)
        self.w_d = self.w_d_world
        self.theta = float(self.get_parameter("theta").value)
        self.f_mag = float(self.get_parameter("f_mag").value)
        self.lam = float(self.get_parameter("lambda").value)
        self.lam_min = float(self.get_parameter("lambda_min").value)
        self.lam_max = float(self.get_parameter("lambda_max").value)
        self.pso_iters = int(self.get_parameter("pso_iters").value)

        self.mu = np.array(self.get_parameter("mu_init").value, dtype=float)  # R^3
        self.alpha_ub = 1.0
        self.v_meas = np.zeros(3)
        self.s1_opt = None
        self.s2_opt = None
        #endregion

        #region ROS2 IO
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
        self.sub_obj = self.create_subscription(
            PoseStamped,
            self.get_parameter("object_pose_topic").value,
            self.pose_cb, 10
        )
        self.sub_desired = self.create_subscription(
            Float32MultiArray,
            self.get_parameter("wrench_desired_topic_intake").value,
            self.wrench_cb, 10
        )
        self.pub_push = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("push_points_topic").value, 10
        )
        self.pub_mu = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("mu_alpha_topic").value, 10
        )
        self.pub_desired = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("wrench_desired_topic").value, 10
        )
        self.gripper_twist = self.create_publisher(
            TwistStamped,
            self.get_parameter("gripper_twist_topic").value, 10
        )
        # self.timer = self.create_timer(
        #     1.0 / float(self.get_parameter("update_rate_hz").value),
        #     self.control_step
        # )
        #endregion

        self.get_logger().info("control_allocation_node ready.")

    #region -------- Callbacks --------
    def boundary_cb(self, msg: Float32MultiArray):
        """Boundary is in body frame"""
        # print("Received new boundary!!")
        data = np.array(msg.data, dtype=float)
        if data.size < 6 or data.size % 2 != 0:
            self.get_logger().warn("Boundary malformed; expected [x1,y1,x2,y2,...].")
            return
        pts = data.reshape(-1, 2) # [x,y; x,y; ...]
        self.boundary = pts
        self.bx, self.by = periodic_interp(pts)

    def vel_cb(self, msg: TwistStamped):
        self.v_meas = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z], dtype=float)

    def pose_cb(self, msg: PoseStamped):
        # self.com = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        body_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                              msg.pose.orientation.z, msg.pose.orientation.w])
        body_rpy = R.from_quat(body_quat).as_euler('xyz', degrees=False)
        self.theta_body = body_rpy[2]
        w_d_lin_body = rot2(-self.theta_body) @ self.w_d_world[0:2, :] # Update desired wrench
        self.w_d = np.vstack((w_d_lin_body, self.w_d_world[2, :])) # Wrench in body frame

    def wrench_cb(self, msg: Float32MultiArray):
        """Process Wrench Desired Intake, which would be from the OverController module"""
        self.w_d_world = np.array(msg.data, dtype=float).reshape(3,1)

        # update w_d real quick (?)
        w_d_lin_body = rot2(-self.theta_body) @ self.w_d_world[0:2, :] # Update desired wrench
        self.w_d = np.vstack((w_d_lin_body, self.w_d_world[2, :])) # Wrench in body frame
        self.control_step()


    #endregion

    # -------- Main loop --------
    def control_step(self):
        print("\n\n---------- Starting Control Step ----------")
        print(f"Using W_d = {self.w_d_world}, Rotation = {self.theta_body}")
        # print(f"Using W_d_body = {self.w_d}")
        if self.boundary is None:
            return

        # Build PSO cost with current boundary
        cost = PSOCostWrapper(
            self.bx, self.by,
            self.w_d, self.theta, self.f_mag,
            self.lam, self.lam_min, self.lam_max, self.pso_iters
        )

        lb = np.array(self.get_parameter("lb").value, dtype=float)
        ub = np.array(self.get_parameter("ub").value, dtype=float)

        optimizer = GlobalBestPSO(
            n_particles=int(self.get_parameter("pso_swarm").value),
            dimensions=2,
            options={"c1": 1.3, "c2": 1.3, "w": 0.6},
            bounds=(lb, ub),
        )
        b_cost, (s_opt, d_opt) = optimizer.optimize(cost, iters=self.pso_iters, verbose=False)
        # Test the best point
        cost_dbg, dbg = self.debug_solve_at(s_opt, d_opt)
        print(f"SOLUTION STATS:                     ")
        # print(f"Body Angle: {self.com}, {self.theta_body}")
        # print(f"Desired Wrench World: {self.w_d_world}")
        for dbk, dbv in dbg.items():
            print(dbk, dbv)

        # Compute/contact points
        self.s1_opt = float(s_opt)
        self.s2_opt = float((s_opt + d_opt) % 1.0)
        p1 = np.array([float(self.bx(self.s1_opt)), float(self.by(self.s1_opt))])
        p2 = np.array([float(self.bx(self.s2_opt)), float(self.by(self.s2_opt))])

        # Publish push points
        msg_pts = Float32MultiArray()
        msg_pts.data = [p1[0], p1[1], p2[0], p2[1]]
        self.pub_push.publish(msg_pts)
        # print(f"Got Contacts at {1000*p1_world[0]:.0f}, {1000*p1_world[1]:.0f} "
                        #   f"and {1000*p2_world[0]:.0f}, {1000*p2_world[1]:.0f} mm")

        # ---- Adapt μ using e = v_hat - ṽ and central diff Jacobian ----
        # Model placeholder for v_hat(μ): diag(μ) * w_d
        # Needs a better mapping for twist contribution given wrench, given pusher-body mu.
        def v_hat(mu_vec, w_hat):
            return np.asarray(mu_vec).reshape(3,) * w_hat.reshape(3,)

        vhat = v_hat(self.mu_base, dbg["w_hat"])
        # e = vhat - self.v_meas  # e = v̂ - ṽ
        print(f"Got Vhat = {vhat}")
        # print(f"Got Vmeas = {self.v_meas}")
        # print(f"Error Twist = {e}")

        # Find required gripper twist to track object (body frame)
        v_lin = vhat[0:2]
        omega = vhat[2]
        v_p1_rot = omega*np.array([-p1[1], p1[0]]) # 90 degree rot
        v_p2_rot = omega*np.array([-p2[1], p2[0]]) # 90 degree rot
        print(f"Got v_p1_rot = {v_p1_rot}")
        print(f"Got v_p2_rot = {v_p2_rot}")
        v_p1 = v_lin + v_p1_rot
        v_p2 = v_lin + v_p2_rot
        v_g_lin = (v_p1 + v_p2) * 0.5
        v_g = np.array([v_g_lin[0], v_g_lin[1], omega])
        print(f"Got v_g = {v_g}")

        msg_twist = TwistStamped()
        msg_twist.header.stamp = self.get_clock().now().to_msg()
        msg_twist.header.frame_id = "body"
        msg_twist.twist.linear.x = v_lin[0]
        msg_twist.twist.linear.y = v_lin[1]
        msg_twist.twist.linear.z = 0.0
        msg_twist.twist.angular.x = 0.0
        msg_twist.twist.angular.y = 0.0
        msg_twist.twist.angular.z = omega
        self.gripper_twist.publish(msg_twist)

        msg_w_d = Float32MultiArray()
        msg_w_d.data = [float(self.w_d_world[0]),float(self.w_d_world[1]),float(self.w_d_world[2])]
        self.pub_desired.publish(msg_w_d) # World frame

        # eps = float(self.get_parameter("fd_eps").value)
        # J = np.zeros((3, 3))
        # for i in range(3):
        #     mu_f = self.mu.copy(); mu_f[i] += eps
        #     mu_b = self.mu.copy(); mu_b[i] -= eps
        #     J[:, i] = (v_hat(mu_f) - v_hat(mu_b)) / (2.0 * eps)

        # grad = 2.0 * (J.T @ e)  # ∂||e||^2/∂μ
        # eta = float(self.get_parameter("eta").value)
        # self.mu = self.mu - eta * grad
        # print(f"Got Mu = {self.mu}")

        # α_ub = (v̂^T ṽ)/||v̂||^2
        # denom = float(np.dot(self.v_meas, self.v_meas)) + 1e-12
        # self.alpha_ub = float(np.dot(vhat, self.v_meas) / denom)
        # print(f"Got Alpha_ub = {self.alpha_ub}")

        # Publish μ and α_ub
        # msg_mu = Float32MultiArray()
        # msg_mu.data = [float(self.mu[0]), float(self.mu[1]), float(self.mu[2]), float(self.alpha_ub)]
        # self.pub_mu.publish(msg_mu)

    def debug_solve_at(self, s, d):
        cost, dbg = pso_wrench_cost_python(
            s, d, self.bx, self.by, self.w_d,
            self.theta, self.f_mag, self.lam,
            self.lam_min, self.lam_max
        )
        return cost, dbg

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