# =================================================================
# PROJECT: 6-Axis Robot Trajectory Planner
#
# DESCRIPTION:
#   Calculates and plans collision-free trajectories on an XY plane
#   for a 6-axis robotic arm modeled with DH parameters.
#   Two planners are implemented: a reward-guided midpoint planner
#   and an RRT* planner for comparison.
#
# ROBOT PARAMETERS:
#   Link      a      d      alpha
#      1      0    0.1       pi/2
#      2   -0.4      0          0
#      3   -0.4      0          0
#      4      0    0.1       pi/2
#      5      0    0.1      -pi/2
#      6      0    0.1          0
# =================================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================
# Build the Robot Model
# ============================

class Robot:
    """6-axis serial manipulator defined by DH parameters.
    Parameters
        ----------
        a            : list[float]  Link lengths (translation along x_i).
        alpha        : list[float]   Link twists (rotation about x_i), in radians.
        d            : list[float]   Link offsets (translation along z_{i-1}).
        theta_offset : list[float]   Constant angular offsets added to each joint
                                      variable (e.g. to align the zero-configuration
                                      with a physical home pose).
        joint_limits : list[tuple]  lower, upper) radian limits per joint.
                                      Defaults to [-pi, pi] for every joint.
    """

    def __init__(self, a, alpha, d, theta_offset=None, joint_limits=None):
        # Store DH parameters as arrays
        self.a     = np.asarray(a,     dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.d     = np.asarray(d,     dtype=float)
        self.n     = len(self.a)

        if theta_offset is None:
            theta_offset = np.zeros(self.n)
        self.theta_offset = np.asarray(theta_offset, dtype=float)

        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi)] * self.n
        self.joint_limits = np.asarray(joint_limits, dtype=float)

    def _dh_transform(self, a, alpha, d, theta):
        """        
        Computes the 4x4 homogeneous DH transformation matrix for a single
        link, following the standard (modified) DH convention:

            T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

        This maps frame {i-1} to frame {i}.

        Parameters:
        a     : float  Link length.
        alpha : float  Link twist (rad).
        d     : float  Link offset.
        theta : float  Joint angle (rad).

        Returns:
        T : np.ndarray, shape (4, 4) Homogeneous transform."""
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)

        # Standard DH convention: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
        T = np.array([
            [ ct,  -st*ca,  st*sa,  a*ct ],
            [ st,   ct*ca, -ct*sa,  a*st ],
            [  0,      sa,     ca,     d ],
            [  0,       0,      0,     1 ],
        ], dtype=float)

        return T

    def _clip_to_limits(self, q):
        """
        Clamps every joint angle in q to its configured [lower, upper] limit,
        preventing the IK solver from driving the robot into mechanical stops.

        Parameters:
        q : np.ndarray  Joint configuration vector, shape (n,).

        Returns:
        np.ndarray  Clipped joint configuration.
        """
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def forward_kinematics(self, q):
        """
        Computes the end-effector pose (position + orientation) for a given
        joint configuration by chaining the individual DH transforms from
        the base frame to the tool frame:

            T_0_6 = T_0_1 * T_1_2 * ... * T_5_6

        Parameters:
        q : array-like, shape (n,)  Joint angles in radians.

        Returns:
        T : np.ndarray, shape (4, 4)  End-effector homogeneous transform.
            [:3, 3]  → position (x, y, z).
            [:3, :3] → rotation matrix.
        """
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        for i in range(self.n):
            theta_i = q[i] + self.theta_offset[i]
            T = T @ self._dh_transform(self.a[i], self.alpha[i], self.d[i], theta_i)
        return T

    def jacobian_numeric(self, q, eps=1e-6):
        """
        Estimates the 6xn geometric Jacobian at configuration q using central-
        difference finite differences. The top 3 rows are the translational
        Jacobian (∂p/∂q_i) and the bottom 3 rows are the rotational Jacobian
        approximated from the skew-symmetric part of R_dot * R^T.

        Parameters:
        q   : array-like, shape (n,) - Current joint configuration.
        eps : float - Finite-difference step size.

        Returns:
        J : np.ndarray, shape (6, n) - Geometric Jacobian.
        """
        T0 = self.forward_kinematics(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        J  = np.zeros((6, self.n))

        for i in range(self.n):
            dq    = np.zeros(self.n)
            dq[i] = eps
            T_eps = self.forward_kinematics(q + dq)
            J[:3, i] = (T_eps[:3, 3] - p0) / eps

            R_dot = (T_eps[:3, :3] - R0) / eps
            skew  = R_dot @ R0.T
            J[3:, i] = np.array([
                skew[2, 1] - skew[1, 2],
                skew[0, 2] - skew[2, 0],
                skew[1, 0] - skew[0, 1],
            ]) * 0.5

        return J

    def inverse_kinematics(self, T_targ, q_init, tol_pos=1e-3, max_iters=200):
        """
        Damped least-squares IK (position only).
        Uses J^T(JJ^T + lambdaI)^-1 to avoid singularity blow-up.
        Returns (q, success).
        """
        q       = q_init.copy()
        pt_targ = T_targ[:3, 3]

        for _ in range(max_iters):
            pt_curr = self.forward_kinematics(q)[:3, 3]
            e_p     = pt_targ - pt_curr

            if np.linalg.norm(e_p) < tol_pos:
                return self._clip_to_limits(q), True

            J     = self.jacobian_numeric(q)[:3, :]  # position rows only
            lamda = 1e-3                              # damping factor
            J_inv = J.T @ np.linalg.inv(J @ J.T + lamda * np.eye(3))
            q     = self._clip_to_limits(q + J_inv @ e_p)

        return q, False


# ============================
# Build the Environment
# ============================

class SphereObstacle:
    """Spherical obstacle defined by center and radius."""

    def __init__(self, center, radius):
        self.center = np.asarray(center)
        self.radius = radius


class Environment:
    """Workspace obstacle collection with clearance query."""

    def __init__(self):
        self.obstacles = []

    def add_sphere(self, center, radius):
        self.obstacles.append(SphereObstacle(center, radius))

    def get_min_dist(self, point):
        """Signed clearance from the nearest obstacle surface. Positive = safe."""
        if not self.obstacles:
            return np.inf
        return min(np.linalg.norm(point[:3] - obs.center) - obs.radius
                   for obs in self.obstacles)


# ============================
# Build the Trajectory Planner
# ============================

class MidpointRewardPlanner:
    """
    Greedy local planner that discretises the straight-line path to goal into
    waypoints, then steers around obstacles using a reward-scored deviation step.

    Bug fixed from original: _get_deviation_point had an inverted collision check
    (was skipping safe candidates instead of unsafe ones). Also increased sample
    count to 200 and added a deterministic fallback grid for stuck cases.
    """

    def __init__(self, robot, env, step_size, safe_dist):
        """
        Parameters:
        robot     : Robot       - The kinematic model.
        env       : Environment - Obstacle collection.
        step_size : float       - Nominal distance between consecutive waypoints (m).
        safe_dist : float       - Minimum clearance from any obstacle surface (m).
        """
        self.robot     = robot
        self.env       = env
        self.step_size = step_size
        self.dist_safe = safe_dist

        # Goal weight guides direction; obstacle weight enforces the safety margin.
        self.weight_goal = 1.0
        self.weight_obs  = 50.0

    def _generate_midpoints(self, pt_start, pt_goal):
        """Linear interpolation from pt_start to pt_goal at step_size resolution.
        Parameters:
        pt_start : np.ndarray - Starting 3-D position.
        pt_goal  : np.ndarray - Goal 3-D position.

        Returns:
        list[np.ndarray] - Ordered waypoints from start+step to goal."""
        dist    = np.linalg.norm(pt_goal - pt_start)
        n_steps = int(np.ceil(dist / self.step_size))
        if n_steps == 0:
            return []
        return [pt_start + (pt_goal - pt_start) * i / n_steps
                for i in range(1, n_steps + 1)]

    def _get_deviation_point(self, pt_curr, pt_goal):
        """
        Samples 200 random XY directions and picks the one maximising
        R = w_goal * cos(θ_goal) + w_obs * clearance, subject to clearance >= dist_safe.
        Falls back to a 16-direction grid sweep if random sampling finds nothing.
        Parameters:
        pt_curr : np.ndarray - Current end-effector position.
        pt_goal : np.ndarray - Goal position.

        Returns:
        np.ndarray - Best safe deviation target in 3-D space.
        """
        vec_goal    = pt_goal - pt_curr
        vec_goal   /= (np.linalg.norm(vec_goal) + 1e-9)
        escape_step = max(self.step_size, self.dist_safe * 2.5)

        best_score = -np.inf
        pt_best    = None

        # Stochastic phase
        for _ in range(200):
            rand_dir    = np.random.randn(3)
            rand_dir[2] = 0.0
            rand_dir   /= (np.linalg.norm(rand_dir) + 1e-9)

            pt_cand  = pt_curr + rand_dir * escape_step
            min_dist = self.env.get_min_dist(pt_cand)

            if min_dist < self.dist_safe:  # skip unsafe candidates
                continue

            reward = self.weight_goal * np.dot(rand_dir, vec_goal) + self.weight_obs * min_dist
            if reward > best_score:
                best_score = reward
                pt_best    = pt_cand

        if pt_best is not None:
            return pt_best

        # Deterministic fallback: sweep 16 evenly spaced angles at increasing radii
        for radius in [escape_step, escape_step * 1.5, escape_step * 2.0]:
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                pt_cand   = pt_curr + direction * radius
                if self.env.get_min_dist(pt_cand) >= self.dist_safe:
                    return pt_cand

        return pt_curr  # completely stuck — planner will stall gracefully

    def _create_T(self, p):
        """Build a position-only IK target (identity orientation).
        Parameters:
        p : array-like, shape (3,) - Desired end-effector position.

        Returns:
        np.ndarray, shape (4, 4) - Position-only homogeneous transform."""
        T        = np.eye(4)
        T[:3, 3] = p
        return T

    def plan(self, q_start, pt_goal, max_steps=500):
        """
        Main planning loop. Steps through waypoints toward the goal; when a waypoint
        is inside the safety margin, reroutes via _get_deviation_point and replans.
        Parameters
        q_start   : np.ndarray, shape (n,) - Initial joint configuration.
        pt_goal   : np.ndarray, shape (3,) - Goal position in world frame.
        max_steps : int - Safety cap on planning iterations.
        """
        q_path    = [q_start]
        q_curr    = q_start.copy()
        pt_curr   = self.robot.forward_kinematics(q_curr)[:3, 3]
        midpoints = self._generate_midpoints(pt_curr, pt_goal)

        steps = 0
        while midpoints and steps < max_steps:
            steps  += 1
            pt_targ = midpoints[0]
            pt_curr = self.robot.forward_kinematics(q_curr)[:3, 3]

            if np.linalg.norm(pt_curr - pt_goal) < 0.02:
                break

            dist_curr = self.env.get_min_dist(pt_curr)
            dist_targ = self.env.get_min_dist(pt_targ)

            if dist_curr < self.dist_safe or dist_targ < self.dist_safe:
                safe_targ = self._get_deviation_point(pt_curr, pt_goal)

                # If completely stuck, try a small random nudge before giving up
                if np.linalg.norm(safe_targ - pt_curr) < 1e-5:
                    for _ in range(10):
                        nudge    = pt_curr + np.random.randn(3) * 0.03
                        nudge[2] = pt_curr[2]  # keep Z fixed
                        if self.env.get_min_dist(nudge) >= self.dist_safe:
                            safe_targ = nudge
                            break

                q_next, success = self.robot.inverse_kinematics(self._create_T(safe_targ), q_curr)
                if success:
                    q_curr    = q_next
                    q_path.append(q_curr)
                    midpoints = self._generate_midpoints(safe_targ, pt_goal)
                else:
                    midpoints.pop(0)  # skip if IK can't reach deviation point

            else:
                q_next, success = self.robot.inverse_kinematics(self._create_T(pt_targ), q_curr)
                if success:
                    q_curr = q_next
                    q_path.append(q_curr)
                midpoints.pop(0)  # advance regardless of IK result

        return q_path


# ============================
# Visualization
# ============================

def visualize_result(robot, env, q_path, pt_start, pt_goal, safe_margin):
    """Top-down XY path plot with obstacles, safe margin rings, and clearance stats."""
    path_points   = np.array([robot.forward_kinematics(q)[:3, 3] for q in q_path])
    clearances    = [env.get_min_dist(p) for p in path_points]
    min_clearance = min(clearances)

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, obs in enumerate(env.obstacles):
        ax.add_patch(plt.Circle(obs.center[:2], obs.radius,
                                color='r', alpha=0.5,
                                label='Obstacle' if i == 0 else None))
        ax.add_patch(plt.Circle(obs.center[:2], obs.radius + safe_margin,
                                color='r', fill=False, linestyle='--',
                                label='Safe Margin' if i == 0 else None))

    ax.plot([pt_start[0], pt_goal[0]], [pt_start[1], pt_goal[1]],
            'k--', alpha=0.3, label='Straight Line')
    ax.plot(path_points[:, 0], path_points[:, 1], 'b.-', label='Robot Path', linewidth=1.5)
    ax.plot(pt_start[0], pt_start[1], 'go', markersize=10, label='Start')
    ax.plot(pt_goal[0],  pt_goal[1],  'rx', markersize=10, label='Goal')

    ax.set_aspect('equal')
    all_x = list(path_points[:, 0]) + [pt_start[0], pt_goal[0]] + [o.center[0] for o in env.obstacles]
    all_y = list(path_points[:, 1]) + [pt_start[1], pt_goal[1]] + [o.center[1] for o in env.obstacles]
    mg = 0.15
    ax.set_xlim(min(all_x) - mg, max(all_x) + mg)
    ax.set_ylim(min(all_y) - mg, max(all_y) + mg)

    safe_str = ' PASS' if min_clearance >= safe_margin else ' VIOLATION'
    ax.set_title(
        f"Midpoint Planner — Obstacle Avoidance\n"
        f"Steps: {len(q_path)}  |  Min clearance: {min_clearance*100:.1f} cm  |  {safe_str}"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_joint_angles(robot, q_path):
    """Per-joint angle plots with red shading outside configured limits."""
    q_path   = np.array(q_path)
    n_joints = robot.n
    steps    = np.arange(len(q_path))

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 14), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        q_i         = q_path[:, i]
        lo, hi      = robot.joint_limits[i]

        ax.axhspan(hi,   100, color='red', alpha=0.15, label='Violation Zone' if i == 0 else None)
        ax.axhspan(-100, lo,  color='red', alpha=0.15)
        ax.axhline(lo, color='red', linestyle='--', linewidth=1.5)
        ax.axhline(hi, color='red', linestyle='--', linewidth=1.5)
        ax.plot(steps, q_i, label=f'Joint {i+1}', linewidth=2, color='steelblue')
        ax.set_ylabel(f'q{i+1} (rad)')
        ax.set_ylim(min(np.min(q_i), lo) - 0.5, max(np.max(q_i), hi) + 0.5)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Simulation Step')
    fig.suptitle('Joint Trajectories & Safety Limits', fontsize=13)
    plt.tight_layout()
    plt.show()


# =================================================================
# Evaluation Framework
# =================================================================

class PlannerMetrics:
    """Stores outcome metrics and the joint path for one (planner, scenario) run."""

    def __init__(self, planner_name, scenario_name,
                 success, goal_dist, steps, min_clearance,
                 safety_ok, path_length, q_path=None):
        self.planner_name  = planner_name
        self.scenario_name = scenario_name
        self.success       = success
        self.goal_dist     = goal_dist
        self.steps         = steps
        self.min_clearance = min_clearance
        self.safety_ok     = safety_ok
        self.path_length   = path_length
        self.q_path        = q_path if q_path is not None else []

    def __repr__(self):
        status = " PASS" if self.success and self.safety_ok else " FAIL"
        return (f"[{status}] {self.planner_name:<22} | {self.scenario_name:<25} | "
                f"goal_dist={self.goal_dist*100:5.1f}cm | "
                f"steps={self.steps:4d} | "
                f"clearance={self.min_clearance*100:5.1f}cm | "
                f"path_len={self.path_length:.3f}m")


class TestScenario:
    """Packages an obstacle environment, start config, goal, and pass/fail thresholds."""

    def __init__(self, name, env, q_start, pt_goal,
                 safe_dist=0.025, goal_tol=0.03, description=""):
        self.name        = name
        self.env         = env
        self.q_start     = np.asarray(q_start, dtype=float)
        self.pt_goal     = np.asarray(pt_goal,  dtype=float)
        self.safe_dist   = safe_dist
        self.goal_tol    = goal_tol
        self.description = description


class PlannerEvaluator:
    """
    Runs a set of TestScenarios against registered planners and collects metrics.

    Usage:
        evaluator = PlannerEvaluator(robot)
        evaluator.add_scenario(s)
        evaluator.add_planner('Name', factory_fn)
        evaluator.run()
        evaluator.print_report()
        evaluator.plot_report(save_dir='evidence')
    """

    def __init__(self, robot):
        self.robot     = robot
        self.scenarios = []
        self.planners  = []  # list of (name, factory)
        self.results   = []  # list of PlannerMetrics

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def add_planner(self, name, factory):
        """factory signature: factory(robot, env, safe_dist, step_size) -> planner"""
        self.planners.append((name, factory))

    def _evaluate_path(self, q_path, pt_goal, env, safe_dist, goal_tol):
        """Computes all scalar metrics from a returned joint path."""
        pts       = np.array([self.robot.forward_kinematics(q)[:3, 3] for q in q_path])
        goal_dist = float(np.linalg.norm(pts[-1] - pt_goal))
        min_clr   = float(min(env.get_min_dist(p) for p in pts))
        arc_len   = float(sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts) - 1)))
        return goal_dist < goal_tol, goal_dist, len(q_path), min_clr, min_clr >= safe_dist, arc_len

    def run(self, max_steps=500, step_size=0.04):
        """Run all planner/scenario combinations and cache results.
         Parameters:
        max_steps : int   - Planning step budget passed to each planner.
        step_size : float - Waypoint resolution passed to each planner."""
        self.results.clear()
        n_total = len(self.scenarios) * len(self.planners)
        idx = 0
        for scenario in self.scenarios:
            for planner_name, factory in self.planners:
                idx += 1
                print(f"  [{idx:2d}/{n_total}] Running {planner_name:<22} on {scenario.name} ...",
                      end=' ', flush=True)
                planner = factory(self.robot, scenario.env, scenario.safe_dist, step_size)
                q_path  = planner.plan(scenario.q_start, scenario.pt_goal, max_steps=max_steps)
                success, goal_dist, steps, min_clr, safety_ok, arc_len = \
                    self._evaluate_path(q_path, scenario.pt_goal,
                                        scenario.env, scenario.safe_dist, scenario.goal_tol)
                print(f"{' PASS' if success and safety_ok else ' FAIL'}"
                      f"  goal={goal_dist*100:.1f}cm  clr={min_clr*100:.1f}cm")
                self.results.append(PlannerMetrics(
                    planner_name=planner_name, scenario_name=scenario.name,
                    success=success, goal_dist=goal_dist, steps=steps,
                    min_clearance=min_clr, safety_ok=safety_ok,
                    path_length=arc_len, q_path=list(q_path),
                ))

    def print_report(self):
        """Print results table and pass-rate summary."""
        W = 110
        print("\n" + "="*W)
        print(f"{'PLANNER EVALUATION RESULTS':^{W}}")
        print("="*W)
        for r in self.results:
            print(r)
        print("="*W)

        print("\nPASS RATE SUMMARY (success AND safety):")
        for pname, _ in self.planners:
            subset = [r for r in self.results if r.planner_name == pname]
            passed = sum(1 for r in subset if r.success and r.safety_ok)
            bar    = "█" * passed + "░" * (len(subset) - passed)
            print(f"  {pname:<22}: {passed}/{len(subset)}  [{bar}]")
        print()

    def plot_comparison(self, save_dir=None):
        """3-panel bar chart comparing goal distance, clearance, and path length across scenarios.
        Parameters
        ----------
        save_dir : str | None - If provided, saves the figure to
                                <save_dir>/00_metric_comparison.png.
        """
        if len(self.planners) < 2:
            print("[PlannerEvaluator] Need ≥2 planners to plot comparison.")
            return

        import os
        scenario_names = [s.name for s in self.scenarios]
        n_s    = len(scenario_names)
        n_p    = len(self.planners)
        x      = np.arange(n_s)
        w      = 0.8 / n_p
        colors = ['steelblue', 'darkorange', 'seagreen', 'purple']

        fig, axes = plt.subplots(3, 1, figsize=(max(14, n_s * 2.2), 12), sharex=True)

        metrics_info = [
            ("Goal Distance (cm)",        lambda r: r.goal_dist * 100,  "Lower is better. Dashed = 3 cm goal tolerance."),
            ("Min Obstacle Clearance (cm)", lambda r: r.min_clearance * 100, "Higher is better. Red = 2.5 cm safety threshold."),
            ("Path Arc Length (m)",        lambda r: r.path_length,     "Lower is better. High values indicate looping."),
        ]

        for ax, (ylabel, getter, note) in zip(axes, metrics_info):
            for pi, (pname, _) in enumerate(self.planners):
                vals = [getter(next((r for r in self.results
                                    if r.planner_name == pname and r.scenario_name == sname), None))
                        if any(r.planner_name == pname and r.scenario_name == sname
                               for r in self.results) else 0.0
                        for sname in scenario_names]
                bars = ax.bar(x + pi * w, vals, w * 0.9,
                              label=pname, color=colors[pi % len(colors)],
                              alpha=0.85, edgecolor='white', linewidth=0.5)
                for bar, v in zip(bars, vals):
                    if abs(v) > 0.05:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                                f"{v:.1f}", ha='center', va='bottom', fontsize=6.5, color='#333333')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(note, fontsize=8, color='#555555', style='italic', pad=2)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(axis='y', linestyle=':', alpha=0.55)

        axes[0].axhline(y=3.0, color='green',   linestyle='--', linewidth=1.2, label='Goal tol (3 cm)')
        axes[0].legend(fontsize=8, loc='upper right')
        axes[1].axhline(y=2.5, color='red',     linestyle='--', linewidth=1.4, label='Safety (2.5 cm)')
        axes[1].axhline(y=0.0, color='darkred', linestyle='-',  linewidth=0.8, alpha=0.5)
        axes[1].legend(fontsize=8, loc='upper right')

        axes[-1].set_xticks(x + w * (n_p - 1) / 2)
        axes[-1].set_xticklabels(scenario_names, rotation=28, ha='right', fontsize=9)
        fig.suptitle("Planner Comparison Across All Scenarios", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, "00_metric_comparison.png")
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  [saved] {fpath}")

        plt.show()
        plt.close(fig)

    def plot_paths(self, scenario, save_dir=None):
        """Side-by-side Cartesian path plots for all planners on a given scenario.
        Parameters
        scenario : TestScenario - The scenario whose paths to visualise.
        save_dir : str | None   - If provided, saves to
                                  <save_dir>/<scenario.name>_paths.png.
        """
        
        import os

        n_p       = len(self.planners)
        fig, axes = plt.subplots(1, n_p, figsize=(8 * n_p, 8))
        if n_p == 1:
            axes = [axes]

        pt_start = self.robot.forward_kinematics(scenario.q_start)[:3, 3]

        for ax, (pname, _) in zip(axes, self.planners):
            result = next((r for r in self.results
                           if r.planner_name == pname and r.scenario_name == scenario.name), None)

            if result is None or len(result.q_path) == 0:
                ax.text(0.5, 0.5, f"No cached path\nfor {pname}",
                        ha='center', va='center', transform=ax.transAxes)
                continue

            pts     = np.array([self.robot.forward_kinematics(q)[:3, 3] for q in result.q_path])
            min_clr = result.min_clearance

            for i, obs in enumerate(scenario.env.obstacles):
                ax.add_patch(plt.Circle(obs.center[:2], obs.radius,
                                        color='tomato', alpha=0.55,
                                        label='Obstacle' if i == 0 else None, zorder=2))
                ax.add_patch(plt.Circle(obs.center[:2], obs.radius + scenario.safe_dist,
                                        color='tomato', fill=False, linestyle='--', linewidth=1.2,
                                        label='Safe margin' if i == 0 else None, zorder=2))

            ax.plot([pt_start[0], scenario.pt_goal[0]],
                    [pt_start[1], scenario.pt_goal[1]],
                    'k--', alpha=0.2, linewidth=1.2, label='Straight line', zorder=1)

            # Colour path segments by clearance: blue=safe, yellow=close, red=violation
            for j in range(len(pts) - 1):
                seg_clr = scenario.env.get_min_dist(pts[j])
                col = '#cc0000' if seg_clr < 0 else ('#e6a800' if seg_clr < scenario.safe_dist else '#1a73e8')
                ax.plot(pts[j:j+2, 0], pts[j:j+2, 1], color=col, linewidth=2.0, zorder=3)
            ax.plot([], [], color='#1a73e8', linewidth=2, label='Safe')
            ax.plot([], [], color='#e6a800', linewidth=2, label='Near limit')
            ax.plot([], [], color='#cc0000', linewidth=2, label='Violation')

            ax.plot(pt_start[0], pt_start[1], 'go', markersize=10, label='Start', zorder=5)
            ax.plot(scenario.pt_goal[0], scenario.pt_goal[1], 'r*', markersize=12, label='Goal', zorder=5)

            all_x = list(pts[:, 0]) + [pt_start[0], scenario.pt_goal[0]] + [o.center[0] for o in scenario.env.obstacles]
            all_y = list(pts[:, 1]) + [pt_start[1], scenario.pt_goal[1]] + [o.center[1] for o in scenario.env.obstacles]
            mg = 0.18
            ax.set_xlim(min(all_x) - mg, max(all_x) + mg)
            ax.set_ylim(min(all_y) - mg, max(all_y) + mg)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.45, zorder=0)
            ax.legend(fontsize=7.5, loc='upper left')

            ok = "✓ PASS" if result.success and result.safety_ok else "✗ FAIL"
            ax.set_title(
                f"{pname}  [{ok}]\n"
                f"goal={result.goal_dist*100:.1f}cm | clr={min_clr*100:.1f}cm | "
                f"len={result.path_length:.2f}m | steps={result.steps}",
                fontsize=8.5, pad=6
            )
            ax.set_xlabel("X (m)", fontsize=9)
            ax.set_ylabel("Y (m)", fontsize=9)

        fig.suptitle(f"Path Comparison — {scenario.name}\n{scenario.description}",
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, f"{scenario.name}_paths.png")
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  [saved] {fpath}")

        plt.show()
        plt.close(fig)

    def plot_joint_angles(self, scenario, save_dir=None):
        """Joint angle trajectories per planner for a given scenario, with limit shading.
        Parameters
        ----------
        scenario : TestScenario - The scenario to visualise.
        save_dir : str | None   - If provided, saves to
                                  <save_dir>/<scenario.name>_joints.png.
        """
        import os

        n_p = len(self.planners)
        fig, outer_axes = plt.subplots(self.robot.n, n_p,
                                       figsize=(8 * n_p, 2.2 * self.robot.n),
                                       sharex='col')
        if n_p == 1:
            outer_axes = outer_axes.reshape(-1, 1)
        if self.robot.n == 1:
            outer_axes = outer_axes.reshape(1, -1)

        for pi, (pname, _) in enumerate(self.planners):
            result = next((r for r in self.results
                           if r.planner_name == pname and r.scenario_name == scenario.name), None)

            if result is None or len(result.q_path) < 2:
                outer_axes[0, pi].text(0.5, 0.5, f"No data for {pname}",
                                       ha='center', va='center',
                                       transform=outer_axes[0, pi].transAxes)
                continue

            q_arr  = np.array(result.q_path)
            steps  = np.arange(len(q_arr))
            outer_axes[0, pi].set_title(pname, fontsize=10, fontweight='bold')

            for ji in range(self.robot.n):
                ax     = outer_axes[ji, pi]
                q_i    = q_arr[:, ji]
                lo, hi = self.robot.joint_limits[ji]

                ax.axhspan(hi,      hi + 10, color='red', alpha=0.12)
                ax.axhspan(lo - 10, lo,      color='red', alpha=0.12)
                ax.axhline(lo, color='red', linestyle='--', linewidth=1.0)
                ax.axhline(hi, color='red', linestyle='--', linewidth=1.0)
                ax.plot(steps, q_i, color='steelblue', linewidth=1.6, label=f'q{ji+1}')
                ax.set_ylabel(f'q{ji+1} (rad)', fontsize=8)
                ax.grid(True, linestyle=':', alpha=0.5)
                ax.set_ylim(min(np.min(q_i), lo) - 0.4, max(np.max(q_i), hi) + 0.4)
                if pi == 0:
                    ax.legend(fontsize=7, loc='upper right')
                if ji == self.robot.n - 1:
                    ax.set_xlabel('Step', fontsize=8)

        fig.suptitle(f"Joint Trajectories — {scenario.name}", fontsize=11, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, f"{scenario.name}_joints.png")
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  [saved] {fpath}")

        plt.show()
        plt.close(fig)

    def plot_report(self, save_dir="evidence"):
        """Generates and saves all figures: metric comparison + per-scenario path/joint plots.
        Parameters
        save_dir : str - Directory to write PNG evidence files (default: "evidence").
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[plot_report] Writing evidence to '{save_dir}/'  "
              f"({len(self.scenarios)} scenarios, {[p[0] for p in self.planners]})")

        self.plot_comparison(save_dir=save_dir)
        for scenario in self.scenarios:
            print(f"  Plotting scenario '{scenario.name}' ...")
            self.plot_paths(scenario, save_dir=save_dir)
            self.plot_joint_angles(scenario, save_dir=save_dir)

        print(f"\n[plot_report] Done. Files in '{save_dir}/':")
        for fname in sorted(os.listdir(save_dir)):
            print(f"  {fname}")


# =================================================================
# RRT* Cartesian Planner
# =================================================================

class RRTStarPlanner:
    """
    RRT* planner in Cartesian XY space. Addresses the structural failure modes of
    MidpointRewardPlanner (U-traps, full barriers, goal-adjacent obstacles) by
    sampling the global workspace rather than scoring local steps.

    Builds a tree from pt_start, grows it by steering toward random samples,
    does RRT* parent selection and rewiring to minimise path cost, then
    extracts + shortcut-smooths the best path and converts it to joints via IK.
    """

    def __init__(self, robot, env, safe_dist, step_size,
                 max_iter=2000, goal_tol=0.03, rewire_radius=0.15, goal_bias=0.10):
        """
        Parameters:
        robot         : Robot       - Kinematic model.
        env           : Environment - Obstacle set.
        safe_dist     : float       - Minimum clearance from obstacles (m).
        step_size     : float       - Max extension per RRT iteration (m).
        max_iter      : int         - Maximum tree growth iterations.
        goal_tol      : float       - Distance to declare goal reached (m).
        rewire_radius : float       - Neighbourhood radius for RRT* rewiring (m).
        goal_bias     : float       - Probability of sampling the goal directly.
        """
        self.robot         = robot
        self.env           = env
        self.safe_dist     = safe_dist
        self.step_size     = step_size
        self.max_iter      = max_iter
        self.goal_tol      = goal_tol
        self.rewire_radius = rewire_radius
        self.goal_bias     = goal_bias

        # Workspace bounds — generous to cover full robot reach
        self.bounds = np.array([[-1.0, 1.0],
                                 [-1.0, 1.0],
                                 [ 0.0, 0.0]])

    def _make_node(self, pt, parent_idx, cost):
        return {'pt': pt, 'parent': parent_idx, 'cost': cost}

    def _nearest(self, pt):
        """Index of the closest node in the tree to pt."""
        pts = np.array([n['pt'] for n in self._nodes])
        return int(np.argmin(np.linalg.norm(pts - pt, axis=1)))

    def _near(self, pt):
        """Indices of all nodes within rewire_radius of pt."""
        pts   = np.array([n['pt'] for n in self._nodes])
        dists = np.linalg.norm(pts - pt, axis=1)
        return [i for i, d in enumerate(dists) if d <= self.rewire_radius]

    def _steer(self, pt_from, pt_to):
        """Move from pt_from toward pt_to by at most step_size."""
        diff = pt_to - pt_from
        d    = np.linalg.norm(diff)
        if d < 1e-9:
            return pt_from.copy()
        return pt_from + diff / d * min(d, self.step_size)

    def _collision_free(self, pt_a, pt_b, n_checks=8):
        """True if the segment pt_a to pt_b clears all obstacles by at least safe_dist."""
        for t in np.linspace(0, 1, n_checks):
            if self.env.get_min_dist(pt_a + t * (pt_b - pt_a)) < self.safe_dist:
                return False
        return True

    def _sample(self, pt_goal):
        """Random XY sample, biased toward pt_goal with probability goal_bias."""
        if np.random.rand() < self.goal_bias:
            return pt_goal.copy()
        return np.array([
            np.random.uniform(self.bounds[0, 0], self.bounds[0, 1]),
            np.random.uniform(self.bounds[1, 0], self.bounds[1, 1]),
            0.0
        ])

    def _extract_path(self, goal_idx):
        """Walk parent pointers from goal_idx back to root, return ordered waypoints."""
        path, idx = [], goal_idx
        while idx is not None:
            path.append(self._nodes[idx]['pt'])
            idx = self._nodes[idx]['parent']
        path.reverse()
        return path

    def _shortcut_smooth(self, path, n_attempts=200):
        """Randomly try to skip waypoints with a direct collision-free edge."""
        path = list(path)
        for _ in range(n_attempts):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self._collision_free(path[i], path[j], n_checks=12):
                path = path[:i+1] + path[j:]
        return path

    def _path_to_joints(self, cart_path, q_init):
        """Convert Cartesian waypoints to joint path via sequential warm-started IK."""
        q_path = [q_init.copy()]
        q_curr = q_init.copy()
        for pt in cart_path[1:]:
            T_targ        = np.eye(4)
            T_targ[:3, 3] = pt
            q_next, ok    = self.robot.inverse_kinematics(T_targ, q_curr)
            if ok:
                q_curr = q_next
                q_path.append(q_curr)
        return q_path

    def plan(self, q_start, pt_goal, max_steps=None):
        """
        Grow an RRT* tree from q_start toward pt_goal, extract + smooth the best
        Cartesian path, then convert to joints. Falls back to nearest node if
        goal region is never reached.
        Parameters:
        q_start   : np.ndarray - Start joint configuration.
        pt_goal   : np.ndarray - Goal Cartesian position.
        max_steps : int|None   - Overrides self.max_iter if provided (for
                                 compatibility with PlannerEvaluator).

        Returns
        list[np.ndarray] - Joint configurations from start to (near) goal.
        """
        max_iter = max_steps if max_steps is not None else self.max_iter

        pt_start      = self.robot.forward_kinematics(q_start)[:3, 3].copy()
        pt_start[2]   = 0.0
        pt_goal_xy    = pt_goal.copy()
        pt_goal_xy[2] = 0.0

        self._nodes    = [self._make_node(pt_start, parent_idx=None, cost=0.0)]
        best_goal_idx  = None
        best_goal_cost = np.inf

        for _ in range(max_iter):
            pt_rand  = self._sample(pt_goal_xy)
            idx_near = self._nearest(pt_rand)
            pt_near  = self._nodes[idx_near]['pt']
            pt_new   = self._steer(pt_near, pt_rand)

            if self.env.get_min_dist(pt_new) < self.safe_dist:
                continue
            if not self._collision_free(pt_near, pt_new):
                continue

            # RRT* parent selection: pick cheapest collision-free parent in neighbourhood
            near_idxs   = self._near(pt_new)
            best_parent = idx_near
            best_cost   = self._nodes[idx_near]['cost'] + np.linalg.norm(pt_new - pt_near)

            for ni in near_idxs:
                c = self._nodes[ni]['cost'] + np.linalg.norm(pt_new - self._nodes[ni]['pt'])
                if c < best_cost and self._collision_free(self._nodes[ni]['pt'], pt_new):
                    best_cost, best_parent = c, ni

            new_idx = len(self._nodes)
            self._nodes.append(self._make_node(pt_new, best_parent, best_cost))

            # Rewire: reparent neighbours if going through pt_new is cheaper
            for ni in near_idxs:
                rewired = best_cost + np.linalg.norm(self._nodes[ni]['pt'] - pt_new)
                if rewired < self._nodes[ni]['cost'] and self._collision_free(pt_new, self._nodes[ni]['pt']):
                    self._nodes[ni]['parent'] = new_idx
                    self._nodes[ni]['cost']   = rewired

            d_goal = np.linalg.norm(pt_new - pt_goal_xy)
            if d_goal < self.goal_tol and best_cost < best_goal_cost:
                best_goal_idx, best_goal_cost = new_idx, best_cost

        # Fallback: use closest node if goal region was never reached
        if best_goal_idx is None:
            pts = np.array([n['pt'] for n in self._nodes])
            best_goal_idx = int(np.argmin(np.linalg.norm(pts - pt_goal_xy, axis=1)))

        cart_path = self._shortcut_smooth(self._extract_path(best_goal_idx))
        return self._path_to_joints(cart_path, q_start)


# =================================================================
# Scenario Suite & Robot Builder
# =================================================================

def build_robot():
    """Construct the 6-axis robot from the assignment DH parameters.
    Returns
    -------
    Robot - Configured 6-axis manipulator.
    """
    a     = [ 0.0, -0.4, -0.4,       0.0,       0.0, 0.0]
    d     = [ 0.1,  0.0,  0.0,       0.1,       0.1, 0.1]
    alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
    return Robot(a=a, alpha=alpha, d=d)


def build_scenario_suite(robot):
    """
    Seven test scenarios targeting the known failure modes of MidpointRewardPlanner:
    Scenarios:
    1. Baseline         - Original assignment environment (sanity check).
    2. Narrow corridor  - Two obstacles forming a tight passage.
    3. U-Trap           - Horseshoe of three obstacles; greedy planner enters
                          and cannot reverse out.
    4. Wall / barrier   - Linear wall of five spheres perpendicular to the
                          goal direction; forces a lateral detour.
    5. Dense field      - Eight randomly placed obstacles; tests general
                          robustness and path quality.
    6. Goal adjacent    - Goal placed just outside an obstacle; MRP replanning
                          loop keeps steering into the obstacle.
    7. Start in margin  - Start EE position already inside the safe-distance
                          bubble; tests recovery from infeasible initial state.

    Parameters:
        robot : Robot - Used to compute the FK start position.

    Returns:
        list[TestScenario]
    """
    q_start  = np.array([0.0, -0.3, 0.6, 0.0, 0.0, 0.0])
    pt_start = robot.forward_kinematics(q_start)[:3, 3]
    scenarios = []

    # 1. Baseline: original assignment environment
    e = Environment()
    e.add_sphere([0.2,  0.0, 0.0], 0.05)
    e.add_sphere([0.0,  0.2, 0.0], 0.05)
    e.add_sphere([0.2,  0.2, 0.0], 0.05)
    e.add_sphere([0.4, -0.2, 0.0], 0.05)
    scenarios.append(TestScenario("1_baseline", e, q_start, [0.5, 0.0, 0.0],
                                  description="Original assignment environment."))

    # 2. Narrow corridor: tight gap between two large obstacles
    e = Environment()
    e.add_sphere([0.1,  0.12, 0.0], 0.08)
    e.add_sphere([0.1, -0.12, 0.0], 0.08)
    e.add_sphere([0.3,  0.10, 0.0], 0.05)
    scenarios.append(TestScenario("2_narrow_corridor", e, q_start, [0.5, 0.0, 0.0],
                                  description="Tight passage between two large obstacles."))

    # 3. U-trap: horseshoe; greedy planner enters and can't escape
    e = Environment()
    e.add_sphere([0.3,  0.15, 0.0], 0.07)
    e.add_sphere([0.3, -0.15, 0.0], 0.07)
    e.add_sphere([0.45, 0.0,  0.0], 0.07)
    scenarios.append(TestScenario("3_u_trap", e, q_start, [0.5, 0.0, 0.0],
                                  description="Horseshoe — greedy planner enters and cannot escape."))

    # 4. Wall barrier: linear wall of spheres, forces a lateral bypass
    e = Environment()
    for y in np.linspace(-0.28, 0.28, 5):
        e.add_sphere([0.1, y, 0.0], 0.055)
    scenarios.append(TestScenario("4_wall_barrier", e, q_start, [0.5, 0.0, 0.0],
                                  description="Linear wall across direct path; requires lateral bypass."))

    # 5. Dense field: 8 random obstacles
    e = Environment()
    rng = np.random.default_rng(42)
    for _ in range(8):
        e.add_sphere([rng.uniform(-0.2, 0.6), rng.uniform(-0.4, 0.4), 0.0], 0.06)
    scenarios.append(TestScenario("5_dense_field", e, q_start, [0.5, 0.0, 0.0],
                                  description="8 randomly placed obstacles; general robustness."))

    # 6. Goal adjacent: goal just outside obstacle, causes MRP replanning loop
    e = Environment()
    e.add_sphere([0.43, 0.0, 0.0], 0.06)
    scenarios.append(TestScenario("6_goal_adjacent", e, q_start, [0.5, 0.0, 0.0],
                                  description="Goal just outside obstacle; replanning trap for MRP."))

    # 7. Start in margin: EE starts inside the safe bubble
    e = Environment()
    e.add_sphere([pt_start[0]+0.03, pt_start[1], 0.0], 0.05)
    e.add_sphere([0.2, 0.0, 0.0], 0.05)
    scenarios.append(TestScenario("7_start_in_margin", e, q_start, [0.5, 0.0, 0.0],
                                  description="Start EE inside obstacle safe zone; tests initial recovery."))

    return scenarios


# ============================
# Main
# ============================

def main():
    np.random.seed(0)
    robot = build_robot()

    # --- Part 1: Original assignment scenario ---
    env = Environment()
    env.add_sphere(center=[0.2,  0.0, 0.0], radius=0.05)
    env.add_sphere(center=[0.0,  0.2, 0.0], radius=0.05)
    env.add_sphere(center=[0.2,  0.2, 0.0], radius=0.05)
    env.add_sphere(center=[0.4, -0.2, 0.0], radius=0.05)

    q_start  = np.array([0.0, -0.3, 0.6, 0.0, 0.0, 0.0])
    pt_start = robot.forward_kinematics(q_start)[:3, 3]
    pt_goal  = np.array([0.5, 0.0, 0.0])

    print(f"Start (FK): {pt_start.round(4)}   Goal: {pt_goal}")

    safe_margin = 0.025
    planner     = MidpointRewardPlanner(robot, env, step_size=0.04, safe_dist=safe_margin)
    q_path      = planner.plan(q_start, pt_goal)

    path_points  = np.array([robot.forward_kinematics(q)[:3, 3] for q in q_path])
    clearances   = [env.get_min_dist(p) for p in path_points]
    dist_to_goal = np.linalg.norm(path_points[-1] - pt_goal)

    print(f"MidpointRewardPlanner -  baseline:")
    print(f"  Steps: {len(q_path)}  |  Dist to goal: {dist_to_goal*100:.2f} cm  |  "
          f"Min clearance: {min(clearances)*100:.2f} cm  "
          f"({'SAFE' if min(clearances) >= safe_margin else 'VIOLATION'})")

    visualize_result(robot, env, q_path, pt_start, pt_goal, safe_margin)
    visualize_joint_angles(robot, q_path)

    # --- Part 2: Evaluation suite: MidpointReward vs RRT* ---
    print("\n" + "="*105)
    print("RUNNING EVALUATION SUITE ...")
    print("="*105)

    def mrp_factory(robot, env, safe_dist, step_size):
        return MidpointRewardPlanner(robot, env, step_size=step_size, safe_dist=safe_dist)

    def rrt_factory(robot, env, safe_dist, step_size):
        return RRTStarPlanner(robot, env, safe_dist=safe_dist, step_size=step_size,
                              max_iter=2000, goal_tol=0.03, rewire_radius=0.15, goal_bias=0.10)

    evaluator = PlannerEvaluator(robot)
    for s in build_scenario_suite(robot):
        evaluator.add_scenario(s)
    evaluator.add_planner("MidpointReward", mrp_factory)
    evaluator.add_planner("RRT*",           rrt_factory)

    evaluator.run(max_steps=500, step_size=0.04)
    evaluator.print_report()
    evaluator.plot_report(save_dir="evidence")


if __name__ == "__main__":
    main()