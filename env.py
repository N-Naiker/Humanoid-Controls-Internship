"""
Gymnasium RL environment for Franka trajectory tracking.

Architecture:
  Trajectory → [+ RL correction] → PD Controller → MuJoCo
                                        ↑
                                  Kalman Filter
                                        ↑
                                  Noisy sensor
                                        ↑
                                  True EE position

Uncertainty sources:
  1. Observation noise (5mm Gaussian on EE position)
  2. Control delay (1-step action delay = 20ms)
"""
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces


class KalmanFilter3D:
    """Kalman with process_noise=1e-6"""

    def __init__(self, dt, process_noise=1e-6, measurement_noise=0.005):
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        q = process_noise
        self.Q = np.diag([q, q, q, q*10, q*10, q*10])
        self.R = np.eye(3) * measurement_noise**2
        self.P = np.eye(6) * 0.01
        self.x = np.zeros(6)
        self.ready = False

    def reset(self, pos):
        self.x[:3] = pos.copy()
        self.x[3:] = 0.0
        self.P = np.eye(6) * 0.01
        self.ready = True

    def update(self, z):
        if not self.ready:
            self.reset(z)
            return z.copy(), np.zeros(3)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x[:3].copy(), self.x[3:].copy()


class FrankaTrackingEnv(gym.Env):
    """Franka trajectory tracking with RL task-space residual policy."""

    metadata = {"render_modes": []}

    def __init__(self, obs_noise_std=0.005, enable_delay=True,
                 ctrl_substeps=10, residual_scale=0.003):
        """
        Here:
            obs_noise_std: Gaussian noise std on EE position sensor (m)
            enable_delay: if True, apply 1-step action delay
            ctrl_substeps: simulation steps per RL step (10 * 2ms = 20ms)
            residual_scale: max RL correction in meters (3mm)
        """
        super().__init__()

        # ── Load simulation ──
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene_modified.xml")

        # ── Soften actuator gains to reduce physical jitter ──
        actuator_damping_scale = 2.0   
        actuator_stiffness_scale = 0.5  

        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] *= actuator_stiffness_scale
            self.model.actuator_biasprm[i, 1] *= actuator_stiffness_scale
            self.model.actuator_biasprm[i, 2] *= actuator_damping_scale

        self.data  = mujoco.MjData(self.model)
        self.dt_sim  = self.model.opt.timestep          
        self.ctrl_substeps = ctrl_substeps
        self.dt_ctrl = self.dt_sim * ctrl_substeps      

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.nv = self.model.nv
        self.q_home = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853])
        self.KP = 5
        self.KD = 30    


        # ── Trajectory parameters ──
        self.omega  = 2.0 * np.pi * 0.08
        self.radius = 0.2
        self.traj_x = 0.6
        self.traj_cz = 0.6

        # ── Uncertainty parameters ──
        self.obs_noise_std = obs_noise_std
        self.enable_delay  = enable_delay
        self.residual_scale = residual_scale

        # ── Kalman filter (tuned for in-loop use) ──
        self.kf = KalmanFilter3D(
            dt=self.dt_sim,   
            process_noise=1e-6,
            measurement_noise=obs_noise_std
        )

        # ── Episode config ──
        self.episode_duration = 13.0
        self.max_steps = int(self.episode_duration / self.dt_ctrl)  

        # ── Spaces ──
        # Observation: pos_err(3) + vel_err(3) + orient_err(3) + phase(2) + prev_action(3) = 14
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(14,), dtype=np.float32)

        # Action: 3D task-space correction, scaled to residual_scale
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32)

        # ── Internal state ──
        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(3)
        self.delayed_action = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.last_ee_filtered = np.zeros(3)
        self.last_vel_filtered = np.zeros(3)
        self.desired_orientation = None   
        self.last_orient_error = np.zeros(3)
        

    # ── Trajectory ──
    def _traj_pos(self, t):
        return np.array([self.traj_x,
                         self.radius * np.cos(self.omega * t),
                         self.traj_cz + self.radius * np.sin(self.omega * t)])

    def _traj_vel(self, t):
        return np.array([0.0,
                         -self.radius * self.omega * np.sin(self.omega * t),
                          self.radius * self.omega * np.cos(self.omega * t)])

    # ── Helpers ──
    def _ee_pos_true(self):
        """True EE position from MuJoCo (for reward computation only)."""
        return self.data.site_xpos[self.site_id].copy()
    
    def _ee_orientation(self):
        """Current EE orientation as 3x3 rotation matrix from MuJoCo."""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def _orientation_error(self):
        """
        Compute orientation error as 3D axis-angle vector.
        Small errors ≈ [ex, ey, ez] where magnitude = angle in radians.
        """
        R_cur = self._ee_orientation()
        R_des = self.desired_orientation

        # Error rotation: R_err = R_des^T @ R_cur
        R_err = R_des.T @ R_cur

        # Extract axis-angle from rotation matrix
        error = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) * 0.5
        return error

    def _pd_control(self, ee_pos_estimate, pos_target, vel_ff, ee_vel_estimate=None):
        """
        Task-space PD controller with orientation tracking.
        Uses full 6x7 Jacobian (3 position + 3 orientation).
        """
        pos_error = pos_target - ee_pos_estimate

        # Position: P term + velocity feedforward
        dx_pos = self.KP * pos_error + vel_ff * self.dt_sim
        if ee_vel_estimate is not None:
            vel_error = vel_ff - ee_vel_estimate
            dx_pos += self.KD * vel_error * self.dt_sim

        # Orientation: P term only 
        orient_error = self._orientation_error()
        KP_orient = 5   # orientation gain (lower than position)
        dx_orient = KP_orient * orient_error

        # Full 6D desired movement
        dx = np.concatenate([dx_pos, dx_orient])

        # Full 6x7 Jacobian
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J = np.vstack([jacp, jacr])   # 6x7

        # Damped pseudoinverse of 6x7 Jacobian
        damping = 1e-4
        JJT = J @ J.T + damping * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))

        # Null-space projection
        N = np.eye(self.nv) - J_pinv @ J

        return (self.data.qpos[:self.nv]
                + J_pinv @ dx
                + N @ (1.0 * (self.q_home - self.data.qpos[:self.nv])))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # Reset Kalman filter
        self.kf = KalmanFilter3D(
            dt=self.dt_sim, process_noise=1e-6,
            measurement_noise=self.obs_noise_std)

        # Use the home pose orientation
        self.desired_orientation = self._ee_orientation()

        # Settle to circle start 
        p0 = self._traj_pos(0.0)
        for _ in range(5000):
            ee_true = self._ee_pos_true()
            q_cmd = self._pd_control(ee_true, p0, np.zeros(3))
            self.data.ctrl[:] = q_cmd
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.last_orient_error = np.zeros(3)

        # Initialize KF with true position
        self.kf.reset(self._ee_pos_true())

        # Reset internal state
        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(3)
        self.delayed_action = np.zeros(3)
        self.prev_pos_error = np.zeros(3)

        self.last_ee_filtered = self._ee_pos_true()
        self.last_vel_filtered = np.zeros(3)

        return self._get_obs(), {}

    def step(self, action):
        # ── Scale RL action to task-space correction ──
        correction = action * self.residual_scale   # ±3mm max

        # ── Control delay ──
        if self.enable_delay:
            applied = self.delayed_action.copy()
            self.delayed_action = correction.copy()
        else:
            applied = correction.copy()

        # ── Get desired trajectory ──
        pos_des = self._traj_pos(self.sim_time)
        vel_des = self._traj_vel(self.sim_time)

        # RL adjusts the PD controller's target position
        pos_target = pos_des + applied

        # Update visual marker
        self.data.mocap_pos[0] = pos_des

        # ── Inner loop: sensor + KF + controller at 2ms rate ──
        # The RL correction (pos_target) stays constant for all 10 substeps.
        # But the sensor, filter, and controller update every 2ms.
        for _ in range(self.ctrl_substeps):
            ee_true = self._ee_pos_true()
            noise = self.np_random.normal(0, self.obs_noise_std, 3)
            ee_noisy = ee_true + noise
            ee_filtered, vel_filtered = self.kf.update(ee_noisy)

            q_cmd = self._pd_control(ee_filtered, pos_target, vel_des, vel_filtered)
            self.data.ctrl[:] = q_cmd
            mujoco.mj_step(self.model, self.data)

            # Save last filtered estimates for the observation
            self.last_ee_filtered = ee_filtered.copy()
            self.last_vel_filtered = vel_filtered.copy()
            self.last_orient_error = self._orientation_error()

        self.sim_time += self.dt_ctrl
        self.step_count += 1

        # ── Compute reward using TRUE position ──
        ee_true_after = self._ee_pos_true()
        pos_error = pos_des - ee_true_after
        error_norm = np.linalg.norm(pos_error)

        # Reward design:
        # - Linear error penalty 
        # - Smoothness: penalize changes in error 
        # - Action regularization: discourage unnecessary corrections
        # - Action rate: penalize rapid action changes 
        r_track  = -error_norm * 1000.0                     
        delta_err = pos_error - self.prev_pos_error
        r_smooth = -5000.0 * np.linalg.norm(delta_err)**2     
        r_act    = -0.05 * np.linalg.norm(action)**2            
        delta_act = action - self.prev_action
        r_rate   = -0.5 * np.linalg.norm(delta_act)**2          

        # Orientation tracking
        orient_error = self._orientation_error()
        orient_error_deg = np.linalg.norm(orient_error) * 180 / np.pi
        r_orient = -orient_error_deg * 10.0    

        reward = r_track + r_smooth + r_act + r_rate + r_orient

        self.last_orient_error = orient_error.copy()

        # Update state
        self.prev_pos_error = pos_error.copy()
        self.prev_action = action.copy()

        truncated = self.step_count >= self.max_steps
        info = {
            "tracking_error_mm": error_norm * 1000,
            "orient_error_deg": orient_error_deg,
            "ee_pos": ee_true_after.copy(),
            "des_pos": pos_des.copy(),
        }

        return self._get_obs(), reward, False, truncated, info

    def _get_obs(self):
        pos_des = self._traj_pos(self.sim_time)
        vel_des = self._traj_vel(self.sim_time)

        pos_err = pos_des - self.last_ee_filtered
        vel_err = vel_des - self.last_vel_filtered

        phase = self.omega * self.sim_time

        return np.concatenate([
            pos_err * 100.0,                       
            vel_err * 10.0,                        
            self.last_orient_error * 10.0,         
            [np.sin(phase), np.cos(phase)],         
            self.prev_action,                       
        ]).astype(np.float32)