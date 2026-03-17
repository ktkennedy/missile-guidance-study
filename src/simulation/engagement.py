"""End-to-end missile-target engagement simulation.

Integrates: Target -> Seeker -> Filter -> Guidance -> Autopilot -> Actuator -> Dynamics -> INS feedback

Signal chain (per timestep):
    1. Target.get_state(t)          -> r_T, v_T, a_T
    2. SeekerModel.measure(...)     -> LOS angles/rates (noisy)
    3. AlphaBetaFilter.update(...)  -> smoothed LOS rates
    4. ProportionalNavigation.compute_pitch_yaw(...) -> a_pitch, a_yaw
    5. Missile3DOF.derivatives(...)  integrated with RK4
    6. Termination check
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from ..dynamics.missile_3dof import Missile3DOF
from ..dynamics.missile_6dof import Missile6DOF
from ..guidance.proportional_navigation import ProportionalNavigation, compute_los_geometry
from ..control.autopilot import TwoLoopAutopilot
from ..control.three_loop_autopilot import ThreeLoopAutopilot
from ..control.actuator import FinActuator
from ..sensors.seeker import SeekerModel
from ..sensors.kalman_filter import AlphaBetaFilter
from ..sensors.imu import IMUModel
from ..sensors.gps_model import GPSModel
from ..sensors.aided_navigation import AidedNavigationSystem
from ..sensors.strapdown_ins import StrapdownINS
from ..sensors.nav_kalman_filter import NavKalmanFilter
from ..targets.target_models import Target
from ..utils.coordinate_transforms import (
    euler_to_quat, quat_to_dcm, quat_normalize, wind_angles,
    dcm_to_euler,
)


class _ThreeAxisAutopilot:
    """Minimal three-axis autopilot wrapper used by _run_6dof.

    Delegates pitch and yaw to separate ThreeLoopAutopilot instances and
    provides a simple roll-rate damper via proportional gain K_roll.
    """

    def __init__(self, pitch_ap: ThreeLoopAutopilot, yaw_ap: ThreeLoopAutopilot,
                 K_roll: float = 0.1) -> None:
        self.pitch_ap = pitch_ap
        self.yaw_ap = yaw_ap
        self.K_roll = K_roll

    def compute(self, a_cmd_pitch, a_cmd_yaw,
                a_meas_pitch, a_meas_yaw,
                q_meas, r_meas, p_meas, dt):
        """Return (delta_e, delta_r, delta_a) fin commands."""
        delta_e = self.pitch_ap.compute(a_cmd_pitch, a_meas_pitch, q_meas, dt)
        delta_r = self.yaw_ap.compute(a_cmd_yaw, a_meas_yaw, r_meas, dt)
        delta_a = -self.K_roll * p_meas
        return delta_e, delta_r, delta_a


@dataclass
class EngagementConfig:
    """Configuration for engagement simulation."""
    # Missile initial conditions
    missile_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 5000.0]))
    missile_speed: float = 300.0
    missile_heading: float = 0.0   # rad, measured clockwise from North (psi)
    missile_gamma: float = 0.0    # rad, flight-path angle (positive up)

    # Target configuration
    # target_pos: [x_north, y_east, alt] — altitude positive up, same convention as missile_pos
    # target_vel: [vx_north, vy_east, vz_up] — positive-up vertical component
    target_pos: np.ndarray = field(default_factory=lambda: np.array([10000.0, 0.0, 5000.0]))
    target_vel: np.ndarray = field(default_factory=lambda: np.array([-200.0, 0.0, 0.0]))
    target_maneuver: str = 'constant_velocity'
    target_params: Optional[Dict] = None

    # Guidance settings
    nav_constant: float = 4.0
    guidance_variant: str = 'APN'

    # Simulation settings
    dt: float = 0.001
    t_max: float = 30.0
    miss_threshold: float = 1.0   # m, hit declared when range < this
    diverge_steps: int = 50       # consecutive increasing-range steps to declare miss
    guidance_a_max: float = 400.0 # m/s², max guidance acceleration (~40g default)
    fidelity: str = '3dof'        # '3dof' or '6dof'

    # 6-DOF autopilot design parameters (used when fidelity='6dof')
    autopilot_omega: float = 20.0
    autopilot_zeta: float = 0.7
    autopilot_tau: float = 0.5
    K_roll: float = 0.1


@dataclass
class EngagementResult:
    """Results from engagement simulation."""
    t: np.ndarray               # [N] time stamps (s)
    missile_pos: np.ndarray     # [N, 3] missile positions (m)
    missile_vel: np.ndarray     # [N, 3] missile velocity vectors (m/s)
    target_pos: np.ndarray      # [N, 3] target positions (m)
    target_vel: np.ndarray      # [N, 3] target velocities (m/s)
    a_cmd: np.ndarray           # [N, 2] guidance commands [a_pitch, a_yaw] (m/s²)
    a_achieved: np.ndarray      # [N, 2] achieved (pitch, yaw) acceleration (m/s²)
    range_history: np.ndarray   # [N] range missile-to-target (m)
    los_rate: np.ndarray        # [N] LOS rate magnitude (rad/s)
    fin_deflection: np.ndarray  # [N] fin deflection from actuator (rad)
    miss_distance: float        # closest approach distance (m)
    time_of_flight: float       # total flight time until termination (s)
    hit: bool                   # True when miss < miss_threshold
    intercept_index: int        # index of closest approach in history arrays
    # 6-DOF fields (None for 3-DOF mode)
    states_6dof: Optional[np.ndarray] = None      # [N, 13] full 6-DOF state history
    quat_history: Optional[np.ndarray] = None      # [N, 4] quaternion history
    euler_history: Optional[np.ndarray] = None     # [N, 3] Euler angles [phi, theta, psi]
    fin_deflections_3ch: Optional[np.ndarray] = None  # [N, 3] (delta_e, delta_r, delta_a)
    fidelity: str = '3dof'                         # '3dof' or '6dof'


class EngagementSimulator:
    """Full-loop engagement simulator.

    Signal chain:
        Target -> Seeker -> Filter -> Guidance -> Autopilot -> Actuator -> 3-DOF Dynamics
                                                                           |
                                                                    INS/IMU feedback

    For the 3-DOF model, attitude dynamics are not modelled; the guidance
    acceleration commands are applied directly to the point-mass EOM.
    The autopilot and actuator are still exercised to produce a realistic
    fin-deflection record, but they do not affect the trajectory.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def run(self, config: EngagementConfig) -> EngagementResult:
        """Run engagement simulation with fidelity selection."""
        if config.fidelity == '6dof':
            return self._run_6dof(config)
        return self._run_3dof(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_vel(state: np.ndarray) -> np.ndarray:
        """Extract NED velocity vector from 3-DOF state [x, y, alt, V, gamma, psi]."""
        _, _, _, V, gamma, psi = state
        vx = V * np.cos(gamma) * np.cos(psi)
        vy = V * np.cos(gamma) * np.sin(psi)
        vz_ned = -V * np.sin(gamma)   # NED z is positive down; alt is positive up
        return np.array([vx, vy, vz_ned])

    @staticmethod
    def _state_to_pos(state: np.ndarray) -> np.ndarray:
        """Extract NED position [x, y, -alt] from 3-DOF state.

        The 3-DOF state stores altitude (positive up).  NED uses z positive
        down.  We return [x_north, y_east, z_down] = [x, y, -alt].
        """
        x, y, alt, _, _, _ = state
        return np.array([x, y, -alt])

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def _run_3dof(self, config: EngagementConfig) -> EngagementResult:
        """Run 3-DOF engagement simulation.

        Algorithm:
          1. Initialise all subsystems from config.
          2. Main RK4 loop:
             a. Get target state analytically.
             b. Seeker measures LOS (with noise and gimbal lag).
             c. AlphaBetaFilter smooths LOS rate measurements.
             d. Guidance computes (a_pitch, a_yaw) via compute_pitch_yaw().
             e. Autopilot + actuator compute fin deflection (logged but not
                fed back into 3-DOF dynamics — the 3-DOF model takes accel
                commands directly).
             f. RK4 integrates missile state.
             g. Detect termination: hit, ground, time-out, or diverging
                geometry (range increasing after initial decrease).
          3. Package all recorded histories into EngagementResult.

        Returns:
            EngagementResult dataclass with full engagement history.
        """
        cfg = config
        dt = float(cfg.dt)
        t_max = float(cfg.t_max)

        # ----------------------------------------------------------------
        # 1. Initialise subsystems
        # ----------------------------------------------------------------
        missile = Missile3DOF()

        guidance = ProportionalNavigation(
            N=cfg.nav_constant,
            variant=cfg.guidance_variant,
            a_max=cfg.guidance_a_max,
        )

        seeker = SeekerModel(
            angle_noise_std=0.003,
            rate_noise_std=0.003,
            gimbal_tau=0.02,
        )

        # Alpha-beta filter tracking [lam_dot_az, lam_dot_el]
        ab_filter = AlphaBetaFilter(alpha=0.3, beta=0.05, dim=2)

        autopilot = TwoLoopAutopilot()
        actuator = FinActuator()

        # Target model uses NED frame (z positive down).
        # EngagementConfig stores target_pos as [x_north, y_east, alt_up]
        # and target_vel as [vx_north, vy_east, vz_up].
        # Convert to NED by negating the z component.
        t_pos_cfg = np.array(cfg.target_pos, dtype=float)
        t_vel_cfg = np.array(cfg.target_vel, dtype=float)
        t_pos_ned = np.array([t_pos_cfg[0], t_pos_cfg[1], -t_pos_cfg[2]])
        t_vel_ned = np.array([t_vel_cfg[0], t_vel_cfg[1], -t_vel_cfg[2]])

        target = Target(
            r0=t_pos_ned,
            v0=t_vel_ned,
            maneuver_type=cfg.target_maneuver,
            maneuver_params=cfg.target_params,
        )

        # ----------------------------------------------------------------
        # 2. Build initial 3-DOF state: [x, y, alt, V, gamma, psi]
        # ----------------------------------------------------------------
        pos0 = np.array(cfg.missile_pos, dtype=float)
        state = np.array([
            pos0[0],           # x  (North)
            pos0[1],           # y  (East)
            pos0[2],           # alt (positive up)
            float(cfg.missile_speed),
            float(cfg.missile_gamma),
            float(cfg.missile_heading),
        ], dtype=float)

        # ----------------------------------------------------------------
        # 3. History buffers (record every 10 steps)
        # ----------------------------------------------------------------
        t_hist: List[float] = []
        mpos_hist: List[np.ndarray] = []
        mvel_hist: List[np.ndarray] = []
        tpos_hist: List[np.ndarray] = []
        tvel_hist: List[np.ndarray] = []
        acmd_hist: List[np.ndarray] = []
        aach_hist: List[np.ndarray] = []
        rng_hist: List[float] = []
        losrate_hist: List[float] = []
        fin_hist: List[float] = []

        # ----------------------------------------------------------------
        # 4. Main loop
        # ----------------------------------------------------------------
        t = 0.0
        step = 0
        prev_range = np.inf
        range_increasing_count = 0
        DIVERGE_STEPS = cfg.diverge_steps

        # Track closest approach
        min_range = np.inf
        min_range_index = 0
        record_index = 0   # index into recorded history

        a_pitch_cmd = 0.0
        a_yaw_cmd = 0.0
        fin_deflection = 0.0
        prev_gamma = float(cfg.missile_gamma)  # for q_proxy in autopilot
        a_achieved_pitch = 0.0  # 1st-order lag model for achieved acceleration
        a_achieved_yaw = 0.0

        # Cache first target acceleration for APN feedforward
        _, _, a_T_est = target.get_state(0.0)

        while t < t_max:
            # ---------- a. Target state ----------
            r_T, v_T, a_T = target.get_state(t)

            # ---------- b. Missile NED position/velocity from state ----------
            m_ned_pos = self._state_to_pos(state)   # [x, y, -alt]
            m_ned_vel = self._state_to_vel(state)

            # Actual range (NED)
            R_vec = r_T - m_ned_pos
            current_range = float(np.linalg.norm(R_vec))

            # ---------- c. Termination checks ----------
            # Hit check
            if current_range < cfg.miss_threshold:
                # Record this step
                _record(t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                        a_pitch_cmd, a_yaw_cmd, fin_deflection,
                        current_range, seeker, ab_filter,
                        t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                        acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist)
                record_index += 1
                break

            # Ground impact — record final state before breaking
            if state[2] < 0.0 and t > 0.0:
                _record(t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                        a_pitch_cmd, a_yaw_cmd, fin_deflection,
                        current_range, seeker, ab_filter,
                        t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                        acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist)
                record_index += 1
                break

            # Diverging geometry (missile flying away from target)
            if current_range > prev_range:
                range_increasing_count += 1
            else:
                range_increasing_count = 0
            if range_increasing_count >= DIVERGE_STEPS and t > 2.0:
                break

            prev_range = current_range

            # Track closest approach
            if current_range < min_range:
                min_range = current_range
                min_range_index = record_index

            # ---------- d. Seeker ----------
            meas = seeker.measure(m_ned_pos, m_ned_vel, r_T, v_T, dt)

            # ---------- e. Alpha-beta filter on LOS rates ----------
            z_rates = np.array([meas['lam_dot_az'], meas['lam_dot_el']])
            los_rate_filtered, _ = ab_filter.update(z_rates, dt)

            # ---------- f. Guidance ----------
            # Use filtered LOS rates from the seeker+filter chain instead of
            # true geometry.  This closes the realistic sensor loop:
            #   Seeker(noisy) -> AB filter -> Guidance
            # The true target position is still used for range/geometry, but
            # the LOS rates driving the PN law come from the filtered
            # measurements, making miss-distance statistics realistic.
            geo = compute_los_geometry(m_ned_pos, m_ned_vel, r_T, v_T)
            V_c = geo['V_c']
            lam_el = geo['lam_el']

            # Filtered LOS rates from the alpha-beta filter
            filtered_az_rate = los_rate_filtered[0] if ab_filter.initialized else 0.0
            filtered_el_rate = los_rate_filtered[1] if ab_filter.initialized else 0.0

            cos_el = float(np.cos(lam_el))
            effective_speed = max(V_c, 0.0) if cfg.guidance_variant != 'PPN' else float(np.linalg.norm(m_ned_vel))

            a_pitch_cmd = guidance.N * effective_speed * filtered_el_rate
            a_yaw_cmd = guidance.N * effective_speed * filtered_az_rate * cos_el

            # APN augmentation with target acceleration estimate
            if cfg.guidance_variant == 'APN':
                a_T_est = a_T  # true value (would be EKF estimate in full system)
                lam_az = geo['lam_az']
                sin_el = float(np.sin(lam_el))
                cos_az = float(np.cos(lam_az))
                sin_az = float(np.sin(lam_az))
                e_el = np.array([-sin_el * cos_az, -sin_el * sin_az, -cos_el])
                e_az = np.array([-sin_az, cos_az, 0.0])
                a_pitch_cmd += (guidance.N / 2.0) * float(np.dot(a_T_est, e_el))
                a_yaw_cmd += (guidance.N / 2.0) * float(np.dot(a_T_est, e_az))

            # Saturate combined magnitude
            a_mag = np.sqrt(a_pitch_cmd**2 + a_yaw_cmd**2)
            if a_mag > guidance.a_max:
                scale = guidance.a_max / a_mag
                a_pitch_cmd *= scale
                a_yaw_cmd *= scale

            # ---------- g. Autopilot + Actuator ----------
            # Model achieved acceleration as a first-order lag of the commanded
            # value (time constant ~0.05s, representative of airframe response).
            # This gives the autopilot a realistic tracking error to work with,
            # producing a meaningful fin-deflection record.
            tau_accel = 0.05  # airframe acceleration response time constant (s)
            alpha_lag = 1.0 - np.exp(-dt / tau_accel)
            a_achieved_pitch += alpha_lag * (a_pitch_cmd - a_achieved_pitch)
            a_achieved_yaw += alpha_lag * (a_yaw_cmd - a_achieved_yaw)
            q_proxy = (state[4] - prev_gamma) / dt if step > 0 else 0.0
            delta_cmd = autopilot.compute(
                a_cmd=a_pitch_cmd,
                a_measured=a_achieved_pitch,
                q_measured=q_proxy,
                dt=dt,
            )
            fin_deflection = actuator.update(delta_cmd, dt)

            # ---------- h. Record (every 10 steps) ----------
            if step % 10 == 0:
                _record(t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                        a_pitch_cmd, a_yaw_cmd, fin_deflection,
                        current_range, seeker, ab_filter,
                        t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                        acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist)
                record_index += 1

            # ---------- i. RK4 integration ----------
            def _deriv(tt, ss):
                return missile.derivatives(tt, ss, a_achieved_pitch, a_achieved_yaw)

            k1 = _deriv(t,            state)
            k2 = _deriv(t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = _deriv(t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = _deriv(t + dt,       state +       dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            prev_gamma = state[4]  # update for next step's q_proxy
            t += dt
            step += 1

        # ----------------------------------------------------------------
        # 5. Package results
        # ----------------------------------------------------------------
        if not t_hist:
            # Edge case: loop never recorded a step
            empty = np.empty((0, 3))
            return EngagementResult(
                t=np.array([]),
                missile_pos=empty,
                missile_vel=empty,
                target_pos=empty,
                target_vel=empty,
                a_cmd=np.empty((0, 2)),
                a_achieved=np.empty((0, 2)),
                range_history=np.array([]),
                los_rate=np.array([]),
                fin_deflection=np.array([]),
                miss_distance=float(current_range),
                time_of_flight=float(t),
                hit=False,
                intercept_index=0,
            )

        t_arr = np.array(t_hist)
        mpos_arr = np.array(mpos_hist)        # [N, 3]
        mvel_arr = np.array(mvel_hist)        # [N, 3]
        tpos_arr = np.array(tpos_hist)        # [N, 3]
        tvel_arr = np.array(tvel_hist)        # [N, 3]
        acmd_arr = np.array(acmd_hist)        # [N, 2]
        aach_arr = np.array(aach_hist)        # [N, 2]
        rng_arr = np.array(rng_hist)          # [N]
        losrate_arr = np.array(losrate_hist)  # [N]
        fin_arr = np.array(fin_hist)          # [N]

        # Closest approach in recorded data
        if len(rng_arr) > 0:
            min_range_index = int(np.argmin(rng_arr))
            miss_distance = float(rng_arr[min_range_index])
        else:
            miss_distance = min_range
            min_range_index = 0

        hit = miss_distance < cfg.miss_threshold

        return EngagementResult(
            t=t_arr,
            missile_pos=mpos_arr,
            missile_vel=mvel_arr,
            target_pos=tpos_arr,
            target_vel=tvel_arr,
            a_cmd=acmd_arr,
            a_achieved=aach_arr,
            range_history=rng_arr,
            los_rate=losrate_arr,
            fin_deflection=fin_arr,
            miss_distance=miss_distance,
            time_of_flight=float(t),
            hit=hit,
            intercept_index=min_range_index,
        )


    # ------------------------------------------------------------------
    # 6-DOF Engagement Loop
    # ------------------------------------------------------------------

    def _run_6dof(self, config: EngagementConfig) -> EngagementResult:
        """Run 6-DOF engagement simulation.

        Signal chain (per timestep):
            Target -> Seeker -> Filter -> Guidance -> ThreeAxisAutopilot
            -> 3x FinActuator -> Missile6DOF dynamics (RK4)
            -> IMU -> AidedNavigationSystem -> feedback

        The navigation system never receives truth state directly.
        IMU and GPS measurements are generated externally.
        """
        cfg = config
        dt = float(cfg.dt)
        t_max = float(cfg.t_max)

        # ----------------------------------------------------------------
        # 1. Initialise subsystems
        # ----------------------------------------------------------------
        missile = Missile6DOF()

        guidance = ProportionalNavigation(
            N=cfg.nav_constant,
            variant=cfg.guidance_variant,
            a_max=cfg.guidance_a_max,
        )

        seeker = SeekerModel(
            angle_noise_std=0.003,
            rate_noise_std=0.003,
            gimbal_tau=0.02,
        )

        ab_filter = AlphaBetaFilter(alpha=0.3, beta=0.05, dim=2)

        # Three-axis autopilot: pitch + yaw three-loop + roll damper
        # Compute dimensional stability derivatives from aero coefficients
        aero = missile.aero
        _, _, rho_0, _ = missile.atm.get_properties(max(float(cfg.missile_pos[2]), 0.0))
        q_bar_0 = 0.5 * rho_0 * cfg.missile_speed**2

        M_alpha_cb = aero.Cm_alpha * q_bar_0 * missile.S_ref * missile.d_ref / missile.Iyy
        M_delta_cb = aero.Cm_delta * q_bar_0 * missile.S_ref * missile.d_ref / missile.Iyy
        M_q_cb = aero.Cm_q * q_bar_0 * missile.S_ref * missile.d_ref**2 / (missile.Iyy * cfg.missile_speed)
        Z_alpha_cb = -(aero.CL_alpha * q_bar_0 * missile.S_ref) / (missile.mass_0 * cfg.missile_speed)
        # CL_delta not explicitly modeled; use a reasonable fraction of CL_alpha
        Z_delta_cb = Z_alpha_cb * 0.2  # typical: CL_delta ~ 0.2 * CL_alpha

        pitch_ap = ThreeLoopAutopilot(
            M_alpha=M_alpha_cb, M_q=M_q_cb, M_delta=M_delta_cb,
            Z_alpha=Z_alpha_cb, Z_delta=Z_delta_cb, V=cfg.missile_speed,
            omega=cfg.autopilot_omega, zeta=cfg.autopilot_zeta,
            tau=cfg.autopilot_tau,
        )
        yaw_ap = ThreeLoopAutopilot(
            M_alpha=M_alpha_cb, M_q=M_q_cb, M_delta=M_delta_cb,
            Z_alpha=Z_alpha_cb, Z_delta=Z_delta_cb, V=cfg.missile_speed,
            omega=cfg.autopilot_omega, zeta=cfg.autopilot_zeta,
            tau=cfg.autopilot_tau,
        )
        three_axis_ap = _ThreeAxisAutopilot(pitch_ap, yaw_ap, K_roll=cfg.K_roll)

        actuator_e = FinActuator()
        actuator_r = FinActuator()
        actuator_a = FinActuator()

        # IMU and navigation
        imu = IMUModel()
        gps = GPSModel(seed=42)

        # Target
        t_pos_cfg = np.array(cfg.target_pos, dtype=float)
        t_vel_cfg = np.array(cfg.target_vel, dtype=float)
        t_pos_ned = np.array([t_pos_cfg[0], t_pos_cfg[1], -t_pos_cfg[2]])
        t_vel_ned = np.array([t_vel_cfg[0], t_vel_cfg[1], -t_vel_cfg[2]])

        target = Target(
            r0=t_pos_ned,
            v0=t_vel_ned,
            maneuver_type=cfg.target_maneuver,
            maneuver_params=cfg.target_params,
        )

        # ----------------------------------------------------------------
        # 2. Build initial 6-DOF state [x,y,z, u,v,w, q0,q1,q2,q3, p,q,r]
        # ----------------------------------------------------------------
        pos0 = np.array(cfg.missile_pos, dtype=float)
        # NED position: z_ned = -altitude
        x0, y0 = pos0[0], pos0[1]
        z0_ned = -pos0[2]  # altitude positive up -> NED z positive down

        # Body velocity from speed, gamma, heading
        V0 = float(cfg.missile_speed)
        gamma0 = float(cfg.missile_gamma)
        psi0 = float(cfg.missile_heading)

        # NED velocity
        vx_ned = V0 * np.cos(gamma0) * np.cos(psi0)
        vy_ned = V0 * np.cos(gamma0) * np.sin(psi0)
        vz_ned = -V0 * np.sin(gamma0)

        # Initial attitude: align body x-axis with velocity vector
        theta0 = gamma0  # pitch angle: align body x-axis with velocity vector (alpha=0)
        phi0 = 0.0
        q0_quat = euler_to_quat(phi0, theta0, psi0)

        # Transform NED velocity to body frame
        C_nb = quat_to_dcm(q0_quat)
        vel_body = C_nb @ np.array([vx_ned, vy_ned, vz_ned])

        state = np.array([
            x0, y0, z0_ned,
            vel_body[0], vel_body[1], vel_body[2],
            q0_quat[0], q0_quat[1], q0_quat[2], q0_quat[3],
            0.0, 0.0, 0.0,  # p, q, r = 0 initially
        ], dtype=float)

        # Initialize aided navigation from initial conditions
        ins = StrapdownINS(
            initial_pos_ned=np.array([x0, y0, z0_ned]),
            initial_vel_ned=np.array([vx_ned, vy_ned, vz_ned]),
            initial_quat=q0_quat.copy(),
        )
        nav_filter = NavKalmanFilter(
            gyro_arw=0.001,        # rad/√s (tactical grade)
            accel_vrw=0.01,        # m/s/√s
            gyro_bias_std=0.001,   # rad/s
            accel_bias_std=0.01,   # m/s²
        )
        aided_nav = AidedNavigationSystem(ins, nav_filter)

        # ----------------------------------------------------------------
        # 3. History buffers
        # ----------------------------------------------------------------
        t_hist: List[float] = []
        mpos_hist: List[np.ndarray] = []
        mvel_hist: List[np.ndarray] = []
        tpos_hist: List[np.ndarray] = []
        tvel_hist: List[np.ndarray] = []
        acmd_hist: List[np.ndarray] = []
        aach_hist: List[np.ndarray] = []
        rng_hist: List[float] = []
        losrate_hist: List[float] = []
        fin_hist: List[float] = []
        state_hist: List[np.ndarray] = []
        quat_hist: List[np.ndarray] = []
        euler_hist: List[np.ndarray] = []
        fin3_hist: List[np.ndarray] = []

        # ----------------------------------------------------------------
        # 4. Main loop
        # ----------------------------------------------------------------
        t = 0.0
        step = 0
        prev_range = np.inf
        range_increasing_count = 0
        DIVERGE_STEPS = cfg.diverge_steps

        min_range = np.inf
        min_range_index = 0
        record_index = 0

        a_pitch_cmd = 0.0
        a_yaw_cmd = 0.0
        delta_e = 0.0
        delta_r = 0.0
        delta_a = 0.0

        while t < t_max:
            # ---------- a. Target state ----------
            r_T, v_T, a_T = target.get_state(t)

            # ---------- b. Missile NED position/velocity from 6-DOF state ----------
            m_ned_pos = state[0:3].copy()  # [x, y, z_ned]
            quat_curr = quat_normalize(state[6:10])
            C_nb_curr = quat_to_dcm(quat_curr)
            C_bn_curr = C_nb_curr.T
            vel_body_curr = state[3:6]
            m_ned_vel = C_bn_curr @ vel_body_curr

            # Actual range
            R_vec = r_T - m_ned_pos
            current_range = float(np.linalg.norm(R_vec))

            # ---------- c. Termination checks ----------
            if current_range < cfg.miss_threshold:
                if step % 10 == 0 or True:
                    self._record_6dof(
                        t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                        a_pitch_cmd, a_yaw_cmd, delta_e,
                        current_range, ab_filter, quat_curr,
                        delta_e, delta_r, delta_a,
                        t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                        acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist,
                        state_hist, quat_hist, euler_hist, fin3_hist)
                    record_index += 1
                break

            # Ground impact
            alt = -state[2]
            if alt < 0.0 and t > 0.0:
                self._record_6dof(
                    t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                    a_pitch_cmd, a_yaw_cmd, delta_e,
                    current_range, ab_filter, quat_curr,
                    delta_e, delta_r, delta_a,
                    t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                    acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist,
                    state_hist, quat_hist, euler_hist, fin3_hist)
                record_index += 1
                break

            # Diverging geometry
            if current_range > prev_range:
                range_increasing_count += 1
            else:
                range_increasing_count = 0
            if range_increasing_count >= DIVERGE_STEPS and t > 2.0:
                break

            prev_range = current_range

            if current_range < min_range:
                min_range = current_range
                min_range_index = record_index

            # ---------- d. Navigation feedback ----------
            # Generate IMU measurements from true state
            omega_true = state[10:13]  # [p, q, r] body angular rates
            # Specific force in body frame: total accel - gravity
            # Approximate from body velocities + gravity
            g_ned = np.array([0.0, 0.0, 9.80665])
            g_body = C_nb_curr @ g_ned
            V_total = np.linalg.norm(vel_body_curr)
            if V_total > 1.0:
                alpha_curr, beta_curr = wind_angles(vel_body_curr[0], vel_body_curr[1], vel_body_curr[2])
            else:
                alpha_curr, beta_curr = 0.0, 0.0

            # Specific force approximation from dynamics
            # f_specific = a_total - g = (F_aero + F_thrust) / m
            m_curr = missile.get_mass(t)
            T_curr = missile.get_thrust(t)
            _, _, rho_curr, _ = missile.atm.get_properties(max(float(-state[2]), 0.0))
            q_bar_curr = 0.5 * rho_curr * V_total**2
            forces_aero, _ = missile.aero.get_forces_moments(
                alpha_curr, beta_curr, V_total, q_bar_curr,
                state[10], state[11], state[12],
                delta_e, delta_r, delta_a
            )
            f_specific_body = np.array([
                (forces_aero[0] + T_curr) / m_curr,
                forces_aero[1] / m_curr,
                forces_aero[2] / m_curr,
            ])

            gyro_meas, accel_meas = imu.measure(omega_true, f_specific_body, dt)
            nav_state = aided_nav.propagate(gyro_meas, accel_meas, dt)

            # GPS update at low rate
            if gps.is_update_due(t):
                gps_pos, gps_vel = gps.measure(m_ned_pos, m_ned_vel)
                R_gps = np.diag([gps.pos_sigma**2]*3 + [gps.vel_sigma**2]*3)
                aided_nav.correct('gps_posvel', np.concatenate([gps_pos, gps_vel]), R_gps)

            # ---------- e. Seeker ----------
            # Use nav solution for missile position/velocity (not truth)
            nav_pos = nav_state.pos_ned
            nav_vel = nav_state.vel_ned
            meas = seeker.measure(nav_pos, nav_vel, r_T, v_T, dt)

            # ---------- f. Alpha-beta filter on LOS rates ----------
            z_rates = np.array([meas['lam_dot_az'], meas['lam_dot_el']])
            los_rate_filtered, _ = ab_filter.update(z_rates, dt)

            # ---------- g. Guidance ----------
            geo = compute_los_geometry(nav_pos, nav_vel, r_T, v_T)
            V_c = geo['V_c']
            lam_el = geo['lam_el']

            filtered_az_rate = los_rate_filtered[0] if ab_filter.initialized else 0.0
            filtered_el_rate = los_rate_filtered[1] if ab_filter.initialized else 0.0

            cos_el = float(np.cos(lam_el))
            effective_speed = max(V_c, 0.0) if cfg.guidance_variant != 'PPN' else float(np.linalg.norm(nav_vel))

            a_pitch_cmd = guidance.N * effective_speed * filtered_el_rate
            a_yaw_cmd = guidance.N * effective_speed * filtered_az_rate * cos_el

            if cfg.guidance_variant == 'APN':
                a_T_est = a_T
                lam_az = geo['lam_az']
                sin_el = float(np.sin(lam_el))
                cos_az = float(np.cos(lam_az))
                sin_az = float(np.sin(lam_az))
                e_el = np.array([-sin_el * cos_az, -sin_el * sin_az, -cos_el])
                e_az = np.array([-sin_az, cos_az, 0.0])
                a_pitch_cmd += (guidance.N / 2.0) * float(np.dot(a_T_est, e_el))
                a_yaw_cmd += (guidance.N / 2.0) * float(np.dot(a_T_est, e_az))

            a_mag = np.sqrt(a_pitch_cmd**2 + a_yaw_cmd**2)
            if a_mag > guidance.a_max:
                scale = guidance.a_max / a_mag
                a_pitch_cmd *= scale
                a_yaw_cmd *= scale

            # ---------- h. Autopilot ----------
            # The three-loop autopilot tracks body-frame SPECIFIC FORCE (the
            # plant model a_z = V*(Z_alpha*alpha + Z_delta*delta) is body-z
            # specific force, positive = downward).  PN guidance outputs
            # INERTIAL acceleration (positive = upward for pitch).
            #
            # Conversion:
            #   specific_force = inertial_accel - gravity
            #   pitch: body_z_sf = -a_pitch_inertial - g_body_z
            #   yaw:   body_y_sf = +a_yaw_inertial  - g_body_y
            #
            # Sign inversion for yaw: the yaw plant has Cn_beta > 0
            # (opposite to Cm_alpha < 0), so the steady-state response
            # from delta_r to a_y is inverted relative to delta_e → a_z.
            # Negate both command and measurement so the autopilot sees
            # the correct feedback polarity.
            g_body = C_nb_curr @ np.array([0.0, 0.0, 9.80665])
            a_cmd_pitch_sf = -a_pitch_cmd - g_body[2]
            a_cmd_yaw_sf = -(a_yaw_cmd - g_body[1])

            # Measurements: body-frame specific force from accelerometer
            a_meas_pitch_sf = accel_meas[2]    # body z specific force
            a_meas_yaw_sf = -accel_meas[1]     # negated body y (sign inversion)

            q_meas = gyro_meas[1]          # pitch rate
            r_meas = gyro_meas[2]          # yaw rate
            p_meas = gyro_meas[0]          # roll rate

            # Dynamic gain scheduling: update autopilot gains every 100 steps
            if step % 100 == 0:
                V_now = max(float(np.linalg.norm(vel_body_curr)), 1.0)
                alt_now = max(float(-state[2]), 0.0)
                _, _, rho_now, _ = missile.atm.get_properties(alt_now)
                q_bar_now = 0.5 * rho_now * V_now**2
                m_now = missile.get_mass(t)

                M_q_now = aero.Cm_q * q_bar_now * missile.S_ref * missile.d_ref**2 / (missile.Iyy * V_now)
                M_delta_now = aero.Cm_delta * q_bar_now * missile.S_ref * missile.d_ref / missile.Iyy
                Z_alpha_now = -(aero.CL_alpha * q_bar_now * missile.S_ref) / (m_now * V_now)

                omega_ap = cfg.autopilot_omega
                zeta_ap = cfg.autopilot_zeta
                tau_ap = cfg.autopilot_tau
                Kg_new = (-2 * zeta_ap * omega_ap - M_q_now) / M_delta_now
                K_omega_new = 0.1 * abs(Kg_new)
                KA_new = 1.0 / (tau_ap * abs(Z_alpha_now) * V_now) if abs(Z_alpha_now) > 1e-6 else pitch_ap.KA

                for _ap in [pitch_ap, yaw_ap]:
                    _ap.Kg = Kg_new
                    _ap.K_omega = K_omega_new
                    _ap.KA = KA_new

            delta_e_cmd, delta_r_cmd, delta_a_cmd = three_axis_ap.compute(
                a_cmd_pitch_sf, a_cmd_yaw_sf,
                a_meas_pitch_sf, a_meas_yaw_sf,
                q_meas, r_meas, p_meas,
                dt
            )

            # ---------- i. Actuators ----------
            delta_e = actuator_e.update(delta_e_cmd, dt)
            delta_r = actuator_r.update(delta_r_cmd, dt)
            delta_a = actuator_a.update(delta_a_cmd, dt)

            # ---------- j. Record (every 10 steps) ----------
            if step % 10 == 0:
                # Achieved inertial acceleration (PN convention: positive up/right)
                a_achieved_pitch = -(f_specific_body[2] + g_body[2])
                a_achieved_yaw = f_specific_body[1] + g_body[1]
                self._record_6dof(
                    t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                    a_pitch_cmd, a_yaw_cmd, delta_e,
                    current_range, ab_filter, quat_curr,
                    delta_e, delta_r, delta_a,
                    t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                    acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist,
                    state_hist, quat_hist, euler_hist, fin3_hist,
                    a_achieved_pitch=a_achieved_pitch, a_achieved_yaw=a_achieved_yaw)
                record_index += 1

            # ---------- k. RK4 integration ----------
            de, dr, da = delta_e, delta_r, delta_a

            k1 = missile.derivatives(t,            state,              de, dr, da)
            k2 = missile.derivatives(t + 0.5 * dt, state + 0.5*dt*k1, de, dr, da)
            k3 = missile.derivatives(t + 0.5 * dt, state + 0.5*dt*k2, de, dr, da)
            k4 = missile.derivatives(t + dt,       state +     dt*k3, de, dr, da)

            state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

            # Re-normalise quaternion
            state[6:10] = quat_normalize(state[6:10])

            t += dt
            step += 1

        # ----------------------------------------------------------------
        # 5. Package results
        # ----------------------------------------------------------------
        if not t_hist:
            empty3 = np.empty((0, 3))
            return EngagementResult(
                t=np.array([]), missile_pos=empty3, missile_vel=empty3,
                target_pos=empty3, target_vel=empty3,
                a_cmd=np.empty((0, 2)), a_achieved=np.empty((0, 2)),
                range_history=np.array([]), los_rate=np.array([]),
                fin_deflection=np.array([]),
                miss_distance=float(current_range), time_of_flight=float(t),
                hit=False, intercept_index=0,
                states_6dof=np.empty((0, 13)), quat_history=np.empty((0, 4)),
                euler_history=np.empty((0, 3)), fin_deflections_3ch=np.empty((0, 3)),
                fidelity='6dof',
            )

        t_arr = np.array(t_hist)
        rng_arr = np.array(rng_hist)

        if len(rng_arr) > 0:
            min_idx = int(np.argmin(rng_arr))
            miss_dist = float(rng_arr[min_idx])
        else:
            miss_dist = min_range
            min_idx = 0

        return EngagementResult(
            t=t_arr,
            missile_pos=np.array(mpos_hist),
            missile_vel=np.array(mvel_hist),
            target_pos=np.array(tpos_hist),
            target_vel=np.array(tvel_hist),
            a_cmd=np.array(acmd_hist),
            a_achieved=np.array(aach_hist),
            range_history=rng_arr,
            los_rate=np.array(losrate_hist),
            fin_deflection=np.array(fin_hist),
            miss_distance=miss_dist,
            time_of_flight=float(t),
            hit=miss_dist < cfg.miss_threshold,
            intercept_index=min_idx,
            states_6dof=np.array(state_hist),
            quat_history=np.array(quat_hist),
            euler_history=np.array(euler_hist),
            fin_deflections_3ch=np.array(fin3_hist),
            fidelity='6dof',
        )

    @staticmethod
    def _record_6dof(t, state, m_ned_pos, m_ned_vel, r_T, v_T,
                     a_pitch_cmd, a_yaw_cmd, fin_deflection,
                     current_range, ab_filter, quat_curr,
                     delta_e, delta_r, delta_a,
                     t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
                     acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist,
                     state_hist, quat_hist, euler_hist, fin3_hist,
                     a_achieved_pitch=None, a_achieved_yaw=None):
        """Record one sample for 6-DOF history."""
        from ..utils.coordinate_transforms import dcm_to_euler, quat_to_dcm, quat_normalize

        t_hist.append(t)
        mpos_hist.append(m_ned_pos.copy())
        mvel_hist.append(m_ned_vel.copy())
        tpos_hist.append(r_T.copy())
        tvel_hist.append(v_T.copy())
        acmd_hist.append(np.array([a_pitch_cmd, a_yaw_cmd]))

        if a_achieved_pitch is not None:
            aach_hist.append(np.array([a_achieved_pitch, a_achieved_yaw]))
        else:
            aach_hist.append(np.array([a_pitch_cmd, a_yaw_cmd]))

        rng_hist.append(current_range)

        if ab_filter.initialized:
            los_rate_mag = float(np.linalg.norm(ab_filter.x))
        else:
            los_rate_mag = 0.0
        losrate_hist.append(los_rate_mag)
        fin_hist.append(fin_deflection)

        # 6-DOF specific
        state_hist.append(state.copy())
        quat_hist.append(quat_curr.copy())
        C_nb = quat_to_dcm(quat_normalize(quat_curr))
        phi, theta, psi = dcm_to_euler(C_nb)
        euler_hist.append(np.array([phi, theta, psi]))
        fin3_hist.append(np.array([delta_e, delta_r, delta_a]))


# ------------------------------------------------------------------
# Module-level helper (avoids repeating long argument list)
# ------------------------------------------------------------------

def _record(t, state, m_ned_pos, m_ned_vel, r_T, v_T,
            a_pitch_cmd, a_yaw_cmd, fin_deflection,
            current_range, seeker, ab_filter,
            t_hist, mpos_hist, mvel_hist, tpos_hist, tvel_hist,
            acmd_hist, aach_hist, rng_hist, losrate_hist, fin_hist):
    """Append one sample to all history lists."""
    t_hist.append(t)
    mpos_hist.append(m_ned_pos.copy())
    mvel_hist.append(m_ned_vel.copy())
    tpos_hist.append(r_T.copy())
    tvel_hist.append(v_T.copy())
    acmd_hist.append(np.array([a_pitch_cmd, a_yaw_cmd]))
    aach_hist.append(np.array([a_pitch_cmd, a_yaw_cmd]))  # 3-DOF: achieved = commanded
    rng_hist.append(current_range)
    # LOS rate magnitude from filter state
    if ab_filter.initialized:
        los_rate_mag = float(np.linalg.norm(ab_filter.x))
    else:
        los_rate_mag = 0.0
    losrate_hist.append(los_rate_mag)
    fin_hist.append(fin_deflection)
