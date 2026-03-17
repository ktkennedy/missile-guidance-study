"""
6-DOF rigid-body missile model with quaternion attitude representation.

State vector (13 states):
  Index  Symbol   Description
  -----  ------   -----------
  0      x        NED North position (m)
  1      y        NED East  position (m)
  2      z        NED Down  position (m)  — altitude = -z
  3      u        Body x (forward) velocity (m/s)
  4      v        Body y (right)   velocity (m/s)
  5      w        Body z (down)    velocity (m/s)
  6      q0       Quaternion scalar part   (–)
  7      q1       Quaternion vector part x (–)
  8      q2       Quaternion vector part y (–)
  9      q3       Quaternion vector part z (–)
  10     p        Roll  rate (rad/s)
  11     q        Pitch rate (rad/s)
  12     r        Yaw   rate (rad/s)

Coordinate conventions:
  - NED inertial frame: x = North, y = East, z = Down
  - Body frame: x = forward (roll axis), y = right, z = down
  - Quaternion: q = [q0, q1, q2, q3], q0 scalar, represents rotation from
    NED to body so that v_body = DCM @ v_ned where DCM = quat_to_dcm(q)
  - Euler products: L (roll), M (pitch), N (yaw) about body x, y, z axes

The model assumes a constant-thrust, constant-mass-flow motor and linear
aerodynamics as implemented in MissileAerodynamics.
"""

import numpy as np
from .atmosphere import StandardAtmosphere1976
from .aerodynamics import MissileAerodynamics
try:
    from ..utils.coordinate_transforms import quat_to_dcm, quat_normalize, wind_angles
except ImportError:
    from utils.coordinate_transforms import quat_to_dcm, quat_normalize, wind_angles


class Missile6DOF:
    """6-DOF rigid-body missile model with quaternion attitude representation.

    The model integrates the full Newton-Euler equations of motion for a
    rigid body with variable mass.  The Coriolis and gyroscopic coupling
    terms are included in both the translational and rotational equations.

    Moment of inertia variation with propellant consumption is neglected
    (constant Ixx, Iyy, Izz are used).  The missile is assumed axisymmetric
    so that Ixy = Ixz = Iyz = 0 and Iyy = Izz.

    Attributes:
        mass_0       (float): Launch mass (kg).
        mass_burnout (float): Post-burnout mass (kg).
        Ixx          (float): Roll  moment of inertia (kg·m²).
        Iyy          (float): Pitch moment of inertia (kg·m²).
        Izz          (float): Yaw   moment of inertia (kg·m²).
        thrust       (float): Constant motor thrust (N).
        burn_time    (float): Motor burn duration (s).
        S_ref        (float): Reference area (m²).
        d_ref        (float): Reference diameter (m).
        length       (float): Total vehicle length (m).
    """

    def __init__(self, aero_model=None) -> None:
        # ------------------------------------------------------------------
        # Mass properties
        # ------------------------------------------------------------------
        self.mass_0: float = 85.3        # kg
        self.mass_burnout: float = 71.5  # kg
        self.Ixx: float = 0.45           # kg·m²  (roll  — small due to slender body)
        self.Iyy: float = 12.8           # kg·m²  (pitch)
        self.Izz: float = 12.8           # kg·m²  (yaw — equal to pitch for axisymmetric)

        # ------------------------------------------------------------------
        # Propulsion
        # ------------------------------------------------------------------
        self.thrust: float = 17_500.0    # N
        self.burn_time: float = 2.2      # s
        self.mass_rate: float = (self.mass_0 - self.mass_burnout) / self.burn_time

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        self.S_ref: float = 0.01267      # m²
        self.d_ref: float = 0.127        # m
        self.length: float = 2.87        # m

        # ------------------------------------------------------------------
        # Sub-models
        # ------------------------------------------------------------------
        self.atm = StandardAtmosphere1976()
        if aero_model is not None:
            self.aero = aero_model
        else:
            self.aero = MissileAerodynamics()
        self.g: float = 9.80665          # m/s²

    # ------------------------------------------------------------------
    # Propulsion helpers
    # ------------------------------------------------------------------

    def get_mass(self, t: float) -> float:
        """Return current vehicle mass (kg) at time t (s)."""
        if t < self.burn_time:
            return self.mass_0 - self.mass_rate * t
        return self.mass_burnout

    def get_thrust(self, t: float) -> float:
        """Return thrust magnitude (N) at time t (s)."""
        return self.thrust if t < self.burn_time else 0.0

    # ------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        delta_e: float,
        delta_r: float = 0.0,
        delta_a: float = 0.0,
    ) -> np.ndarray:
        """Compute the 13-element state derivative vector.

        Implements the full Newton-Euler equations:
          Translational: m*(dV_body/dt + omega x V_body) = F_aero + F_thrust + F_gravity
          Rotational:    I * domega/dt + omega x (I*omega) = M_aero
          Position:      dx_ned/dt = DCM^T @ V_body
          Attitude:      dq/dt = 0.5 * q * [0, p, q, r]  (quaternion kinematics)

        The gravity vector in the body frame is computed directly from the
        quaternion, avoiding the intermediate Euler angle extraction and its
        associated singularity.

        Args:
            t       : Current time (s).
            state   : 13-element state vector [x,y,z, u,v,w, q0,q1,q2,q3, p,q,r].
            delta_e : Elevator (pitch fin) deflection (rad). Positive: trailing-edge-up.
            delta_r : Rudder (yaw fin) deflection (rad).     Positive: trailing-edge-right.
            delta_a : Aileron (roll fin) deflection (rad).   Positive: right-aileron-up.

        Returns:
            np.ndarray of shape (13,) with time derivatives.
        """
        # ------------------------------------------------------------------
        # Unpack state
        # ------------------------------------------------------------------
        x, y, z = state[0:3]          # NED position
        u, v, w = state[3:6]          # body-frame velocity
        q0, q1, q2, q3 = state[6:10]  # attitude quaternion
        p, qr, r = state[10:13]       # body angular rates  (qr avoids clash with q_bar)

        # ------------------------------------------------------------------
        # Quaternion normalisation — enforce unit constraint at every step
        # ------------------------------------------------------------------
        qn = quat_normalize(np.array([q0, q1, q2, q3]))
        q0, q1, q2, q3 = qn

        # DCM from body to NED: v_ned = C_bn @ v_body
        # quat_to_dcm returns C_nb (NED->body), so transpose gives C_bn.
        C_nb = quat_to_dcm(qn)          # NED -> body
        C_bn = C_nb.T                   # body -> NED

        # ------------------------------------------------------------------
        # Aerodynamic angles and airspeed
        # ------------------------------------------------------------------
        V = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        alpha, beta = wind_angles(u, v, w)

        # ------------------------------------------------------------------
        # Atmosphere (altitude = -z in NED, positive up)
        # ------------------------------------------------------------------
        alt = -z
        _, _, rho, _ = self.atm.get_properties(max(float(alt), 0.0))
        q_bar = 0.5 * rho * V ** 2

        # ------------------------------------------------------------------
        # Mass and thrust
        # ------------------------------------------------------------------
        m = self.get_mass(t)
        T = self.get_thrust(t)

        # ------------------------------------------------------------------
        # Aerodynamic forces and moments
        # ------------------------------------------------------------------
        forces_aero, moments_aero = self.aero.get_forces_moments(
            alpha, beta, V, q_bar, p, qr, r, delta_e, delta_r, delta_a
        )
        Fx_a, Fy_a, Fz_a = forces_aero
        L_a, M_a, N_a = moments_aero

        # ------------------------------------------------------------------
        # Thrust (aligned with body x-axis, no thrust misalignment modelled)
        #
        # The thrust value (17500 N) is treated as *effective thrust*
        # (T = mdot * V_exhaust), equivalent to what is measured on a thrust
        # stand.  Therefore the translational EOM  m*dV/dt = F_total  is
        # correct as-is.  If this were raw motor chamber thrust instead,
        # an additional  -(dm/dt)*V_body  term would be required in the
        # translational equations (variable-mass Meshchersky equation).
        # For the 2.2 s burn time of this tactical motor the difference
        # is negligible either way (~0.5 m/s integrated velocity error).
        # ------------------------------------------------------------------
        Fx_T = T

        # ------------------------------------------------------------------
        # Gravity in body frame via quaternion
        #
        # In NED, gravity is g_ned = [0, 0, g]^T (pointing down = +z_ned).
        # Body-frame gravity: g_body = C_nb @ g_ned
        # Using quaternion elements directly (avoids redundant DCM multiply):
        #   g_body_x =  2*g*(q1*q3 - q0*q2)      ← DCM row 0, col 2, * g
        #   g_body_y =  2*g*(q2*q3 + q0*q1)
        #   g_body_z =  g*(q0^2 - q1^2 - q2^2 + q3^2)
        # These come from the (0,2), (1,2), (2,2) elements of C_nb * g.
        # ------------------------------------------------------------------
        gx =  self.g * 2.0 * (q1 * q3 - q0 * q2)
        gy =  self.g * 2.0 * (q2 * q3 + q0 * q1)
        gz =  self.g * (q0**2 - q1**2 - q2**2 + q3**2)

        # ------------------------------------------------------------------
        # Translational equations of motion (body frame)
        # m * a_body = F_total - omega x (m * V_body)
        # where omega x V_body = [qr*w - r*v, r*u - p*w, p*v - qr*u]
        # ------------------------------------------------------------------
        du = (Fx_a + Fx_T) / m + gx + (r  * v - qr * w)
        dv =  Fy_a         / m + gy + (p  * w -  r * u)
        dw =  Fz_a         / m + gz + (qr * u -  p * v)

        # ------------------------------------------------------------------
        # Position kinematics (NED)
        # v_ned = C_bn @ v_body
        # ------------------------------------------------------------------
        vel_ned = C_bn @ np.array([u, v, w])
        dx, dy, dz = vel_ned

        # ------------------------------------------------------------------
        # Quaternion kinematics
        # q_dot = 0.5 * q (x) [0, p, q, r]  (Hamilton product, scalar first)
        #
        # Expanded (scalar-first convention):
        #   dq0 = -0.5*(p*q1 + qr*q2 + r*q3)
        #   dq1 =  0.5*(p*q0 + r*q2 - qr*q3)
        #   dq2 =  0.5*(qr*q0 - r*q1 + p*q3)
        #   dq3 =  0.5*(r*q0 + qr*q1 - p*q2)
        # ------------------------------------------------------------------
        dq0 = 0.5 * (-p  * q1 - qr * q2 - r  * q3)
        dq1 = 0.5 * ( p  * q0 + r  * q2 - qr * q3)
        dq2 = 0.5 * ( qr * q0 - r  * q1 + p  * q3)
        dq3 = 0.5 * ( r  * q0 + qr * q1 - p  * q2)

        # ------------------------------------------------------------------
        # Rotational equations of motion (Euler's equations)
        # I * omega_dot = M_total - omega x (I * omega)
        #
        # For diagonal inertia tensor:
        #   Ixx * dp  = L + (Iyy - Izz) * qr * r
        #   Iyy * dqr = M + (Izz - Ixx) * r  * p
        #   Izz * dr  = N + (Ixx - Iyy) * p  * qr
        # ------------------------------------------------------------------
        dp  = (L_a + (self.Iyy - self.Izz) * qr * r ) / self.Ixx
        dqr = (M_a + (self.Izz - self.Ixx) * r  * p ) / self.Iyy
        dr  = (N_a + (self.Ixx - self.Iyy) * p  * qr) / self.Izz

        return np.array([
            dx,  dy,  dz,
            du,  dv,  dw,
            dq0, dq1, dq2, dq3,
            dp,  dqr, dr,
        ])

    # ------------------------------------------------------------------
    # Simulation driver (fixed-step RK4)
    # ------------------------------------------------------------------

    def simulate(
        self,
        state0: np.ndarray,
        control_func,
        t_span: tuple = (0.0, 60.0),
        dt: float = 0.001,
    ) -> dict:
        """Simulate the 6-DOF trajectory using fixed-step RK4 integration.

        The control function is called once per step (zero-order hold) and
        returns the three fin deflections evaluated at the current state.

        Args:
            state0      : Initial 13-element state vector.
            control_func: Callable (t, state) -> (delta_e, delta_r, delta_a)
                          returning fin deflections in radians.
            t_span      : (t_start, t_end) simulation interval (s).
            dt          : Fixed integration step size (s).

        Returns:
            dict with keys:
                't'       : np.ndarray of shape (N,), recorded time stamps (s).
                'state'   : np.ndarray of shape (N, 13), state history.
                'controls': np.ndarray of shape (N, 3), [de, dr, da] history.

        Notes:
            Output is sub-sampled to every 10 steps to limit memory use.
            Simulation terminates early if altitude drops below 0 m (ground
            impact) after the first integration step.
        """
        t_start, t_end = t_span
        t = float(t_start)
        state = np.array(state0, dtype=float)

        # Normalise initial quaternion
        state[6:10] = quat_normalize(state[6:10])

        history: dict = {'t': [], 'state': [], 'controls': []}
        step_count = 0

        while t < t_end:
            # Ground-impact check (alt = -z_ned; terminate if below ground)
            alt = -state[2]
            if alt < 0.0 and t > t_start:
                break

            # Evaluate control law
            delta_e, delta_r, delta_a = control_func(t, state)

            # RK4 integration
            k1 = self.derivatives(t,            state,              delta_e, delta_r, delta_a)
            k2 = self.derivatives(t + 0.5 * dt, state + 0.5*dt*k1, delta_e, delta_r, delta_a)
            k3 = self.derivatives(t + 0.5 * dt, state + 0.5*dt*k2, delta_e, delta_r, delta_a)
            k4 = self.derivatives(t + dt,        state +     dt*k3, delta_e, delta_r, delta_a)

            state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

            # Re-normalise quaternion to counteract drift
            state[6:10] = quat_normalize(state[6:10])

            t += dt
            step_count += 1

            # Sub-sample: record every 10 steps
            if step_count % 10 == 0:
                history['t'].append(t)
                history['state'].append(state.copy())
                history['controls'].append([delta_e, delta_r, delta_a])

        # Convert lists to arrays
        for key in history:
            history[key] = np.array(history[key]) if history[key] else np.empty(0)

        return history
