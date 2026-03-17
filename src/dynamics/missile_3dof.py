"""
3-DOF point-mass missile model.

State vector: [x, y, alt, V, gamma, psi]
  x, y   : horizontal position in NED frame (m), x = North, y = East
  alt    : altitude above sea level (m, positive up)
  V      : total speed (m/s)
  gamma  : flight-path angle (rad, positive up)
  psi    : heading angle (rad, measured clockwise from North)

The equations of motion are the standard flight-path-angle form of the
point-mass equations derived from Newton's second law in the wind frame.
A first-order acceleration-command autopilot is assumed: the guidance law
specifies the desired normal (pitch-plane) and lateral accelerations directly,
and the equations propagate the resulting flight-path and heading changes.
"""

import numpy as np
from .atmosphere import StandardAtmosphere1976


class Missile3DOF:
    """3-DOF point-mass missile with acceleration-command autopilot interface.

    The model computes drag from an effective angle of attack derived from the
    commanded normal acceleration.  Thrust is modelled as a constant value
    during the burn phase and zero thereafter.  Propellant is consumed at a
    constant mass-flow rate.

    Args:
        mass         : Launch mass (kg).
        mass_burnout : Post-burnout mass (kg).
        thrust       : Constant thrust during burn (N).
        burn_time    : Burn duration (s).
        S_ref        : Reference area (m²).
        CL_alpha     : Lift-curve slope (1/rad).
        CD_0         : Zero-lift drag coefficient.
        CD_alpha2    : Induced drag coefficient (1/rad²).
    """

    def __init__(
        self,
        mass: float = 85.3,
        mass_burnout: float = 71.5,
        thrust: float = 17_500.0,
        burn_time: float = 2.2,
        S_ref: float = 0.01267,
        CL_alpha: float = 18.5,
        CD_0: float = 0.35,
        CD_alpha2: float = 8.0,
    ) -> None:
        self.mass_0 = mass
        self.mass_burnout = mass_burnout
        self.thrust = thrust
        self.burn_time = burn_time
        self.S_ref = S_ref
        self.CL_alpha = CL_alpha
        self.CD_0 = CD_0
        self.CD_alpha2 = CD_alpha2
        self.atm = StandardAtmosphere1976()
        self.g = 9.80665  # m/s²

        if burn_time > 0.0:
            self.mass_rate = (mass - mass_burnout) / burn_time
        else:
            self.mass_rate = 0.0

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
        a_cmd_pitch: float,
        a_cmd_yaw: float,
    ) -> np.ndarray:
        """Compute time derivatives of the 6-element state vector.

        The commanded accelerations are the *inertial* normal and lateral
        accelerations that the autopilot is expected to deliver.  The model
        inverts the lift equation to find the effective angle of attack (used
        only for drag bookkeeping) and applies the commanded accelerations
        directly in the kinematic equations.

        Args:
            t           : Current time (s).
            state       : [x, y, alt, V, gamma, psi].
            a_cmd_pitch : Commanded pitch-plane normal acceleration (m/s²).
                          Positive sense: nose-up / upward acceleration.
            a_cmd_yaw   : Commanded lateral acceleration (m/s²).
                          Positive sense: nose-right / rightward acceleration.

        Returns:
            np.ndarray of shape (6,) containing [dx, dy, dalt, dV, dgamma, dpsi].
        """
        x, y, alt, V, gamma, psi = state

        m = self.get_mass(t)
        T = self.get_thrust(t)
        _, _, rho, _ = self.atm.get_properties(max(alt, 0.0))
        q_bar = 0.5 * rho * V ** 2

        # Invert lift equation for effective alpha (drag estimation only)
        # L = q_bar * S_ref * CL_alpha * alpha  =>  alpha = m*a / (q_bar*S*CLa)
        lift_slope = q_bar * self.S_ref * self.CL_alpha + 1e-10
        alpha_eff = (m * a_cmd_pitch) / lift_slope
        alpha_eff = float(np.clip(alpha_eff, -0.30, 0.30))  # ±~17 deg limit

        CD = self.CD_0 + self.CD_alpha2 * alpha_eff ** 2
        D = q_bar * self.S_ref * CD

        # Kinematic equations (wind-axis / Euler flight-path form)
        # Position in NED (alt = -z_ned, positive up)
        dx = V * np.cos(gamma) * np.cos(psi)   # Northward velocity
        dy = V * np.cos(gamma) * np.sin(psi)   # Eastward velocity
        dalt = V * np.sin(gamma)               # Rate of climb

        # Speed equation (along-path force balance)
        dV = (T * np.cos(alpha_eff) - D) / m - self.g * np.sin(gamma)

        # Flight-path angle equation (pitch-plane normal force)
        # dgamma = (a_cmd_pitch + T*sin(alpha_eff)/m - g*cos(gamma)) / V
        # The thrust normal component T*sin(alpha) is the force perpendicular
        # to the velocity vector from thrust misalignment with the flight path.
        thrust_normal = T * np.sin(alpha_eff) / m
        dgamma = (a_cmd_pitch + thrust_normal - self.g * np.cos(gamma)) / max(V, 1.0)

        # Heading rate equation (lateral normal force)
        # dpsi = N_lat / (m * V * cos(gamma))
        # Use min denominator of 1.0 m/s to avoid extreme rates near vertical flight
        dpsi = a_cmd_yaw / max(V * np.cos(gamma), 1.0)

        return np.array([dx, dy, dalt, dV, dgamma, dpsi])

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def simulate(
        self,
        state0: np.ndarray,
        guidance_func,
        t_span: tuple = (0.0, 60.0),
        dt: float = 0.001,
    ) -> dict:
        """Simulate the missile trajectory using fixed-step RK4 integration.

        The guidance function is evaluated at the start of each integration
        step (zero-order-hold over the step).

        Args:
            state0       : Initial state [x0, y0, alt0, V0, gamma0, psi0].
            guidance_func: Callable with signature (t, state) -> (a_pitch, a_yaw).
                           Returns commanded accelerations (m/s²).
            t_span       : (t_start, t_end) simulation time interval (s).
            dt           : Fixed integration step size (s).

        Returns:
            dict with keys:
                't'     : np.ndarray of shape (N,), time stamps (s).
                'state' : np.ndarray of shape (N, 6), state history.
                'a_cmd' : np.ndarray of shape (N, 2), [a_pitch, a_yaw] history.

        Notes:
            Output is sub-sampled to every 10 steps to limit memory usage.
            The missile terminates if altitude drops below 0 (ground impact).
        """
        t_start, t_end = t_span
        t = float(t_start)
        state = np.array(state0, dtype=float)

        history: dict = {'t': [], 'state': [], 'a_cmd': []}
        step_count = 0

        while t < t_end:
            # Check for ground impact
            if state[2] < 0.0 and t > t_start:
                break

            a_pitch, a_yaw = guidance_func(t, state)

            # Fixed-step RK4
            k1 = self.derivatives(t,            state,              a_pitch, a_yaw)
            k2 = self.derivatives(t + 0.5 * dt, state + 0.5*dt*k1, a_pitch, a_yaw)
            k3 = self.derivatives(t + 0.5 * dt, state + 0.5*dt*k2, a_pitch, a_yaw)
            k4 = self.derivatives(t + dt,        state +     dt*k3, a_pitch, a_yaw)

            state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t += dt
            step_count += 1

            # Sub-sample output: record every 10 steps
            if step_count % 10 == 0:
                history['t'].append(t)
                history['state'].append(state.copy())
                history['a_cmd'].append([a_pitch, a_yaw])

        # Convert to numpy arrays
        for key in history:
            history[key] = np.array(history[key]) if history[key] else np.empty(0)

        return history
