"""Target motion models for engagement simulation.

Provides various target maneuver profiles:
- Constant velocity (non-maneuvering)
- Weaving (sinusoidal maneuver)
- Step maneuver (constant-g turn)
- S-maneuver (alternating turns)

All models use analytical integration where possible to avoid accumulation
of numerical errors that would occur with Euler or RK4 propagation.

References:
    Zarchan Ch. 4
"""
import numpy as np


class Target:
    """Target with configurable maneuver profile.

    The target state is computed analytically from the initial conditions
    and a closed-form acceleration profile.

    NED frame convention: x = North, y = East, z = Down (positive downward).

    Args:
        r0: initial position [3] (m) in NED frame
        v0: initial velocity [3] (m/s)
        maneuver_type: one of
            'constant_velocity' - no maneuver
            'weaving'           - sinusoidal lateral acceleration
            'step'              - constant-g turn starting at a given time
            's_maneuver'        - alternating-sign constant-g turns
        maneuver_params: dict with type-specific parameters:
            weaving:
                amplitude_g  (float): peak acceleration in g's, default 5.0
                omega        (float): angular frequency (rad/s), default 0.5
                axis         (int):   NED axis index for acceleration, default 2
            step:
                accel_g      (float): acceleration magnitude in g's, default 5.0
                start_time   (float): time at which maneuver begins (s), default 2.0
                axis         (int):   NED axis index for acceleration, default 2
            s_maneuver:
                accel_g      (float): acceleration magnitude in g's, default 5.0
                switch_time  (float): period between sign reversals (s), default 3.0
                axis         (int):   NED axis index for acceleration, default 2
    """

    VALID_TYPES = {'constant_velocity', 'weaving', 'step', 's_maneuver', 'threat_missile'}

    def __init__(
        self,
        r0,
        v0,
        maneuver_type: str = 'constant_velocity',
        maneuver_params: dict = None,
    ):
        maneuver_type = str(maneuver_type)
        if maneuver_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown maneuver_type '{maneuver_type}'. "
                f"Choose from {self.VALID_TYPES}"
            )

        self.r0 = np.asarray(r0, dtype=float).copy()
        self.v0 = np.asarray(v0, dtype=float).copy()
        self.maneuver_type = maneuver_type
        self.params = dict(maneuver_params) if maneuver_params else {}
        self.g = 9.80665  # m/s²

        # Validate and supply defaults for each maneuver type
        if maneuver_type == 'weaving':
            self.params.setdefault('amplitude_g', 5.0)
            self.params.setdefault('omega', 0.5)
            self.params.setdefault('axis', 2)
        elif maneuver_type == 'step':
            self.params.setdefault('accel_g', 5.0)
            self.params.setdefault('start_time', 2.0)
            self.params.setdefault('axis', 2)
        elif maneuver_type == 's_maneuver':
            self.params.setdefault('accel_g', 5.0)
            self.params.setdefault('switch_time', 3.0)
            self.params.setdefault('axis', 2)
        elif maneuver_type == 'threat_missile':
            # Delegate to a ThreatMissile instance passed in params
            self._threat = self.params.get('threat')
            if self._threat is None:
                raise ValueError(
                    "maneuver_type='threat_missile' requires "
                    "'threat' key in maneuver_params with a ThreatMissile instance"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accel_scalar(self, t: float) -> float:
        """Return the scalar acceleration along the maneuver axis at time t."""
        t = float(t)

        if self.maneuver_type == 'constant_velocity':
            return 0.0

        elif self.maneuver_type == 'weaving':
            A = float(self.params['amplitude_g']) * self.g
            w = float(self.params['omega'])
            return A * float(np.sin(w * t))

        elif self.maneuver_type == 'step':
            A = float(self.params['accel_g']) * self.g
            t_start = float(self.params['start_time'])
            return A if t >= t_start else 0.0

        elif self.maneuver_type == 's_maneuver':
            A = float(self.params['accel_g']) * self.g
            T_sw = float(self.params['switch_time'])
            # Sign alternates every T_sw seconds
            n = int(t / T_sw)
            sign = 1.0 if (n % 2 == 0) else -1.0
            return sign * A

        return 0.0

    def _integrate_accel(self, t: float):
        """Return (delta_v, delta_r) along maneuver axis via analytical integration.

        Computes:
            dv(t) = integral_0^t a(tau) d tau
            dr(t) = integral_0^t dv(tau) d tau  (relative to initial position)

        Returns:
            dv: velocity change due to maneuver (scalar, m/s)
            dr: position change due to maneuver (scalar, m)
        """
        t = float(t)

        if self.maneuver_type == 'constant_velocity':
            return 0.0, 0.0

        elif self.maneuver_type == 'weaving':
            A = float(self.params['amplitude_g']) * self.g
            w = float(self.params['omega'])
            # a(tau) = A * sin(w * tau)
            # dv(t)  = A/w * (1 - cos(w*t))
            # dr(t)  = A/w * (t - sin(w*t)/w)
            if abs(w) < 1e-12:
                # Degenerate: zero frequency => constant acceleration
                dv = A * t
                dr = 0.5 * A * t**2
            else:
                dv = (A / w) * (1.0 - float(np.cos(w * t)))
                dr = (A / w) * (t - float(np.sin(w * t)) / w)
            return dv, dr

        elif self.maneuver_type == 'step':
            A = float(self.params['accel_g']) * self.g
            t0 = float(self.params['start_time'])
            # a(tau) = A * H(tau - t0)
            # dv(t)  = A * max(t - t0, 0)
            # dr(t)  = 0.5 * A * max(t - t0, 0)^2
            dt_eff = max(t - t0, 0.0)
            dv = A * dt_eff
            dr = 0.5 * A * dt_eff**2
            return dv, dr

        elif self.maneuver_type == 's_maneuver':
            A = float(self.params['accel_g']) * self.g
            T_sw = float(self.params['switch_time'])
            # Piecewise constant acceleration: sign alternates every T_sw seconds
            # Integrate analytically segment by segment
            n_full = int(t / T_sw)   # number of complete segments
            t_rem = t - n_full * T_sw  # remainder in current segment

            dv = 0.0
            dr = 0.0
            # Velocity at end of each complete segment
            # Segment k has sign = +1 if k even, -1 if k odd
            v_carry = 0.0  # velocity accumulated from maneuver axis
            r_carry = 0.0  # position accumulated from maneuver axis

            for k in range(n_full):
                sign = 1.0 if (k % 2 == 0) else -1.0
                a_k = sign * A
                # Over interval T_sw:
                dv_seg = a_k * T_sw
                dr_seg = v_carry * T_sw + 0.5 * a_k * T_sw**2
                r_carry += dr_seg
                v_carry += dv_seg

            # Remainder segment
            sign_n = 1.0 if (n_full % 2 == 0) else -1.0
            a_n = sign_n * A
            dr_rem = v_carry * t_rem + 0.5 * a_n * t_rem**2
            dv_rem = a_n * t_rem

            dv = v_carry + dv_rem
            dr = r_carry + dr_rem
            return dv, dr

        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_acceleration(self, t: float) -> np.ndarray:
        """Get target acceleration at time t.

        Args:
            t: time (s)

        Returns:
            acceleration [3] (m/s²) in NED frame
        """
        if self.maneuver_type == 'threat_missile':
            return self._threat.get_acceleration(t)
        accel = np.zeros(3)
        if self.maneuver_type != 'constant_velocity':
            axis = int(self.params['axis'])
            accel[axis] = self._accel_scalar(t)
        return accel

    def get_state(self, t: float):
        """Get target state at time t via analytical integration.

        Args:
            t: time (s), must be >= 0

        Returns:
            position     [3] (m)
            velocity     [3] (m/s)
            acceleration [3] (m/s²)
        """
        t = float(t)
        if t < 0.0:
            t = 0.0

        # Delegate to ThreatMissile if applicable
        if self.maneuver_type == 'threat_missile':
            return self._threat.get_state(t)

        # Base constant-velocity motion
        pos = self.r0 + self.v0 * t
        vel = self.v0.copy()
        acc = np.zeros(3)

        if self.maneuver_type != 'constant_velocity':
            axis = int(self.params['axis'])
            dv, dr = self._integrate_accel(t)
            pos[axis] += dr
            vel[axis] += dv
            acc[axis] = self._accel_scalar(t)

        return pos, vel, acc

    def propagate(self, t_array: np.ndarray) -> dict:
        """Get target trajectory over a time array.

        Args:
            t_array: 1-D array of time points (s)

        Returns:
            dict with:
                'time'         : [N] time array (s)
                'position'     : [N, 3] position (m)
                'velocity'     : [N, 3] velocity (m/s)
                'acceleration' : [N, 3] acceleration (m/s²)
        """
        t_array = np.asarray(t_array, dtype=float).ravel()
        N = len(t_array)

        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        accelerations = np.zeros((N, 3))

        for i, t in enumerate(t_array):
            p, v, a = self.get_state(t)
            positions[i] = p
            velocities[i] = v
            accelerations[i] = a

        return {
            'time': t_array.copy(),
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
        }
