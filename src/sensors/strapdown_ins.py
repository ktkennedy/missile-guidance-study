"""Strapdown Inertial Navigation System (SINS) mechanization.

Implements flat-Earth strapdown INS mechanization equations for tactical missiles
with 2-sample coning and sculling compensation.

References:
    Bezick et al., "Inertial Navigation for Guided Missile Systems",
        JHU/APL Technical Digest, Vol. 28, No. 4, 2010.
    Titterton & Weston, "Strapdown Inertial Navigation Technology", 2nd ed.

Conventions:
    - NED (North-East-Down) navigation frame
    - Body frame: x-forward, y-right, z-down
    - Quaternion: [q0, q1, q2, q3], q0 is scalar part
    - quat_to_dcm(q): v_body = C_nb @ v_ned  (NED -> Body)
    - Gravity in NED: g = [0, 0, +g0]  (z-down positive)
"""

import numpy as np

try:
    from ..utils.coordinate_transforms import quat_to_dcm, quat_normalize
except ImportError:
    from utils.coordinate_transforms import quat_to_dcm, quat_normalize


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ r (both in scalar-first [q0, q1, q2, q3] form)."""
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0,
    ])


# ---------------------------------------------------------------------------
# StrapdownINS
# ---------------------------------------------------------------------------

class StrapdownINS:
    """Flat-Earth strapdown INS mechanization for tactical missiles.

    Integrates body-frame IMU measurements to maintain navigation state:
    position (NED), velocity (NED), and attitude (quaternion).

    Mechanization chain per time step:
        1. Gyro  -> 2-sample coning compensation -> quaternion update
        2. Accel -> 2-sample sculling compensation -> velocity increment in NED
        3. Velocity integration (trapezoidal)
        4. Position integration (trapezoidal)

    Args:
        initial_pos_ned:  Initial position in NED frame, shape (3,) in metres.
        initial_vel_ned:  Initial velocity in NED frame, shape (3,) in m/s.
        initial_quat:     Initial attitude quaternion [q0, q1, q2, q3].
        gravity:          Gravitational acceleration (m/s^2), default 9.80665.
        use_earth_rate:   Include Earth-rate / Coriolis terms (default False).
                          Reserved for future use; currently no-op.
    """

    def __init__(
        self,
        initial_pos_ned: np.ndarray,
        initial_vel_ned: np.ndarray,
        initial_quat: np.ndarray,
        gravity: float = 9.80665,
        use_earth_rate: bool = False,
    ):
        self.pos_ned = np.asarray(initial_pos_ned, dtype=float).copy()
        self.vel_ned = np.asarray(initial_vel_ned, dtype=float).copy()
        self.quat = quat_normalize(np.asarray(initial_quat, dtype=float))
        self.gravity = float(gravity)
        self.use_earth_rate = bool(use_earth_rate)

        # Previous-sample increments for 2-sample coning / sculling compensation
        self._prev_angle_inc = np.zeros(3)   # omega_prev * dt  (rad)
        self._prev_vel_inc   = np.zeros(3)   # accel_prev * dt  (m/s)
        self._first_sample = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(
        self,
        gyro_meas: np.ndarray,
        accel_meas: np.ndarray,
        dt: float,
    ) -> dict:
        """Propagate INS state by one IMU sample.

        Args:
            gyro_meas:  Measured angular velocity in body frame, shape (3,) (rad/s).
            accel_meas: Measured specific force in body frame, shape (3,) (m/s^2).
            dt:         Time step (s).

        Returns:
            Navigation state dict:
                'pos_ned' : position in NED frame (3,) in metres
                'vel_ned' : velocity in NED frame (3,) in m/s
                'quat'    : attitude quaternion (4,) [q0, q1, q2, q3]
                'dcm_bn'  : body-to-NED DCM (3, 3)
        """
        gyro_meas  = np.asarray(gyro_meas,  dtype=float)
        accel_meas = np.asarray(accel_meas, dtype=float)
        dt = float(dt)

        # Current angle and velocity increments
        curr_angle_inc = gyro_meas  * dt   # delta_theta (rad)
        curr_vel_inc   = accel_meas * dt   # delta_v     (m/s)

        # Seed previous values on first call (no compensation on first step)
        if self._first_sample:
            self._prev_angle_inc = curr_angle_inc.copy()
            self._prev_vel_inc   = curr_vel_inc.copy()
            self._first_sample   = False

        # -------------------------------------------------------------------
        # 1. Attitude update with 2-sample coning compensation
        #    Bezick eq.: phi_c = (1/12) * (delta_theta_{k-1} x delta_theta_k)
        # -------------------------------------------------------------------
        coning_comp = (1.0 / 12.0) * np.cross(self._prev_angle_inc, curr_angle_inc)
        rot_vec = curr_angle_inc + coning_comp   # corrected rotation vector

        # Rotation vector -> delta quaternion (exact formula)
        angle = np.linalg.norm(rot_vec)
        if angle > 1e-10:
            half  = 0.5 * angle
            axis  = rot_vec / angle
            sin_h = np.sin(half)
            dq = np.array([
                np.cos(half),
                sin_h * axis[0],
                sin_h * axis[1],
                sin_h * axis[2],
            ])
        else:
            # First-order (Euler) approximation for very small rotations
            dq = np.array([
                1.0,
                0.5 * rot_vec[0],
                0.5 * rot_vec[1],
                0.5 * rot_vec[2],
            ])
            dq = quat_normalize(dq)

        # Post-multiply: body rotation in current body frame
        self.quat = quat_normalize(_quat_multiply(self.quat, dq))

        # DCM: NED -> Body  (quat_to_dcm convention: v_b = C_nb @ v_n)
        C_nb = quat_to_dcm(self.quat)
        C_bn = C_nb.T                    # Body -> NED

        # -------------------------------------------------------------------
        # 2. Velocity update with 2-sample sculling compensation
        #    Bezick eq.: delta_v_scul = (1/12) * (delta_theta_{k-1} x delta_v_k
        #                                        + delta_v_{k-1}   x delta_theta_k)
        # -------------------------------------------------------------------
        sculling_comp = (1.0 / 12.0) * (
            np.cross(self._prev_angle_inc, curr_vel_inc) +
            np.cross(self._prev_vel_inc,  curr_angle_inc)
        )

        # Gravity vector in NED (z-down positive)
        g_ned = np.array([0.0, 0.0, self.gravity])

        # Save velocity before update for trapezoidal position integration
        vel_old = self.vel_ned.copy()

        # Transform corrected specific-force increment to NED, add gravity
        delta_v_ned = C_bn @ (curr_vel_inc + sculling_comp) + g_ned * dt
        self.vel_ned = self.vel_ned + delta_v_ned

        # -------------------------------------------------------------------
        # 3. Position update (trapezoidal - exact for constant acceleration)
        # -------------------------------------------------------------------
        self.pos_ned = self.pos_ned + 0.5 * (vel_old + self.vel_ned) * dt

        # -------------------------------------------------------------------
        # Store increments for next step's compensation
        # -------------------------------------------------------------------
        self._prev_angle_inc = curr_angle_inc.copy()
        self._prev_vel_inc   = curr_vel_inc.copy()

        return self.get_state()

    def get_state(self) -> dict:
        """Return current navigation state.

        Returns:
            Dict with keys: 'pos_ned', 'vel_ned', 'quat', 'dcm_bn'.
        """
        C_nb = quat_to_dcm(self.quat)
        return {
            'pos_ned': self.pos_ned.copy(),
            'vel_ned': self.vel_ned.copy(),
            'quat':    self.quat.copy(),
            'dcm_bn':  C_nb.T,
        }

    def reset(
        self,
        pos_ned: np.ndarray,
        vel_ned: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        """Reset INS to a new initial state, clearing compensation history.

        Args:
            pos_ned: New position in NED frame, shape (3,) in metres.
            vel_ned: New velocity in NED frame, shape (3,) in m/s.
            quat:    New attitude quaternion [q0, q1, q2, q3].
        """
        self.pos_ned = np.asarray(pos_ned, dtype=float).copy()
        self.vel_ned = np.asarray(vel_ned, dtype=float).copy()
        self.quat    = quat_normalize(np.asarray(quat, dtype=float))
        self._prev_angle_inc = np.zeros(3)
        self._prev_vel_inc   = np.zeros(3)
        self._first_sample   = True
