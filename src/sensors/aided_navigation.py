"""Aided Navigation System.

Orchestrates StrapdownINS and NavKalmanFilter to produce a best-estimate
navigation state from sensor measurements only.

Architecture (indirect / error-state EKF):
    HIGH RATE: IMU -> StrapdownINS (attitude/velocity/position)
                   -> NavKalmanFilter.predict() (covariance propagation)
    LOW RATE:  GPS measurement -> innovation = meas - INS_estimate
                               -> NavKalmanFilter.correct() (error-state update)
                               -> apply corrections to INS
                               -> reset filter error state

CRITICAL: This class NEVER receives truth state. All inputs are sensor
measurements generated externally (IMUModel, GPSModel).

References:
    Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
        Navigation Systems", 2nd ed., Artech House, 2013. Chs. 14-15.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.coordinate_transforms import quat_normalize
from .strapdown_ins import StrapdownINS
from .nav_kalman_filter import NavKalmanFilter


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ r (scalar-first [q0, q1, q2, q3] form)."""
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0,
    ])


# ---------------------------------------------------------------------------
# NavState
# ---------------------------------------------------------------------------

@dataclass
class NavState:
    """Best-estimate navigation state produced by AidedNavigationSystem."""
    pos_ned:   np.ndarray   # [3] position in NED (m)
    vel_ned:   np.ndarray   # [3] velocity in NED (m/s)
    quat:      np.ndarray   # [4] attitude quaternion [q0, q1, q2, q3]
    pos_sigma: np.ndarray   # [3] position 1-sigma (m) from filter covariance
    vel_sigma: np.ndarray   # [3] velocity 1-sigma (m/s) from filter covariance


# ---------------------------------------------------------------------------
# AidedNavigationSystem
# ---------------------------------------------------------------------------

class AidedNavigationSystem:
    """GPS-aided strapdown INS using indirect (error-state) EKF.

    Accepts ONLY sensor measurements (never truth state). The engagement
    loop is responsible for generating and passing measurements:

        gyro_meas, accel_meas = imu.measure(true_omega, true_accel, dt)
        nav_state = aided_nav.propagate(gyro_meas, accel_meas, dt)

        if gps.is_update_due(t):
            gps_pos, gps_vel = gps.measure(true_pos, true_vel)
            R = np.diag([gps.pos_sigma**2]*3 + [gps.vel_sigma**2]*3)
            aided_nav.correct('gps_posvel',
                               np.concatenate([gps_pos, gps_vel]), R)

    Args:
        ins:        Initialised StrapdownINS instance.
        nav_filter: Initialised NavKalmanFilter instance.
    """

    def __init__(self, ins: StrapdownINS, nav_filter: NavKalmanFilter):
        self.ins = ins
        self.nav_filter = nav_filter
        self.innovation_outlier: bool = False   # True when last update exceeded 3-sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(
        self,
        gyro_meas: np.ndarray,
        accel_meas: np.ndarray,
        dt: float,
    ) -> NavState:
        """High-rate INS propagation driven by IMU measurements.

        Steps:
            1. INS mechanization (attitude -> velocity -> position)
            2. Filter error-state covariance prediction
            3. Return corrected NavState

        Args:
            gyro_meas:  Measured angular velocity, body frame [3] (rad/s).
            accel_meas: Measured specific force, body frame [3] (m/s^2).
            dt:         IMU time step (s).

        Returns:
            NavState with current best-estimate position, velocity, attitude
            and 1-sigma uncertainties derived from filter covariance.
        """
        ins_state = self.ins.propagate(gyro_meas, accel_meas, dt)

        # predict() needs specific force and body-to-NED DCM from INS attitude
        self.nav_filter.predict(accel_meas, ins_state['dcm_bn'], dt)

        return self._build_nav_state(ins_state)

    def correct(
        self,
        meas_type: str,
        measurement: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """Low-rate aiding update from an external sensor measurement.

        Steps:
            1. Compute innovation = measurement - INS_estimate
            2. Innovation 3-sigma outlier check
            3. Kalman measurement update (nav_filter.correct)
            4. Apply error-state corrections to INS solution
            5. Reset filter error state

        Args:
            meas_type:   'gps_pos'    – 3-DOF position measurement [3] (m)
                         'gps_vel'    – 3-DOF velocity measurement [3] (m/s)
                         'gps_posvel' – combined [pos(3); vel(3)] measurement
            measurement: Raw sensor measurement in NED frame.
            R:           Measurement noise covariance [m x m].
        """
        measurement = np.asarray(measurement, dtype=float)
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = np.diag(R)

        ins_state = self.ins.get_state()

        # 1. Innovation
        innovation = self._compute_innovation(meas_type, measurement, ins_state)

        # 2. Outlier detection
        self.innovation_outlier = self._check_outlier(meas_type, innovation, R)

        # 3. Kalman update
        self.nav_filter.correct(meas_type, innovation, R)

        # 4. Apply error-state corrections to INS
        self._apply_corrections()

        # 5. Reset filter error state (corrections now embedded in INS)
        self.nav_filter.reset_error_state()

    def get_nav_state(self) -> NavState:
        """Return current best-estimate navigation state."""
        return self._build_nav_state(self.ins.get_state())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_innovation(
        self,
        meas_type: str,
        measurement: np.ndarray,
        ins_state: dict,
    ) -> np.ndarray:
        """Compute innovation z = measurement - h(x_ins)."""
        if meas_type == 'gps_pos':
            return measurement - ins_state['pos_ned']
        elif meas_type == 'gps_vel':
            return measurement - ins_state['vel_ned']
        elif meas_type == 'gps_posvel':
            ins_pv = np.concatenate([ins_state['pos_ned'], ins_state['vel_ned']])
            return measurement - ins_pv
        else:
            # Custom types: caller must pre-compute innovation
            return measurement

    def _apply_corrections(self) -> None:
        """Apply EKF error-state correction to INS solution.

        Error state layout (NavKalmanFilter convention):
            x[0:3]  – position error  δr (NED, m)
            x[3:6]  – velocity error  δv (NED, m/s)
            x[6:9]  – attitude error  ψ  (NED small-angle, rad)
            x[9:12] – accel bias      δb_a (body, m/s^2)  [not applied here]
            x[12:15]– gyro bias       δb_g (body, rad/s)  [not applied here]
        """
        dx = self.nav_filter.get_error_state()

        # Position and velocity: direct additive correction
        self.ins.pos_ned += dx[0:3]
        self.ins.vel_ned += dx[3:6]

        # Attitude: ψ is the NED-frame attitude error
        # q_true ≈ δq ⊗ q_ins, where δq ≈ normalize([1, ψ/2])
        delta_psi = dx[6:9]
        half = 0.5 * delta_psi
        dq = quat_normalize(np.array([1.0, half[0], half[1], half[2]]))
        self.ins.quat = quat_normalize(_quat_multiply(dq, self.ins.quat))

    def _build_nav_state(self, ins_state: dict) -> NavState:
        """Construct NavState from INS state and filter covariance diagonal."""
        P_diag = np.diag(self.nav_filter.get_covariance())
        return NavState(
            pos_ned=ins_state['pos_ned'].copy(),
            vel_ned=ins_state['vel_ned'].copy(),
            quat=ins_state['quat'].copy(),
            pos_sigma=np.sqrt(np.maximum(P_diag[0:3], 0.0)),
            vel_sigma=np.sqrt(np.maximum(P_diag[3:6], 0.0)),
        )

    def _check_outlier(
        self,
        meas_type: str,
        innovation: np.ndarray,
        R: np.ndarray,
    ) -> bool:
        """Return True if any innovation component exceeds 3-sigma gate.

        Approximate innovation covariance: S_ii ≈ P_ii + R_ii
        (diagonal elements of S = H @ P @ H.T + R).
        """
        P = self.nav_filter.get_covariance()
        R_diag = np.diag(R)

        if meas_type == 'gps_pos':
            S_diag = np.diag(P[0:3, 0:3]) + R_diag
        elif meas_type == 'gps_vel':
            S_diag = np.diag(P[3:6, 3:6]) + R_diag
        elif meas_type == 'gps_posvel':
            S_diag = np.concatenate([
                np.diag(P[0:3, 0:3]),
                np.diag(P[3:6, 3:6]),
            ]) + R_diag
        else:
            return False

        threshold = 3.0 * np.sqrt(np.maximum(S_diag, 0.0))
        return bool(np.any(np.abs(innovation) > threshold))
