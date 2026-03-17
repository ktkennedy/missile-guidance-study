"""Navigation Kalman Filter -- 15-state error-state (indirect) EKF.

GPS-aided strapdown INS integration using the error-state (indirect) EKF
formulation.  All navigation corrections are computed in error-state space
and then fed back to the INS via the AidedNavigationSystem.

Error state vector (15 states):
    δr  [0:3]   – position error in NED frame (m)
    δv  [3:6]   – velocity error in NED frame (m/s)
    ψ   [6:9]   – attitude error / small-angle tilt errors (rad)
    bₐ  [9:12]  – accelerometer bias in body frame (m/s²)
    bᵍ  [12:15] – gyroscope bias in body frame (rad/s)

Continuous-time F matrix (15×15):
    ┌ 0   I      0      0       0   ┐  δr̈  = δv
    │ 0   0    -skew(fₙ)  Cbn   0   │  δv̈  = -skew(fₙ)·ψ + Cbn·bₐ
    │ 0   0      0      0     -Cbn  │  ψ̇   = -Cbn·bᵍ
    │ 0   0      0    -I/τₐ   0    │  ḃₐ  = -bₐ/τₐ  (Gauss-Markov)
    └ 0   0      0      0    -I/τᵍ ┘  ḃᵍ  = -bᵍ/τᵍ  (Gauss-Markov)

where fₙ = Cbn·f_b is specific force in NED frame,
      Cbn = DCM body-to-NED,
      skew(v) is the 3×3 skew-symmetric (cross-product) matrix of v.

References:
    Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
        Navigation Systems", 2nd ed., Artech House, 2013.  Chs. 14-15.
    Titterton & Weston, "Strapdown Inertial Navigation Technology", 2nd ed.
    Bar-Shalom, "Estimation with Applications to Tracking and Navigation".
"""
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix for vector v (cross-product matrix)."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([
        [   0.0, -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],  v[0],    0.0],
    ])


# ---------------------------------------------------------------------------
# NavKalmanFilter
# ---------------------------------------------------------------------------

class NavKalmanFilter:
    """15-state error-state EKF for GPS-aided inertial navigation.

    Args:
        gyro_arw:              Gyroscope angular random walk (rad/√s).
        accel_vrw:             Accelerometer velocity random walk (m/s/√s).
        gyro_bias_std:         Gyroscope bias 1-sigma (rad/s), steady-state.
        accel_bias_std:        Accelerometer bias 1-sigma (m/s²), steady-state.
        gyro_tau:              Gyroscope bias correlation time (s), default 300.
        accel_tau:             Accelerometer bias correlation time (s), default 600.
        initial_pos_sigma:     Initial position uncertainty 1-sigma (m), default 10.
        initial_vel_sigma:     Initial velocity uncertainty 1-sigma (m/s), default 1.
        initial_att_sigma_deg: Initial attitude uncertainty 1-sigma (deg), default 1.
    """

    N = 15   # total error-state dimension

    # Slice indices into the 15-vector
    _POS = slice(0, 3)
    _VEL = slice(3, 6)
    _ATT = slice(6, 9)
    _BA  = slice(9, 12)
    _BG  = slice(12, 15)

    def __init__(
        self,
        gyro_arw: float,
        accel_vrw: float,
        gyro_bias_std: float,
        accel_bias_std: float,
        gyro_tau: float = 300.0,
        accel_tau: float = 600.0,
        initial_pos_sigma: float = 10.0,
        initial_vel_sigma: float = 1.0,
        initial_att_sigma_deg: float = 1.0,
    ):
        self.gyro_arw       = float(gyro_arw)
        self.accel_vrw      = float(accel_vrw)
        self.gyro_bias_std  = float(gyro_bias_std)
        self.accel_bias_std = float(accel_bias_std)
        self.gyro_tau       = float(gyro_tau)
        self.accel_tau      = float(accel_tau)

        # Error state (initialised to zero — errors unknown, encoded in P)
        self.x = np.zeros(self.N)

        # Initial covariance P₀ (diagonal, 1-sigma values from constructor)
        att_sigma = float(np.radians(initial_att_sigma_deg))
        P_diag = np.concatenate([
            np.full(3, initial_pos_sigma**2),
            np.full(3, initial_vel_sigma**2),
            np.full(3, att_sigma**2),
            np.full(3, accel_bias_std**2),
            np.full(3, gyro_bias_std**2),
        ])
        self.P = np.diag(P_diag)

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def predict(
        self,
        specific_force_body: np.ndarray,
        dcm_bn: np.ndarray,
        dt: float,
    ) -> None:
        """Propagate error-state dynamics for one IMU time step.

        Uses first-order (Euler) discretisation: Φ = I + F·dt, which is
        accurate for small dt (≤ 0.02 s) typical of tactical IMUs.

        Args:
            specific_force_body: Specific force in body frame [3] (m/s²).
                                  This is the raw accelerometer output (includes
                                  gravity reaction for a stationary body).
            dcm_bn:              Body-to-NED direction cosine matrix [3×3].
                                  Convention: v_ned = dcm_bn @ v_body.
            dt:                  IMU sample interval (s).
        """
        f_b = np.asarray(specific_force_body, dtype=float).ravel()
        C   = np.asarray(dcm_bn, dtype=float)          # body → NED
        dt  = float(dt)

        # Specific force rotated to NED frame
        f_n = C @ f_b

        # ----------------------------------------------------------
        # Continuous-time F matrix (15×15)
        # ----------------------------------------------------------
        F = np.zeros((self.N, self.N))

        # δṙ = δv
        F[self._POS, self._VEL] = np.eye(3)

        # δv̇ = −skew(fₙ)·ψ + C·bₐ
        F[self._VEL, self._ATT] = -_skew(f_n)
        F[self._VEL, self._BA]  =  C

        # ψ̇ = −C·bᵍ
        F[self._ATT, self._BG]  = -C

        # ḃₐ = −bₐ/τₐ
        F[self._BA, self._BA]   = -np.eye(3) / self.accel_tau

        # ḃᵍ = −bᵍ/τᵍ
        F[self._BG, self._BG]   = -np.eye(3) / self.gyro_tau

        # ----------------------------------------------------------
        # Discrete state transition matrix: Φ = I + F·dt
        # ----------------------------------------------------------
        Phi = np.eye(self.N) + F * dt

        # ----------------------------------------------------------
        # Discrete process noise covariance Q
        # (noise sources mapped directly to affected states)
        # ----------------------------------------------------------
        Q = np.zeros((self.N, self.N))

        # Velocity noise from accelerometer VRW (white noise, PSD = accel_vrw²)
        Q[self._VEL, self._VEL] = self.accel_vrw**2 * dt * np.eye(3)

        # Attitude noise from gyroscope ARW (white noise, PSD = gyro_arw²)
        Q[self._ATT, self._ATT] = self.gyro_arw**2 * dt * np.eye(3)

        # Accel bias drive noise (Gauss-Markov steady-state: σ_drive² = 2σ²/τ)
        q_ba = 2.0 * self.accel_bias_std**2 / self.accel_tau * dt
        Q[self._BA, self._BA]   = q_ba * np.eye(3)

        # Gyro bias drive noise
        q_bg = 2.0 * self.gyro_bias_std**2 / self.gyro_tau * dt
        Q[self._BG, self._BG]   = q_bg * np.eye(3)

        # ----------------------------------------------------------
        # Propagate state and covariance
        # ----------------------------------------------------------
        self.x = Phi @ self.x
        self.P = Phi @ self.P @ Phi.T + Q

        # Enforce symmetry to prevent numerical drift
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # Correction step
    # ------------------------------------------------------------------

    def correct(
        self,
        meas_type: str,
        measurement: np.ndarray,
        R: np.ndarray,
        H_matrix: np.ndarray = None,
    ) -> np.ndarray:
        """Apply a measurement update using the Joseph form for stability.

        Args:
            meas_type:   Measurement type string.  Supported values:
                         ``'gps_pos'``    – 3-DOF GPS position innovation (m).
                         ``'gps_vel'``    – 3-DOF GPS velocity innovation (m/s).
                         ``'gps_posvel'`` – 6-DOF combined GPS pos+vel innovation.
                         ``'radar_range'``– Scalar or vector range innovation (m);
                                            *requires* ``H_matrix`` to be provided.
            measurement: Innovation vector z = z_meas − z_ins [m].
            R:           Measurement noise covariance [m×m] (or 1-D diagonal).
            H_matrix:    Measurement matrix [m×15].  Required for
                         ``'radar_range'``; ignored for GPS types.

        Returns:
            delta_x: The 15-element error-state correction Δx = K·z.
        """
        z = np.asarray(measurement, dtype=float).ravel()
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = np.diag(R)

        H = self._build_H(meas_type, H_matrix)    # [m × 15]

        # Innovation covariance S = H·P·H' + R  [m × m]
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)                       # enforce symmetry

        # Kalman gain K = P·H'·S⁻¹  [15 × m]
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error-state update
        delta_x = K @ z
        self.x  = self.x + delta_x

        # Joseph-form covariance update (numerically stable):
        #   P = (I − K·H)·P·(I − K·H)' + K·R·K'
        I   = np.eye(self.N)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

        return delta_x.copy()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_error_state(self) -> np.ndarray:
        """Return the 15-element error state vector (copy)."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return the 15×15 covariance matrix (copy)."""
        return self.P.copy()

    def reset_error_state(self) -> None:
        """Zero the error state after its correction has been applied to the INS.

        After calling this method the filter's x vector is all zeros while the
        covariance P is retained, ready for the next propagation cycle.
        """
        self.x[:] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_H(self, meas_type: str, H_matrix) -> np.ndarray:
        """Construct the measurement matrix H [m × 15]."""
        N = self.N
        if meas_type == 'gps_pos':
            H = np.zeros((3, N))
            H[:, self._POS] = np.eye(3)

        elif meas_type == 'gps_vel':
            H = np.zeros((3, N))
            H[:, self._VEL] = np.eye(3)

        elif meas_type == 'gps_posvel':
            H = np.zeros((6, N))
            H[0:3, self._POS] = np.eye(3)
            H[3:6, self._VEL] = np.eye(3)

        elif meas_type == 'radar_range':
            if H_matrix is None:
                raise ValueError(
                    "meas_type='radar_range' requires H_matrix to be provided "
                    "(1×15 row vector with unit-range direction in position columns)."
                )
            H = np.asarray(H_matrix, dtype=float)

        else:
            if H_matrix is None:
                raise ValueError(
                    f"Unknown meas_type='{meas_type}'.  Provide H_matrix for "
                    "custom measurement types."
                )
            H = np.asarray(H_matrix, dtype=float)

        return H
