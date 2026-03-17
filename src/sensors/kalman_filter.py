"""Kalman filter implementations for target tracking.

Includes:
- Alpha-beta filter (fixed-gain, simple)
- Alpha-beta-gamma filter (for maneuvering targets)
- Extended Kalman Filter (EKF) for LOS-based tracking

References:
    Bar-Shalom "Estimation with Applications to Tracking and Navigation"
    Zarchan Ch. 12-13
"""
import numpy as np


class AlphaBetaFilter:
    """Alpha-beta (constant gain) filter for 1D or 2D tracking.

    Tracks position and velocity using fixed gains:
        x_pred = x + v * dt
        v_pred = v
        x_update = x_pred + alpha * (z - x_pred)
        v_update = v_pred + (beta / dt) * (z - x_pred)

    Args:
        alpha: position gain (0 < alpha < 1), default 0.3
        beta:  velocity gain (0 < beta < 1), default 0.05
        dim:   number of dimensions (1 or 2), default 2
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.05, dim: int = 2):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dim = int(dim)
        self.x = np.zeros(dim)   # position estimate
        self.v = np.zeros(dim)   # velocity estimate
        self.initialized = False

    def update(self, z: np.ndarray, dt: float):
        """Update filter with new measurement.

        On the first call the position is initialised to the measurement and
        velocity is set to zero.

        Args:
            z:  measurement [dim] (same units as state)
            dt: time step (s), must be > 0

        Returns:
            x_est: position estimate [dim]
            v_est: velocity estimate [dim]
        """
        z = np.asarray(z, dtype=float).reshape(self.dim)
        dt = float(dt)

        if not self.initialized:
            # First measurement: initialise state
            self.x = z.copy()
            self.v = np.zeros(self.dim)
            self.initialized = True
            return self.x.copy(), self.v.copy()

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Predict
        x_pred = self.x + self.v * dt
        v_pred = self.v.copy()

        # Innovation
        inn = z - x_pred

        # Update
        self.x = x_pred + self.alpha * inn
        self.v = v_pred + (self.beta / dt) * inn

        return self.x.copy(), self.v.copy()

    def reset(self) -> None:
        """Reset filter state."""
        self.x = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.initialized = False


class AlphaBetaGammaFilter:
    """Alpha-beta-gamma filter for tracking maneuvering targets.

    Adds acceleration estimation for better maneuver tracking.
    Three states per dimension: position, velocity, acceleration.

    Prediction:
        x_pred = x + v * dt + 0.5 * a * dt^2
        v_pred = v + a * dt
        a_pred = a

    Update:
        inn    = z - x_pred
        x_est  = x_pred + alpha * inn
        v_est  = v_pred + (beta / dt) * inn
        a_est  = a_pred + (2 * gamma / dt^2) * inn

    Args:
        alpha: position gain (0 < alpha < 1), default 0.5
        beta:  velocity gain (0 < beta < 1), default 0.1
        gamma: acceleration gain (0 < gamma < 1), default 0.02
        dim:   number of dimensions, default 2
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.02,
        dim: int = 2,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.dim = int(dim)
        self.x = np.zeros(dim)
        self.v = np.zeros(dim)
        self.a = np.zeros(dim)
        self.initialized = False

    def update(self, z: np.ndarray, dt: float):
        """Update filter with new measurement.

        Args:
            z:  measurement [dim]
            dt: time step (s), must be > 0

        Returns:
            x_est: position estimate [dim]
            v_est: velocity estimate [dim]
            a_est: acceleration estimate [dim]
        """
        z = np.asarray(z, dtype=float).reshape(self.dim)
        dt = float(dt)

        if not self.initialized:
            self.x = z.copy()
            self.v = np.zeros(self.dim)
            self.a = np.zeros(self.dim)
            self.initialized = True
            return self.x.copy(), self.v.copy(), self.a.copy()

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Predict
        x_pred = self.x + self.v * dt + 0.5 * self.a * dt**2
        v_pred = self.v + self.a * dt
        a_pred = self.a.copy()

        # Innovation
        inn = z - x_pred

        # Update
        self.x = x_pred + self.alpha * inn
        self.v = v_pred + (self.beta / dt) * inn
        self.a = a_pred + (2.0 * self.gamma / dt**2) * inn

        return self.x.copy(), self.v.copy(), self.a.copy()

    def reset(self) -> None:
        """Reset filter state."""
        self.x = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.a = np.zeros(self.dim)
        self.initialized = False


class ExtendedKalmanFilter:
    """Extended Kalman Filter for target tracking using LOS measurements.

    State (2-D):
        x = [target_x, target_y, target_vx, target_vy]  (4 states)

    Measurement (nonlinear):
        z = lam_az = atan2(tgt_y - M_y, tgt_x - M_x)

    Process model: constant velocity
        x(k+1) = F * x(k) + w,  w ~ N(0, Q)

        F = [[1, 0, dt, 0 ],
             [0, 1, 0,  dt],
             [0, 0, 1,  0 ],
             [0, 0, 0,  1 ]]

        Q = q_std^2 * [[dt^3/3,      0, dt^2/2,      0],
                        [0,      dt^3/3,      0, dt^2/2],
                        [dt^2/2,      0,     dt,      0],
                        [0,      dt^2/2,      0,     dt]]

    Measurement model Jacobian:
        h(x)  = atan2(tgt_y - M_y, tgt_x - M_x)
        H     = [-dy/r^2,  dx/r^2, 0, 0]
        where dx = tgt_x - M_x, dy = tgt_y - M_y, r^2 = dx^2 + dy^2

    Covariance update uses the Joseph form for numerical stability:
        P = (I - K*H) * P_pred * (I - K*H)' + K * R * K'

    Args:
        q_std: process noise std for acceleration (m/s²), default 5.0
        r_std: measurement noise std (rad), default 0.003 (3 mrad)
    """

    def __init__(self, q_std: float = 5.0, r_std: float = 0.003):
        self.n = 4           # state dimension
        self.q_std = float(q_std)
        self.r_std = float(r_std)

        self.x = np.zeros(4)   # state estimate [tgt_x, tgt_y, tgt_vx, tgt_vy]
        self.P = np.eye(4)     # state covariance
        self.initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(
        self,
        r_M: np.ndarray,
        r_T_est: np.ndarray,
        v_T_est: np.ndarray = None,
    ) -> None:
        """Initialize filter with first estimate.

        Args:
            r_M:     missile position [2 or 3]
            r_T_est: initial target position estimate [2 or 3]
            v_T_est: initial target velocity estimate [2 or 3]; zeros if None
        """
        r_M = np.asarray(r_M, dtype=float).ravel()
        r_T_est = np.asarray(r_T_est, dtype=float).ravel()

        if v_T_est is None:
            v_T_est_2 = np.zeros(2)
        else:
            v_T_est_2 = np.asarray(v_T_est, dtype=float).ravel()[:2]

        self.x = np.array([
            r_T_est[0],
            r_T_est[1],
            v_T_est_2[0],
            v_T_est_2[1],
        ])

        # Large initial covariance to reflect uncertainty
        pos_var = 1e6    # (1000 m)^2
        vel_var = 1e4    # (100 m/s)^2
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var])
        self.initialized = True

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """EKF prediction step.

        Propagates state and covariance forward by dt using constant-velocity
        dynamics and piecewise-white acceleration noise.

        Args:
            dt: time step (s), must be > 0
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # State transition matrix (constant velocity)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Process noise covariance (continuous white-noise acceleration model)
        dt2 = dt * dt
        dt3 = dt2 * dt
        q = self.q_std**2
        Q = q * np.array([
            [dt3 / 3.0, 0.0,       dt2 / 2.0, 0.0      ],
            [0.0,       dt3 / 3.0, 0.0,       dt2 / 2.0],
            [dt2 / 2.0, 0.0,       dt,        0.0      ],
            [0.0,       dt2 / 2.0, 0.0,       dt       ],
        ])

        # Propagate mean and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self, z_lam: float, r_M: np.ndarray):
        """EKF update step with a scalar LOS azimuth angle measurement.

        Measurement function:
            h(x) = atan2(tgt_y - M_y, tgt_x - M_x)

        Jacobian:
            H = [-dy / rho^2,  dx / rho^2, 0, 0]

        Covariance update uses the Joseph form:
            K     = P_pred * H' * (H * P_pred * H' + R)^{-1}
            IKH   = I - K * H
            P_new = IKH * P_pred * IKH' + K * R * K'

        Args:
            z_lam: measured LOS azimuth angle (rad)
            r_M:   current missile position [2] (m)

        Returns:
            x_est: updated state estimate [4]
            P:     updated covariance [4x4]
        """
        z_lam = float(z_lam)
        r_M = np.asarray(r_M, dtype=float).ravel()

        Mx = r_M[0]
        My = r_M[1]

        tgt_x = self.x[0]
        tgt_y = self.x[1]

        dx = tgt_x - Mx
        dy = tgt_y - My
        rho2 = dx**2 + dy**2

        # Guard against degenerate geometry (target on top of missile)
        RHO2_MIN = 1.0  # 1 m^2 minimum
        if rho2 < RHO2_MIN:
            rho2 = RHO2_MIN

        # Predicted measurement
        h_x = float(np.arctan2(dy, dx))

        # Measurement Jacobian [1x4]
        H = np.array([[-dy / rho2, dx / rho2, 0.0, 0.0]])

        # Measurement noise covariance [1x1]
        R_mat = np.array([[self.r_std**2]])

        # Innovation (with angle wrapping to [-pi, pi])
        inn = z_lam - h_x
        inn = float(np.arctan2(np.sin(inn), np.cos(inn)))

        # Innovation covariance
        S = H @ self.P @ H.T + R_mat   # [1x1]

        # Kalman gain [4x1]
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K.ravel() * inn

        # Joseph form covariance update (numerically stable)
        I = np.eye(self.n)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_mat @ K.T

        # Force symmetry to avoid numerical drift
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_target_estimate(self):
        """Get current target state estimate.

        Returns:
            position [2]: estimated target position (m)
            velocity [2]: estimated target velocity (m/s)
            covariance [4x4]: full state covariance
        """
        return self.x[:2].copy(), self.x[2:].copy(), self.P.copy()

    def reset(self) -> None:
        """Reset filter to uninitialised state."""
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False
