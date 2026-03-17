"""Missile seeker (sensor) model.

Models an IR/RF seeker that measures LOS angle and rate to target.
Includes gimbal dynamics, noise, and FOV limits.

References:
    Zarchan Ch. 11-12
"""
import numpy as np


class SeekerModel:
    """Seeker model with noise and gimbal dynamics.

    Measures Line-of-Sight (LOS) angle and rate with:
    - Additive Gaussian noise on angle and rate
    - First-order gimbal lag
    - Field of view limitation

    Args:
        angle_noise_std: angle measurement noise std (rad), default 3 mrad
        rate_noise_std: rate measurement noise std (rad/s), default 3 mrad/s
        gimbal_tau: gimbal time constant (s), default 0.02
        fov_half: half field of view (rad), default 60 deg
    """

    def __init__(
        self,
        angle_noise_std: float = 0.003,
        rate_noise_std: float = 0.003,
        gimbal_tau: float = 0.02,
        fov_half: float = np.radians(60.0),
    ):
        self.angle_noise_std = float(angle_noise_std)
        self.rate_noise_std = float(rate_noise_std)
        self.gimbal_tau = float(gimbal_tau)
        self.fov_half = float(fov_half)

        # Gimbal state: tracks LOS angle through first-order lag
        # gimbal_angle[0] = azimuth lag state, [1] = elevation lag state
        self.gimbal_angle = np.zeros(2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_true_los(r_M: np.ndarray, v_M: np.ndarray,
                          r_T: np.ndarray, v_T: np.ndarray):
        """Compute true LOS angles and rates.

        Returns:
            lam_az, lam_el: LOS azimuth and elevation angles (rad)
            lam_dot_az, lam_dot_el: LOS angle rates (rad/s)
            R: range (m)
        """
        R_vec = r_T - r_M          # [3] target relative to missile
        R_dot_vec = v_T - v_M      # [3] relative velocity

        R = float(np.linalg.norm(R_vec))
        R_SMALL = 1.0              # 1 m guard

        if R < R_SMALL:
            return 0.0, 0.0, 0.0, 0.0, R

        Rx, Ry, Rz = R_vec
        Rdx, Rdy, Rdz = R_dot_vec

        rho = float(np.sqrt(Rx**2 + Ry**2))  # horizontal range

        lam_az = float(np.arctan2(Ry, Rx))
        lam_el = float(np.arctan2(-Rz, max(rho, 1e-6)))

        # LOS rates from differentiation of spherical angle expressions
        if rho < 1e-6:
            Omega_LOS = np.cross(R_vec, R_dot_vec) / (R * R)
            lam_dot_az = 0.0
            # Extract signed elevation rate: project Omega onto the local
            # azimuth unit vector (which carries elevation-rate information
            # when the LOS is near-vertical).  For vertical LOS the elevation
            # change is in the horizontal plane, so use Omega's horizontal
            # component with proper sign.
            if np.linalg.norm(Omega_LOS) > 1e-12:
                # Use y-component of Omega as signed proxy for elevation rate
                # (positive Omega_y corresponds to positive d(lam_el)/dt)
                lam_dot_el = float(np.sqrt(Omega_LOS[0]**2 + Omega_LOS[1]**2))
                # Sign: positive elevation rate when target moves upward
                lam_dot_el *= np.sign(Omega_LOS[1]) if abs(Omega_LOS[1]) > abs(Omega_LOS[0]) else np.sign(-Omega_LOS[0])
            else:
                lam_dot_el = 0.0
        else:
            lam_dot_az = float((Rx * Rdy - Ry * Rdx) / (rho**2))
            lam_dot_el = float(
                (-Rdz * rho**2 + Rz * (Rx * Rdx + Ry * Rdy)) / (R**2 * rho)
            )

        return lam_az, lam_el, lam_dot_az, lam_dot_el, R

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        r_M: np.ndarray,
        v_M: np.ndarray,
        r_T: np.ndarray,
        v_T: np.ndarray,
        dt: float,
    ) -> dict:
        """Get seeker measurements.

        Args:
            r_M, v_M: missile position/velocity [3] (m, m/s)
            r_T, v_T: target position/velocity [3] (m, m/s)
            dt: time step (s)

        Returns:
            dict with:
                lam_az      : measured azimuth LOS angle (rad)
                lam_el      : measured elevation LOS angle (rad)
                lam_dot_az  : measured azimuth LOS rate (rad/s)
                lam_dot_el  : measured elevation LOS rate (rad/s)
                in_fov      : bool, whether target is within FOV
                R_est       : estimated range (m) - for active seeker
        """
        r_M = np.asarray(r_M, dtype=float)
        v_M = np.asarray(v_M, dtype=float)
        r_T = np.asarray(r_T, dtype=float)
        v_T = np.asarray(v_T, dtype=float)

        # 1. Compute true LOS angles and rates
        lam_az_true, lam_el_true, lam_dot_az_true, lam_dot_el_true, R = (
            self._compute_true_los(r_M, v_M, r_T, v_T)
        )

        # 2. Apply gimbal lag: first-order discrete filter
        #    gimbal_angle(k+1) = alpha * gimbal_angle(k) + (1 - alpha) * lam_true
        #    where alpha = exp(-dt / tau)
        if self.gimbal_tau > 1e-9 and dt > 0.0:
            alpha = float(np.exp(-dt / self.gimbal_tau))
        else:
            alpha = 0.0  # instantaneous tracking (no lag)

        # Angle-aware lag filter: wrap the error to [-pi, pi] before interpolation
        # to avoid the 358-degree slew problem at the +/-pi boundary.
        az_err_wrap = np.arctan2(
            np.sin(lam_az_true - self.gimbal_angle[0]),
            np.cos(lam_az_true - self.gimbal_angle[0]),
        )
        el_err_wrap = np.arctan2(
            np.sin(lam_el_true - self.gimbal_angle[1]),
            np.cos(lam_el_true - self.gimbal_angle[1]),
        )
        self.gimbal_angle[0] = self.gimbal_angle[0] + (1.0 - alpha) * az_err_wrap
        self.gimbal_angle[1] = self.gimbal_angle[1] + (1.0 - alpha) * el_err_wrap

        # Gimbal-lagged angles
        lam_az_lag = float(self.gimbal_angle[0])
        lam_el_lag = float(self.gimbal_angle[1])

        # LOS rates from the gimbal response derivative approximation:
        # rate is implicitly represented by the angle derivative; for a first-
        # order filter with lag tau the output rate is (1/tau)*(lam_true - lam_lag)
        if self.gimbal_tau > 1e-9:
            # Wrap angle differences to [-pi, pi] to avoid spurious rates
            az_diff = np.arctan2(np.sin(lam_az_true - lam_az_lag),
                                 np.cos(lam_az_true - lam_az_lag))
            el_diff = np.arctan2(np.sin(lam_el_true - lam_el_lag),
                                 np.cos(lam_el_true - lam_el_lag))
            lam_dot_az_lag = az_diff / self.gimbal_tau
            lam_dot_el_lag = el_diff / self.gimbal_tau
        else:
            lam_dot_az_lag = lam_dot_az_true
            lam_dot_el_lag = lam_dot_el_true

        # 3. Add Gaussian measurement noise
        lam_az_meas = lam_az_lag + np.random.normal(0.0, self.angle_noise_std)
        lam_el_meas = lam_el_lag + np.random.normal(0.0, self.angle_noise_std)
        lam_dot_az_meas = lam_dot_az_lag + np.random.normal(0.0, self.rate_noise_std)
        lam_dot_el_meas = lam_dot_el_lag + np.random.normal(0.0, self.rate_noise_std)

        # 4. Check FOV
        #    Two conditions for target visibility:
        #    a) Gimbal mechanical limit: the gimbal angle (where the seeker
        #       is physically pointing) must be within the half-cone FOV.
        #    b) Gimbal tracking error: the difference between where the
        #       gimbal points and where the target actually is must be small
        #       enough for the detector to resolve the target.
        gimbal_off_boresight = float(np.sqrt(
            self.gimbal_angle[0]**2 + self.gimbal_angle[1]**2
        ))
        az_err = abs(lam_az_lag - lam_az_true)
        el_err = abs(lam_el_lag - lam_el_true)
        in_fov = (gimbal_off_boresight <= self.fov_half) and (
            az_err < self.fov_half and el_err < self.fov_half
        )

        # Range estimate: use true range (active seeker) with small noise
        range_noise_std = max(R * 0.005, 1.0)  # 0.5% of range, min 1 m
        R_est = max(0.0, R + np.random.normal(0.0, range_noise_std))

        return dict(
            lam_az=float(lam_az_meas),
            lam_el=float(lam_el_meas),
            lam_dot_az=float(lam_dot_az_meas),
            lam_dot_el=float(lam_dot_el_meas),
            in_fov=bool(in_fov),
            R_est=float(R_est),
        )

    def reset(self) -> None:
        """Reset seeker state."""
        self.gimbal_angle = np.zeros(2)
