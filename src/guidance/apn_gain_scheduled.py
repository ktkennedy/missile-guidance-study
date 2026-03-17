"""Gain-Scheduled Augmented Proportional Navigation.

Extends APN with time-varying navigation gain N(t_go) schedule,
parameterized by wavelet coefficients for Bayesian Optimization.
"""
import numpy as np
import pywt
from guidance.proportional_navigation import compute_los_geometry


class GainScheduledAPN:
    """APN with time-varying N(t_go) gain schedule.

    Interface matches ProportionalNavigation for drop-in replacement.

    Args:
        n_schedule: one of:
            - None: use constant n_default
            - callable: n_schedule(t_go) -> N value
            - np.ndarray: wavelet coefficients to reconstruct N(t_go) schedule
        n_default: fallback/constant N value (default 4.0)
        a_max: max acceleration magnitude (m/s^2, default 400.0)
        wavelet: wavelet type for reconstruction (default 'db4')
        level: wavelet decomposition level (default 4)
        t_go_max: maximum t_go for schedule definition (s, default 10.0)
        n_points: number of time points for wavelet reconstruction (default 100)
    """

    def __init__(self, n_schedule=None, n_default=4.0, a_max=400.0,
                 wavelet='db4', level=4, t_go_max=10.0, n_points=100):
        self.n_default = n_default
        self.a_max = a_max
        self.wavelet = wavelet
        self.level = level
        self.t_go_max = t_go_max
        self.n_points = n_points

        # Build the N(t_go) function
        if n_schedule is None:
            self._n_func = lambda tgo: n_default
            self._is_constant = True
        elif callable(n_schedule):
            self._n_func = n_schedule
            self._is_constant = False
        elif isinstance(n_schedule, np.ndarray):
            self._build_wavelet_schedule(n_schedule)
            self._is_constant = False
        else:
            raise ValueError(f"n_schedule must be None, callable, or ndarray")

    def _build_wavelet_schedule(self, coeffs):
        """Reconstruct N(t_go) from wavelet coefficients.

        Uses pywt to reconstruct a smooth schedule from wavelet coefficients.
        The schedule is clipped to [2, 8] for physical reasonableness.

        Args:
            coeffs: 1D array of wavelet coefficients
        """
        # Reconstruct signal from wavelet coefficients
        # pywt.waverec expects a list of coefficient arrays [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # We need to split the flat coefficient vector into the right structure

        # For db4, level 4, with n_points=100:
        # The coefficient structure depends on signal length
        # We'll use a simple approach: treat coeffs as the detail coefficients
        # and reconstruct with zero approximation (centered around n_default)

        # Create coefficient arrays for reconstruction
        # First, figure out expected lengths at each level
        dummy_signal = np.ones(self.n_points) * self.n_default
        coeff_list = pywt.wavedec(dummy_signal, self.wavelet, level=self.level)
        coeff_lengths = [len(c) for c in coeff_list]

        # Build coefficient list from flat array
        # Use n_default as base, add perturbation from coefficients
        recon_coeffs = []
        idx = 0
        for i, length in enumerate(coeff_lengths):
            if idx + length <= len(coeffs):
                recon_coeffs.append(coeffs[idx:idx + length])
                idx += length
            else:
                # Pad with zeros if not enough coefficients provided
                remaining = len(coeffs) - idx
                if remaining > 0:
                    arr = np.zeros(length)
                    arr[:remaining] = coeffs[idx:idx + remaining]
                    recon_coeffs.append(arr)
                    idx = len(coeffs)
                else:
                    recon_coeffs.append(np.zeros(length))

        # Reconstruct and clip to [2, 8]
        n_profile = pywt.waverec(recon_coeffs, self.wavelet)[:self.n_points]
        n_profile = np.clip(n_profile, 2.0, 8.0)

        # Build interpolation: t_go in [0, t_go_max]
        t_go_grid = np.linspace(0, self.t_go_max, self.n_points)

        def interp_func(tgo):
            tgo = float(np.clip(tgo, 0, self.t_go_max))
            return float(np.interp(tgo, t_go_grid, n_profile))

        self._n_func = interp_func
        self._n_profile = n_profile
        self._t_go_grid = t_go_grid

    def get_n(self, t_go):
        """Get navigation gain at given time-to-go."""
        return self._n_func(t_go)

    def get_wavelet_dim(self):
        """Get the total number of wavelet coefficients for this configuration.

        Useful for BO to know the parameter space dimension.
        """
        dummy = np.ones(self.n_points) * self.n_default
        coeffs = pywt.wavedec(dummy, self.wavelet, level=self.level)
        return sum(len(c) for c in coeffs)

    def compute_pitch_yaw(self, r_M, v_M, r_T, v_T, n_T_est=None):
        """Compute guidance command with time-varying N.

        Same signature as ProportionalNavigation.compute_pitch_yaw().

        Args:
            r_M    : missile position [3] NED (m)
            v_M    : missile velocity [3] NED (m/s)
            r_T    : target position [3] NED (m)
            v_T    : target velocity [3] NED (m/s)
            n_T_est: estimated target acceleration [3] (m/s^2)

        Returns:
            (a_pitch, a_yaw): acceleration commands (m/s^2)
        """
        r_M = np.asarray(r_M, dtype=float)
        v_M = np.asarray(v_M, dtype=float)
        r_T = np.asarray(r_T, dtype=float)
        v_T = np.asarray(v_T, dtype=float)

        geo = compute_los_geometry(r_M, v_M, r_T, v_T)

        R = geo['R']
        V_c = geo['V_c']
        lam_dot_az = geo['lam_dot_az']
        lam_dot_el = geo['lam_dot_el']
        lam_el = geo['lam_el']

        if R < 1e-6:
            return 0.0, 0.0

        # Time-to-go
        t_go = geo['t_go']

        # Get time-varying N
        N = self.get_n(t_go)

        cos_el = float(np.cos(lam_el))
        effective_speed = max(V_c, 0.0)

        # Base PN terms
        a_pitch = N * effective_speed * lam_dot_el
        a_yaw = N * effective_speed * lam_dot_az * cos_el

        # APN augmentation
        if n_T_est is not None:
            n_T_est = np.asarray(n_T_est, dtype=float)
            lam_az = geo['lam_az']
            sin_el = float(np.sin(lam_el))
            cos_az = float(np.cos(lam_az))
            sin_az = float(np.sin(lam_az))
            e_el = np.array([-sin_el * cos_az, -sin_el * sin_az, -cos_el])
            e_az = np.array([-sin_az, cos_az, 0.0])
            nT_el = float(np.dot(n_T_est, e_el))
            nT_az = float(np.dot(n_T_est, e_az))
            a_pitch += (N / 2.0) * nT_el
            a_yaw += (N / 2.0) * nT_az

        # Saturate
        a_vec = np.array([a_pitch, a_yaw])
        mag = float(np.linalg.norm(a_vec))
        if mag > self.a_max:
            scale = self.a_max / mag
            a_pitch *= scale
            a_yaw *= scale

        return float(a_pitch), float(a_yaw)
