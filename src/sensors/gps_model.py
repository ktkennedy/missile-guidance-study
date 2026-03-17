"""GPS Receiver model.

Models GPS position and velocity measurements with:
- Independent Gaussian noise per NED axis
- Discrete update rate (e.g., 1 Hz)

References:
    Bar-Shalom Ch. 11, Groves "Principles of GNSS" Ch. 9
"""
import numpy as np


class GPSModel:
    """Simple GPS receiver model with white noise position and velocity errors.

    Position and velocity noise are modeled as zero-mean Gaussian, independent
    across NED axes and across time steps (no bias or Markov process).

    Args:
        pos_sigma:      position noise 1-sigma (m), applied independently per axis.
                        Default 5.0 m (typical urban GPS CEP ~8 m).
        vel_sigma:      velocity noise 1-sigma (m/s), applied independently per axis.
                        Default 0.1 m/s.
        update_rate_hz: GPS measurement update rate (Hz). Default 1 Hz.
        seed:           optional RNG seed for reproducible measurements.
    """

    def __init__(
        self,
        pos_sigma: float = 5.0,
        vel_sigma: float = 0.1,
        update_rate_hz: float = 1.0,
        seed=None,
    ):
        self.pos_sigma = float(pos_sigma)
        self.vel_sigma = float(vel_sigma)
        self.update_interval = 1.0 / float(update_rate_hz)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        true_pos_ned: np.ndarray,
        true_vel_ned: np.ndarray,
    ) -> tuple:
        """Get GPS position and velocity measurements with Gaussian noise.

        Args:
            true_pos_ned: true NED position [3] (m)
            true_vel_ned: true NED velocity [3] (m/s)

        Returns:
            pos_meas_ned: measured NED position [3] (m)
            vel_meas_ned: measured NED velocity [3] (m/s)
        """
        true_pos_ned = np.asarray(true_pos_ned, dtype=float)
        true_vel_ned = np.asarray(true_vel_ned, dtype=float)

        pos_meas_ned = true_pos_ned + self._rng.normal(0.0, self.pos_sigma, size=3)
        vel_meas_ned = true_vel_ned + self._rng.normal(0.0, self.vel_sigma, size=3)

        return pos_meas_ned, vel_meas_ned

    def is_update_due(self, t: float) -> bool:
        """Return True when a GPS update should be processed at simulation time t.

        Uses modulo arithmetic so updates fire at t = 0, T, 2T, ... where
        T = 1/update_rate_hz.  A small epsilon (1 µs) handles floating-point
        rounding when t is an exact multiple of T.

        Args:
            t: current simulation time (s)

        Returns:
            True if a GPS measurement should be generated at this time step.
        """
        eps = 1e-6  # 1 µs tolerance
        remainder = t % self.update_interval
        return remainder < eps or remainder > (self.update_interval - eps)
