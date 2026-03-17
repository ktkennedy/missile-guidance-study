"""Fin actuator dynamics model.

Models a second-order actuator with rate and position limits.

References:
    Garnell, P. "Guided Weapon Control Systems", 2nd Ed., 1980
"""
import numpy as np


class FinActuator:
    """Second-order fin actuator with saturation limits.

    Transfer function: delta/delta_cmd = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

    With position limit: |delta| <= delta_max
    With rate limit: |delta_dot| <= rate_max

    Args:
        wn: natural frequency (rad/s), default 120
        zeta: damping ratio, default 0.7
        delta_max: position limit (rad), default 25 deg = 0.4363 rad
        rate_max: rate limit (rad/s), default 300 deg/s = 5.236 rad/s
    """

    def __init__(self, wn=120.0, zeta=0.7, delta_max=np.radians(25.0),
                 rate_max=np.radians(300.0)):
        self.wn = wn
        self.zeta = zeta
        self.delta_max = delta_max
        self.rate_max = rate_max

        # State: [delta, delta_dot]
        self.delta = 0.0
        self.delta_dot = 0.0

    def _derivatives(self, delta, delta_dot, delta_cmd):
        """Compute 2nd-order actuator derivatives.

        Args:
            delta: current fin position (rad)
            delta_dot: current fin rate (rad/s)
            delta_cmd: commanded fin position (rad)

        Returns:
            (d_delta, d_delta_dot): state derivatives
        """
        d_delta = delta_dot
        d_delta_dot = (self.wn ** 2) * (delta_cmd - delta) - 2.0 * self.zeta * self.wn * delta_dot
        return d_delta, d_delta_dot

    def update(self, delta_cmd, dt):
        """Update actuator state and return actual fin deflection.

        Uses RK4 integration for the 2nd-order ODE:
            delta_ddot = wn^2 * (delta_cmd - delta) - 2*zeta*wn*delta_dot

        Rate limiting is applied to delta_dot before integration propagation.
        Position limiting is applied to delta after each integration stage.

        Args:
            delta_cmd: commanded fin deflection (rad)
            dt: time step (s)

        Returns:
            delta: actual fin deflection after limits (rad)
        """
        d = self.delta
        dd = self.delta_dot

        # RK4 stage 1
        k1d, k1dd = self._derivatives(d, dd, delta_cmd)

        # RK4 stage 2
        d2 = d + 0.5 * dt * k1d
        dd2 = np.clip(dd + 0.5 * dt * k1dd, -self.rate_max, self.rate_max)
        k2d, k2dd = self._derivatives(d2, dd2, delta_cmd)

        # RK4 stage 3
        d3 = d + 0.5 * dt * k2d
        dd3 = np.clip(dd + 0.5 * dt * k2dd, -self.rate_max, self.rate_max)
        k3d, k3dd = self._derivatives(d3, dd3, delta_cmd)

        # RK4 stage 4
        d4 = d + dt * k3d
        dd4 = np.clip(dd + dt * k3dd, -self.rate_max, self.rate_max)
        k4d, k4dd = self._derivatives(d4, dd4, delta_cmd)

        # Combine increments
        new_delta = d + (dt / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        new_delta_dot = dd + (dt / 6.0) * (k1dd + 2.0 * k2dd + 2.0 * k3dd + k4dd)

        # Apply rate limit
        new_delta_dot = np.clip(new_delta_dot, -self.rate_max, self.rate_max)

        # Apply position limit
        new_delta = np.clip(new_delta, -self.delta_max, self.delta_max)

        self.delta = new_delta
        self.delta_dot = new_delta_dot

        return self.delta

    def reset(self):
        """Reset actuator state to zero."""
        self.delta = 0.0
        self.delta_dot = 0.0
