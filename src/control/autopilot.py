"""Missile autopilot (flight control system).

Implements a two-loop acceleration autopilot:
- Inner loop: rate gyro feedback for damping augmentation
- Outer loop: PI controller for acceleration tracking

The architecture:
    a_error = a_cmd - a_measured
    delta_outer = Kp * a_error + Ki * integral(a_error)
    delta_cmd = delta_outer - Kq * q_measured

Validated gains (at Mach 2.0, sea level):
    Kq = -0.20 (rate feedback; negative because M_delta < 0 convention)
    Kp =  0.001 (proportional on accel error)
    Ki =  0.01  (integral on accel error)

    With M_delta < 0, a positive fin command drives q negative. Setting
    Kq = -0.20 in ``delta_cmd = delta_outer - Kq * q`` adds positive
    damping and stabilises the inner loop.

Performance: SS tracking < 0.3% for +-8g commands

References:
    Garnell, P. "Guided Weapon Control Systems"
    Blakelock, J.H. "Automatic Control of Aircraft and Missiles"
"""
import numpy as np

from .autopilot_interface import AutopilotInterface


class TwoLoopAutopilot(AutopilotInterface):
    """Two-loop acceleration autopilot.

    Inner loop provides pitch rate damping via direct rate gyro feedback.
    Outer loop tracks normal acceleration commands using PI control.

    Args:
        Kq: rate feedback gain (default -0.20; negative because M_delta < 0)
        Kp: proportional gain on acceleration error (default 0.001)
        Ki: integral gain on acceleration error (default 0.01)
        integrator_limit: anti-windup limit (rad, default 0.2)
    """

    def __init__(self, Kq=-0.20, Kp=0.001, Ki=0.01, integrator_limit=0.2):
        self.Kq = Kq
        self.Kp = Kp
        self.Ki = Ki
        self.integrator_limit = integrator_limit
        self.integral = 0.0

    def compute(self, a_cmd, a_measured, q_measured, dt):
        """Compute fin deflection command.

        Args:
            a_cmd: commanded normal acceleration (m/s^2)
            a_measured: measured normal acceleration from accelerometer (m/s^2)
            q_measured: measured pitch rate from gyro (rad/s)
            dt: time step (s)

        Returns:
            delta_cmd: fin deflection command (rad)
        """
        # Outer loop: PI on acceleration error
        a_error = a_cmd - a_measured
        self.integral += a_error * dt

        # Anti-windup: clamp integral so Ki*integral stays within integrator_limit
        integral_limit = self.integrator_limit / self.Ki if self.Ki != 0.0 else np.inf
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)

        delta_outer = self.Kp * a_error + self.Ki * self.integral

        # Inner loop: rate feedback (subtract to increase damping)
        delta_cmd = delta_outer - self.Kq * q_measured

        return delta_cmd

    def reset(self):
        """Reset integrator state to zero."""
        self.integral = 0.0


class AirframeShortPeriod:
    """Short-period dynamics model for standalone autopilot testing.

    2-state model: [alpha, q]
        alpha_dot = Z_alpha * alpha + q + Z_delta * delta
        q_dot     = M_alpha * alpha + M_q * q + M_delta * delta

    Normal acceleration: a_z = V * (Z_alpha * alpha + Z_delta * delta)

    Generic missile stability derivatives at Mach 2.0:
        M_alpha = -200 /s^2 (static stability, negative = stable)
        M_q     = -4   /s   (pitch damping)
        M_delta = -80  /s^2 (control effectiveness)
        Z_alpha = -3   /s   (normal force due to alpha)
        Z_delta = -0.2 /s   (normal force due to fin deflection)
        V       =  680 m/s  (flight speed at Mach 2.0)

    Args:
        M_alpha: static stability derivative (/s^2)
        M_q: pitch damping derivative (/s)
        M_delta: control effectiveness derivative (/s^2)
        Z_alpha: normal force due to AoA (/s)
        Z_delta: normal force due to fin deflection (/s)
        V: flight speed (m/s)
    """

    def __init__(self, M_alpha=-200.0, M_q=-4.0, M_delta=-80.0,
                 Z_alpha=-3.0, Z_delta=-0.2, V=680.0):
        self.M_alpha = M_alpha
        self.M_q = M_q
        self.M_delta = M_delta
        self.Z_alpha = Z_alpha
        self.Z_delta = Z_delta
        self.V = V

        self.alpha = 0.0
        self.q = 0.0

    def derivatives(self, alpha, q, delta):
        """Compute state derivatives for the short-period model.

        Args:
            alpha: angle of attack (rad)
            q: pitch rate (rad/s)
            delta: fin deflection (rad)

        Returns:
            (d_alpha, d_q): time derivatives of [alpha, q]
        """
        d_alpha = self.Z_alpha * alpha + q + self.Z_delta * delta
        d_q = self.M_alpha * alpha + self.M_q * q + self.M_delta * delta
        return d_alpha, d_q

    def update(self, delta, dt):
        """RK4 integration step.

        Args:
            delta: fin deflection (rad)
            dt: time step (s)

        Returns:
            (q, a_z): pitch rate (rad/s) and normal acceleration (m/s^2)
        """
        # RK4 for [alpha, q]
        k1a, k1q = self.derivatives(self.alpha, self.q, delta)
        k2a, k2q = self.derivatives(self.alpha + 0.5 * dt * k1a,
                                    self.q + 0.5 * dt * k1q, delta)
        k3a, k3q = self.derivatives(self.alpha + 0.5 * dt * k2a,
                                    self.q + 0.5 * dt * k2q, delta)
        k4a, k4q = self.derivatives(self.alpha + dt * k3a,
                                    self.q + dt * k3q, delta)

        self.alpha += (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        self.q += (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)

        a_z = self.V * (self.Z_alpha * self.alpha + self.Z_delta * delta)
        return self.q, a_z

    def reset(self):
        """Reset airframe states to zero."""
        self.alpha = 0.0
        self.q = 0.0
