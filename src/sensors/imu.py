"""Inertial Measurement Unit (IMU) model.

Models tactical-grade gyroscopes and accelerometers with:
- Bias (Gauss-Markov process)
- White noise
- Scale factor errors (optional)

References:
    Bar-Shalom Ch. 11
"""
import numpy as np


class IMUModel:
    """Tactical-grade IMU model with realistic error characteristics.

    Gyroscope errors:
        - Bias: Gauss-Markov process with correlation time ~300 s
        - White noise: angular random walk
        - Typical: bias 1-10 deg/hr, ARW 0.1-1 deg/sqrt(hr)

    Accelerometer errors:
        - Bias: Gauss-Markov with correlation time ~600 s
        - White noise
        - Typical: bias 0.1-1 mg, VRW 0.01-0.1 m/s/sqrt(hr)

    The Gauss-Markov (first-order Markov) bias model is:
        b(k+1) = exp(-dt / tau) * b(k) + sigma_drive * w(k)
    where sigma_drive ensures the steady-state variance equals bias_std^2:
        sigma_drive = bias_std * sqrt(1 - exp(-2*dt/tau))

    Args:
        gyro_bias_std:  gyro bias 1-sigma (rad/s), default ~5 deg/hr
        gyro_noise_std: gyro white noise 1-sigma per root-second (rad/s),
                        default ~0.5 deg/hr per sqrt(hr) converted to rad/s
        gyro_tau:       gyro bias correlation time (s), default 300
        accel_bias_std: accel bias 1-sigma (m/s²), default ~0.5 mg
        accel_noise_std:accel white noise 1-sigma per root-second (m/s²),
                        default ~0.1 mg
        accel_tau:      accel bias correlation time (s), default 600
    """

    def __init__(
        self,
        gyro_bias_std: float = np.radians(5.0) / 3600.0,        # 5 deg/hr -> rad/s
        gyro_noise_std: float = np.radians(0.5) / np.sqrt(3600.0),  # 0.5 deg/sqrt(hr) -> rad/sqrt(s)
        gyro_tau: float = 300.0,
        accel_bias_std: float = 0.5e-3 * 9.81,
        accel_noise_std: float = 0.1e-3 * 9.81,
        accel_tau: float = 600.0,
    ):
        self.gyro_bias_std = float(gyro_bias_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.gyro_tau = float(gyro_tau)
        self.accel_bias_std = float(accel_bias_std)
        self.accel_noise_std = float(accel_noise_std)
        self.accel_tau = float(accel_tau)

        # Initialize bias states for 3 axes each from N(0, bias_std)
        self.gyro_bias = np.random.normal(0.0, self.gyro_bias_std, size=3)
        self.accel_bias = np.random.normal(0.0, self.accel_bias_std, size=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        omega_true: np.ndarray,
        accel_true: np.ndarray,
        dt: float,
    ):
        """Get IMU measurements with errors.

        Args:
            omega_true:  true angular velocity [3] (rad/s)
            accel_true:  true specific force [3] (m/s²)
            dt:          time step (s)

        Returns:
            omega_meas: measured angular velocity [3] (rad/s)
            accel_meas: measured specific force [3] (m/s²)
        """
        omega_true = np.asarray(omega_true, dtype=float)
        accel_true = np.asarray(accel_true, dtype=float)
        dt = float(dt)

        # 1. Update Gauss-Markov biases:
        #    b(k+1) = phi * b(k) + sigma_drive * w,  w ~ N(0, I)
        #    steady-state variance: var = sigma_drive^2 / (1 - phi^2) = bias_std^2
        #    => sigma_drive = bias_std * sqrt(1 - phi^2)

        # Gyroscope bias update
        if self.gyro_tau > 1e-9 and dt > 0.0:
            phi_g = float(np.exp(-dt / self.gyro_tau))
        else:
            phi_g = 0.0
        sigma_drive_g = self.gyro_bias_std * float(np.sqrt(max(0.0, 1.0 - phi_g**2)))
        self.gyro_bias = phi_g * self.gyro_bias + sigma_drive_g * np.random.randn(3)

        # Accelerometer bias update
        if self.accel_tau > 1e-9 and dt > 0.0:
            phi_a = float(np.exp(-dt / self.accel_tau))
        else:
            phi_a = 0.0
        sigma_drive_a = self.accel_bias_std * float(np.sqrt(max(0.0, 1.0 - phi_a**2)))
        self.accel_bias = phi_a * self.accel_bias + sigma_drive_a * np.random.randn(3)

        # 2. Add bias and white noise to true values
        #    White noise is scaled by 1/sqrt(dt) to represent a continuous-time
        #    noise process (power spectral density) sampled at rate 1/dt.
        #    When dt is very small, guard against division by zero.
        if dt > 1e-9:
            noise_scale = 1.0 / float(np.sqrt(dt))
        else:
            noise_scale = 0.0

        omega_meas = (
            omega_true
            + self.gyro_bias
            + self.gyro_noise_std * noise_scale * np.random.randn(3)
        )
        accel_meas = (
            accel_true
            + self.accel_bias
            + self.accel_noise_std * noise_scale * np.random.randn(3)
        )

        return omega_meas, accel_meas

    def reset(self) -> None:
        """Reset IMU biases (re-draw from prior distribution)."""
        self.gyro_bias = np.random.normal(0.0, self.gyro_bias_std, size=3)
        self.accel_bias = np.random.normal(0.0, self.accel_bias_std, size=3)
