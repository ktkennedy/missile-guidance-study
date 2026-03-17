"""Three-loop acceleration autopilot.

Implements the three-loop topology from Abd-Elatif et al. (Paper 1):
- Inner loop  (rate gyro):          pitch rate feedback with gain Kg
- Middle loop (synthetic stability): body rate feedback with gain K_omega
- Outer loop  (acceleration):       accelerometer feedback with gain KA

Topology (codebase convention where M_delta < 0):
    a_cmd → (+) → [KA + Ki/s] → (+) → fin actuator → airframe
             |-(-)               |(+)
             a_meas        (K_omega + Kg) * q_meas

    Rate feedback is ADDED because M_delta < 0: adding positive q to delta
    produces a nose-down moment (opposing nose-up motion), which is damping.
    This matches TwoLoopAutopilot where Kq = -0.20 and delta = outer - Kq*q
    effectively adds +0.20*q.

Sign convention (CODEBASE):
    Negative derivatives indicate stability, e.g. M_alpha < 0 means stable.
    Paper 1 uses the opposite sign convention (positive = stable).
    The static method _to_paper_convention() handles the conversion.

References:
    Abd-Elatif et al., "Three-Loop Autopilot Design for Tail-Controlled Missiles"
    Blakelock, J.H., "Automatic Control of Aircraft and Missiles"
"""
import warnings
import numpy as np

from .autopilot_interface import AutopilotInterface


class ThreeLoopAutopilot(AutopilotInterface):
    """Three-loop acceleration autopilot (Abd-Elatif et al., Paper 1).

    Gain design uses the codebase convention directly with an addition
    topology for rate feedback (physically correct when M_delta < 0).

    The inner loop (Kg) provides pitch damping augmentation.
    The middle loop (K_omega) provides additional stability margin.
    The outer loop (KA + Ki) tracks acceleration commands.

    Args:
        M_alpha:          static stability derivative (/s^2), codebase convention (negative = stable)
        M_q:              pitch damping derivative (/s)         (negative in both conventions)
        M_delta:          control effectiveness (/s^2),         codebase convention (negative = stable)
        Z_alpha:          normal force due to AoA (/s),         codebase convention (negative = stable)
        Z_delta:          normal force due to fin deflection (/s), codebase convention (negative = stable)
        V:                flight speed (m/s)
        omega:            desired closed-loop bandwidth (rad/s)
        zeta:             desired closed-loop damping ratio
        tau:              desired closed-loop time constant (s)
        omega_ACT:        actuator bandwidth (rad/s), used for crossover check
        Ki:               integral gain on acceleration error (rad/s per m/s^2)
        integrator_limit: anti-windup saturation limit (rad)
    """

    def __init__(self, M_alpha=-240.0, M_q=-4.0, M_delta=-204.0,
                 Z_alpha=-1.17, Z_delta=-0.239, V=914.0,
                 omega=20.0, zeta=0.7, tau=0.5,
                 omega_ACT=150.0, Ki=0.02, integrator_limit=0.3):

        # Store airframe parameters for reference
        self._M_alpha = M_alpha
        self._M_delta = M_delta
        self._V = V

        # ------------------------------------------------------------------ #
        # Gain design (codebase convention, addition topology)               #
        #                                                                     #
        # With delta = outer + K*q in codebase convention (M_delta < 0):     #
        #   M_q_eff = M_q + M_delta * K  (more negative → more damping)      #
        # ------------------------------------------------------------------ #

        # Inner loop: rate gyro feedback for damping
        # Target: M_q_eff = -2*zeta*omega
        # Kg = (-2*zeta*omega - M_q) / M_delta
        Kg = (-2.0 * zeta * omega - M_q) / M_delta

        # Middle loop: additional rate feedback for stability margin
        # Provides ~10% additional damping beyond inner loop
        K_omega = 0.1 * abs(Kg)

        # Outer loop: acceleration feedback gain
        # Sets closed-loop bandwidth to ~1/tau
        KA = 1.0 / (tau * abs(Z_alpha) * V)

        # DC gain: unity (integral handles steady-state tracking)
        KDC = 1.0

        # ------------------------------------------------------------------ #
        # Store gains and design parameters                                   #
        # ------------------------------------------------------------------ #
        self.Kg = Kg
        self.K_omega = K_omega
        self.KDC = KDC
        self.KA = KA
        self.Ki = Ki
        self.integrator_limit = integrator_limit

        # Integrator state
        self.integral = 0.0

        # ------------------------------------------------------------------ #
        # Crossover frequency check                                           #
        # ------------------------------------------------------------------ #
        omega_CR = omega
        if omega_CR > omega_ACT / 3.0:
            warnings.warn(
                f"Crossover frequency omega_CR={omega_CR:.1f} rad/s exceeds "
                f"omega_ACT/3={omega_ACT / 3.0:.1f} rad/s.  "
                "Actuator bandwidth separation may be insufficient.",
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # Gain sign validation                                                #
        # ------------------------------------------------------------------ #
        self._validate_gain_signs(KA, K_omega, Kg, KDC)

    # ---------------------------------------------------------------------- #
    # Interface implementation                                                #
    # ---------------------------------------------------------------------- #

    def compute(self, a_cmd: float, a_measured: float, q_measured: float, dt: float) -> float:
        """Compute fin deflection command for one time step.

        Rate feedback is ADDED (not subtracted) because in codebase convention
        M_delta < 0: adding positive q to delta produces opposing moment.
        This is equivalent to TwoLoopAutopilot's ``delta - Kq*q`` where
        Kq = -0.20 (subtracting a negative = adding a positive).

        Args:
            a_cmd:       commanded normal acceleration (m/s^2)
            a_measured:  measured normal acceleration from accelerometer (m/s^2)
            q_measured:  measured pitch rate from rate gyro (rad/s)
            dt:          time step (s)

        Returns:
            Fin deflection command (rad).
        """
        # Outer loop: acceleration error
        a_error = self.KDC * a_cmd - a_measured

        # Integral for steady-state accuracy (anti-windup: clamp so Ki*integral <= limit)
        self.integral += a_error * dt
        if self.Ki != 0.0:
            int_clamp = self.integrator_limit / self.Ki
        else:
            int_clamp = 1e10
        self.integral = np.clip(self.integral, -int_clamp, int_clamp)

        # Outer loop output (PI on acceleration error)
        delta_outer = self.KA * a_error + self.Ki * self.integral

        # Rate feedback: ADD in codebase convention (M_delta < 0)
        # Middle loop (K_omega) + Inner loop (Kg) both use q_measured
        delta_cmd = delta_outer + (self.K_omega + self.Kg) * q_measured

        return delta_cmd

    def reset(self) -> None:
        """Reset integrator state to zero."""
        self.integral = 0.0

    # ---------------------------------------------------------------------- #
    # Helper / static methods                                                 #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _to_paper_convention(M_alpha: float, M_delta: float,
                             Z_alpha: float, Z_delta: float):
        """Convert codebase derivatives to Paper 1 sign convention.

        Codebase convention: negative derivatives indicate stability.
        Paper 1 convention: positive derivatives indicate stability.

        Negates all four values.

        Returns:
            (M_alpha_p, M_delta_p, Z_alpha_p, Z_delta_p) in Paper 1 convention.
        """
        return -M_alpha, -M_delta, -Z_alpha, -Z_delta

    @staticmethod
    def _validate_gain_signs(KA: float, K_omega: float, Kg: float, KDC: float) -> None:
        """Warn if any computed gain has an unexpected sign.

        All four gains should be positive for a stable missile with
        correct codebase-convention derivatives (negative = stable).

        Args:
            KA:      outer loop acceleration gain
            K_omega: middle loop synthetic stability gain
            Kg:      inner loop rate gyro gain
            KDC:     DC gain for steady-state tracking
        """
        expected = {"KA": KA, "K_omega": K_omega, "Kg": Kg, "KDC": KDC}
        for name, value in expected.items():
            if value <= 0.0:
                warnings.warn(
                    f"ThreeLoopAutopilot: gain {name}={value:.4f} is non-positive. "
                    "Expected all gains > 0 for a stable missile.  "
                    "Check airframe derivatives and design parameters.",
                    stacklevel=3,
                )
