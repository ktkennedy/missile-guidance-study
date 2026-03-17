"""
Linear aerodynamic coefficient model for a generic tactical missile.

Conventions:
- Body frame: x-forward (roll axis), y-right, z-down
- Positive alpha: nose above velocity vector (w > 0)
- Positive beta : nose to the right of velocity vector (v > 0)
- Positive delta_e: trailing-edge-up (generates negative Cm, nose-down)
- All forces in body frame (N), all moments about the centre of mass (N·m)

The model uses a simplified linear aerodynamic representation appropriate for
small-to-moderate angles of attack (|alpha|, |beta| < ~20 deg).
"""

import numpy as np
from .aerodynamics_interface import AerodynamicsInterface


class MissileAerodynamics(AerodynamicsInterface):
    """Linear aerodynamic coefficient model for a generic tactical missile.

    All force coefficients are nondimensionalised by q_bar * S_ref.
    All moment coefficients are nondimensionalised by q_bar * S_ref * d_ref.

    Pitch and yaw channels are treated as symmetric (axisymmetric missile
    assumption), so Cn_beta = -Cm_alpha and yaw damping = pitch damping.

    Attributes:
        S_ref  (float): Reference area (m²), typically pi/4 * d_ref².
        d_ref  (float): Reference length / diameter (m).
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Reference geometry
        # ------------------------------------------------------------------
        self.S_ref: float = 0.01267   # m²  (pi/4 * 0.127²)
        self.d_ref: float = 0.127     # m   reference diameter

        # ------------------------------------------------------------------
        # Force coefficients (body-axis)
        # ------------------------------------------------------------------
        self.CL_alpha: float = 18.5   # /rad  lift-curve slope
        self.CD_0: float = 0.35       # –     zero-lift drag coefficient
        self.CD_alpha2: float = 8.0   # /rad² induced drag coefficient
        self.CY_beta: float = -18.5   # /rad  side-force due to sideslip
                                       #        (= -CL_alpha for axisymmetric)

        # ------------------------------------------------------------------
        # Moment coefficients (referenced to d_ref)
        # ------------------------------------------------------------------
        # Pitch axis (body y, positive nose-up)
        self.Cm_alpha: float = -3.0   # /rad  static pitch stability (negative = stable)
        self.Cm_q: float = -15.0      # /rad  pitch damping (per unit q_hat = q*d/(2V))
        self.Cm_delta: float = -1.2   # /rad  elevator (pitch fin) control effectiveness

        # Yaw axis (body z, positive nose-right)
        self.Cn_beta: float = 3.0     # /rad  static yaw stability ( positive = stable)
        self.Cn_r: float = -15.0      # /rad  yaw damping (per unit r_hat = r*d/(2V))
        self.Cn_delta_r: float = -1.2 # /rad  rudder (yaw fin) control effectiveness

        # Roll axis (body x, positive right-wing-down)
        self.Cl_p: float = -5.0       # /rad  roll damping (per unit p_hat = p*d/(2V))
        self.Cl_delta_a: float = -0.8 # /rad  aileron control effectiveness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_forces_moments(
        self,
        alpha: float,
        beta: float,
        V: float,
        q_bar: float,
        p: float,
        q: float,
        r: float,
        delta_e: float,
        delta_r: float = 0.0,
        delta_a: float = 0.0,
    ) -> tuple:
        """Compute aerodynamic forces and moments in the body frame.

        The force signs follow the body-axis convention where Fz is positive
        downward (same direction as body z).  A positive normal acceleration
        (pull-up) therefore requires Fz < 0.

        Args:
            alpha:   Angle of attack (rad).  Positive nose-up.
            beta:    Sideslip angle (rad).   Positive nose-right.
            V:       Total airspeed (m/s).   Must be > 0 for meaningful results.
            q_bar:   Dynamic pressure (Pa) = 0.5 * rho * V².
            p:       Roll rate (rad/s),  body x-axis.
            q:       Pitch rate (rad/s), body y-axis.
            r:       Yaw rate (rad/s),   body z-axis.
            delta_e: Elevator deflection (rad).  Positive trailing-edge-up.
            delta_r: Rudder deflection (rad).    Positive trailing-edge-right.
            delta_a: Aileron deflection (rad).   Positive right-aileron-up.

        Returns:
            forces  (np.ndarray): [Fx, Fy, Fz] in body frame (N).
            moments (np.ndarray): [L, M, N] about body axes (N·m).
                                   L – roll, M – pitch, N – yaw.
        """
        qS = q_bar * self.S_ref
        qSd = qS * self.d_ref

        # Normalised angular rates: hat = rate * d_ref / (2 * V)
        if V > 1e-6:
            norm = self.d_ref / (2.0 * V)
        else:
            norm = 0.0

        p_hat = p * norm
        q_hat = q * norm
        r_hat = r * norm

        # ------------------------------------------------------------------
        # Force coefficients
        # ------------------------------------------------------------------
        # Lift (opposes body z in stability-axis sense; we map to body CZ)
        CL = self.CL_alpha * alpha
        CD = self.CD_0 + self.CD_alpha2 * alpha ** 2

        # Body-axis force coefficients.
        # For small angles: CX ≈ -CD (axial, thrust axis opposes drag)
        #                   CZ ≈ -CL (normal, lift opposes body-down z)
        # This is consistent with the stability->body transform at small alpha.
        CX = -(CD * np.cos(alpha) - CL * np.sin(alpha))
        CZ = -(CL * np.cos(alpha) + CD * np.sin(alpha))
        CY = self.CY_beta * beta

        forces = qS * np.array([CX, CY, CZ])

        # ------------------------------------------------------------------
        # Moment coefficients
        # ------------------------------------------------------------------
        # Pitch moment (body y)
        Cm = (self.Cm_alpha * alpha
              + self.Cm_q * q_hat
              + self.Cm_delta * delta_e)

        # Yaw moment (body z)
        Cn = (self.Cn_beta * beta
              + self.Cn_r * r_hat
              + self.Cn_delta_r * delta_r)

        # Roll moment (body x)
        Cl = (self.Cl_p * p_hat
              + self.Cl_delta_a * delta_a)

        moments = qSd * np.array([Cl, Cm, Cn])

        return forces, moments

    def lift_coefficient(self, alpha: float) -> float:
        """Return the lift coefficient CL for a given angle of attack (rad)."""
        return self.CL_alpha * alpha

    def drag_coefficient(self, alpha: float) -> float:
        """Return the drag coefficient CD for a given angle of attack (rad)."""
        return self.CD_0 + self.CD_alpha2 * alpha ** 2

    def trim_alpha(self, normal_accel: float, q_bar: float, mass: float) -> float:
        """Estimate the trim angle of attack needed to sustain a normal acceleration.

        Uses the linearised relation: n = q_bar * S_ref * CL_alpha * alpha / (m * g).
        This is a convenience method for autopilot design; it does not account
        for gravity or pitch-rate terms.

        Args:
            normal_accel: Desired normal (pitch-plane) acceleration (m/s²).
            q_bar:        Dynamic pressure (Pa).
            mass:         Vehicle mass (kg).

        Returns:
            Trim angle of attack (rad).
        """
        denominator = q_bar * self.S_ref * self.CL_alpha / mass
        if abs(denominator) < 1e-10:
            return 0.0
        return normal_accel / denominator
