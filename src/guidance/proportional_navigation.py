"""Proportional Navigation Guidance Laws.

Implements Pure PN (PPN), True PN (TPN), and Augmented PN (APN)
for 3D missile-target engagement scenarios.

References:
    Zarchan, P. "Tactical and Strategic Missile Guidance", 7th Ed., AIAA, 2019
    Siouris, G.M. "Missile Guidance and Control Systems", Springer, 2004
"""
import numpy as np


def compute_los_geometry(r_M, v_M, r_T, v_T):
    """Compute Line-of-Sight geometry between missile and target.

    The LOS vector points from the missile to the target.  All quantities
    are expressed in the inertial (e.g. NED) frame.

    Args:
        r_M: missile position [3] (m)
        v_M: missile velocity [3] (m/s)
        r_T: target position [3] (m)
        v_T: target velocity [3] (m/s)

    Returns:
        dict with keys:
            R_vec     : relative position vector, T - M  [3] (m)
            R         : range scalar (m, >= 0)
            R_hat     : LOS unit vector [3]
            R_dot_vec : relative velocity vector, v_T - v_M [3] (m/s)
            V_c       : closing velocity (m/s, positive when missile approaches target)
            Omega_LOS : LOS angular velocity vector [3] (rad/s)
            lam_az    : LOS azimuth angle  (rad, atan2 of y vs x component)
            lam_el    : LOS elevation angle (rad, positive = target above missile)
            lam_dot_az: LOS azimuth rate   (rad/s)
            lam_dot_el: LOS elevation rate  (rad/s)
            t_go      : estimated time-to-go (s)
    """
    r_M = np.asarray(r_M, dtype=float)
    v_M = np.asarray(v_M, dtype=float)
    r_T = np.asarray(r_T, dtype=float)
    v_T = np.asarray(v_T, dtype=float)

    # --- Relative kinematics ---
    R_vec = r_T - r_M          # target w.r.t. missile
    R_dot_vec = v_T - v_M      # relative velocity

    R = float(np.linalg.norm(R_vec))

    # Guard against zero (or near-zero) range
    R_SMALL = 1e-6
    if R < R_SMALL:
        # Return a safe, degenerate geometry; callers should detect t_go ~ 0
        return dict(
            R_vec=R_vec,
            R=R,
            R_hat=np.zeros(3),
            R_dot_vec=R_dot_vec,
            V_c=0.0,
            Omega_LOS=np.zeros(3),
            lam_az=0.0,
            lam_el=0.0,
            lam_dot_az=0.0,
            lam_dot_el=0.0,
            t_go=0.0,
        )

    R_hat = R_vec / R

    # Closing velocity: rate of decrease of range
    # V_c = -dR/dt = -(R_hat . R_dot_vec)
    V_c = float(-np.dot(R_hat, R_dot_vec))

    # LOS angular velocity: Omega = (R_vec x R_dot_vec) / R^2
    Omega_LOS = np.cross(R_vec, R_dot_vec) / (R * R)

    # --- LOS angles (spherical coordinates) ---
    # Convention: x = North (or downrange), y = East, z = Down
    #   lam_az = atan2(Ry, Rx)   -- azimuth in the horizontal plane
    #   lam_el = atan2(-Rz, rho) -- elevation above the horizontal plane
    #             (negative z-down => positive elevation when target is above)
    Rx, Ry, Rz = R_vec
    rho = float(np.sqrt(Rx**2 + Ry**2))   # horizontal range

    lam_az = float(np.arctan2(Ry, Rx))
    lam_el = float(np.arctan2(-Rz, max(rho, R_SMALL)))

    # --- LOS rates derived from differentiation of spherical angles ---
    # d(lam_az)/dt = (Rx * R_dot_y - Ry * R_dot_x) / rho^2
    # d(lam_el)/dt = (-R_dot_z * rho^2 + Rz * (Rx*R_dot_x + Ry*R_dot_y)) / (R^2 * rho)
    #
    # These are numerically equivalent to projections of Omega_LOS onto the
    # azimuth and elevation unit vectors but derived directly for clarity.

    Rdx, Rdy, Rdz = R_dot_vec

    if rho < R_SMALL:
        # Near-vertical LOS: azimuth rate undefined; extract signed elevation
        # rate from Omega_LOS horizontal components instead of unsigned norm.
        lam_dot_az = 0.0
        omega_h = float(np.sqrt(Omega_LOS[0]**2 + Omega_LOS[1]**2))
        if omega_h > 1e-12:
            sign = np.sign(Omega_LOS[1]) if abs(Omega_LOS[1]) > abs(Omega_LOS[0]) else np.sign(-Omega_LOS[0])
            lam_dot_el = omega_h * sign
        else:
            lam_dot_el = 0.0
    else:
        lam_dot_az = float((Rx * Rdy - Ry * Rdx) / (rho**2))
        lam_dot_el = float(
            (-Rdz * rho**2 + Rz * (Rx * Rdx + Ry * Rdy)) / (R**2 * rho)
        )

    # --- Time-to-go estimate ---
    # Simple estimate: t_go = R / V_c  (valid when closing)
    # Guard against non-closing (V_c <= 0) or very small V_c
    V_C_MIN = 1.0  # 1 m/s threshold; below this t_go is unreliable
    if V_c > V_C_MIN:
        t_go = float(R / V_c)
    else:
        # Fall back to range / total relative speed (always positive)
        V_rel = float(np.linalg.norm(R_dot_vec))
        t_go = float(R / max(V_rel, V_C_MIN))

    return dict(
        R_vec=R_vec,
        R=R,
        R_hat=R_hat,
        R_dot_vec=R_dot_vec,
        V_c=V_c,
        Omega_LOS=Omega_LOS,
        lam_az=lam_az,
        lam_el=lam_el,
        lam_dot_az=lam_dot_az,
        lam_dot_el=lam_dot_el,
        t_go=t_go,
    )


def compute_zero_effort_miss(R_vec, R_dot_vec, t_go, n_T=None):
    """Compute Zero-Effort Miss (ZEM) distance.

    The ZEM is the miss distance that would occur if no further control were
    applied.  For a maneuvering target the target acceleration contribution
    is included via the second term.

    ZEM = R_vec + R_dot_vec * t_go                        (non-maneuvering)
    ZEM = R_vec + R_dot_vec * t_go + 0.5 * n_T * t_go^2  (maneuvering)

    This is derived from integrating the relative equations of motion forward
    to the predicted intercept time assuming zero missile acceleration.

    Args:
        R_vec    : relative position vector, T - M [3] (m)
        R_dot_vec: relative velocity vector, v_T - v_M [3] (m/s)
        t_go     : time-to-go (s)
        n_T      : target acceleration [3] (m/s², optional)

    Returns:
        ZEM: zero-effort miss vector [3] (m)
    """
    R_vec = np.asarray(R_vec, dtype=float)
    R_dot_vec = np.asarray(R_dot_vec, dtype=float)
    t_go = float(t_go)

    if t_go < 0.0:
        t_go = 0.0

    ZEM = R_vec + R_dot_vec * t_go

    if n_T is not None:
        n_T = np.asarray(n_T, dtype=float)
        ZEM = ZEM + 0.5 * n_T * (t_go**2)

    return ZEM


class ProportionalNavigation:
    """Proportional Navigation guidance law family.

    Supports three variants:

    PPN (Pure PN)
        a_cmd = N * |v_M| * (Omega_LOS x v_M_hat)
        The acceleration is perpendicular to the missile velocity vector.
        Classical 2D result; generalised to 3D here.

    TPN (True PN)
        a_cmd = N * V_c * (Omega_LOS x R_hat)
        The acceleration is perpendicular to the LOS.
        Optimal for non-maneuvering targets (linear-quadratic sense).

    APN (Augmented PN)
        a_cmd = N * V_c * (Omega_LOS x R_hat) + (N/2) * n_T_perp
        Adds a feedforward term for the component of target acceleration
        perpendicular to the LOS.  Optimal for maneuvering targets.

    The acceleration command is saturated to `a_max` in magnitude.

    Args:
        N      : Navigation constant (dimensionless, default 4.0; recommended 3-5)
        variant: 'PPN', 'TPN', or 'APN' (default 'APN')
        a_max  : Maximum acceleration command magnitude (m/s², default 400 ~ 40 g)
    """

    VALID_VARIANTS = {'PPN', 'TPN', 'APN'}

    def __init__(self, N: float = 4.0, variant: str = 'APN', a_max: float = 400.0):
        if N < 0:
            raise ValueError(f"Navigation constant N must be non-negative, got {N}")
        variant = variant.upper()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {self.VALID_VARIANTS}"
            )
        if a_max <= 0:
            raise ValueError(f"a_max must be positive, got {a_max}")

        self.N = float(N)
        self.variant = variant
        self.a_max = float(a_max)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _saturate(a_vec: np.ndarray, a_max: float) -> np.ndarray:
        """Saturate acceleration vector magnitude to a_max."""
        mag = float(np.linalg.norm(a_vec))
        if mag > a_max:
            return a_vec * (a_max / mag)
        return a_vec

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        r_M,
        v_M,
        r_T,
        v_T,
        n_T_est=None,
    ) -> np.ndarray:
        """Compute guidance acceleration command in the inertial frame.

        Args:
            r_M    : missile position [3] (m)
            v_M    : missile velocity [3] (m/s)
            r_T    : target position  [3] (m)
            v_T    : target velocity  [3] (m/s)
            n_T_est: estimated target acceleration [3] (m/s²); used only for APN.
                     If None and variant is APN, the augmentation term is zero.

        Returns:
            a_cmd: acceleration command [3] (m/s²) in the inertial frame,
                   magnitude <= a_max.
        """
        r_M = np.asarray(r_M, dtype=float)
        v_M = np.asarray(v_M, dtype=float)
        r_T = np.asarray(r_T, dtype=float)
        v_T = np.asarray(v_T, dtype=float)

        geo = compute_los_geometry(r_M, v_M, r_T, v_T)

        R = geo['R']
        R_hat = geo['R_hat']
        Omega_LOS = geo['Omega_LOS']
        V_c = geo['V_c']

        # Degenerate case: intercept achieved (range effectively zero)
        if R < 1e-6:
            return np.zeros(3)

        # Clamp closing velocity to non-negative to prevent reversed guidance
        # commands during diverging geometry (e.g., post-intercept overshoot).
        V_c_cmd = max(V_c, 0.0)

        if self.variant == 'PPN':
            V_M = float(np.linalg.norm(v_M))
            if V_M < 1e-6:
                return np.zeros(3)
            v_M_hat = v_M / V_M
            # Acceleration perpendicular to missile velocity
            a_cmd = self.N * V_M * np.cross(Omega_LOS, v_M_hat)

        elif self.variant == 'TPN':
            # Acceleration perpendicular to LOS
            a_cmd = self.N * V_c_cmd * np.cross(Omega_LOS, R_hat)

        else:  # APN
            # Base TPN term
            a_cmd = self.N * V_c_cmd * np.cross(Omega_LOS, R_hat)

            # Augmentation: component of target acceleration perpendicular to LOS
            if n_T_est is not None:
                n_T_est = np.asarray(n_T_est, dtype=float)
                # Remove LOS-parallel component
                n_T_perp = n_T_est - np.dot(n_T_est, R_hat) * R_hat
                a_cmd = a_cmd + (self.N / 2.0) * n_T_perp

        return self._saturate(a_cmd, self.a_max)

    def compute_pitch_yaw(
        self,
        r_M,
        v_M,
        r_T,
        v_T,
        n_T_est=None,
    ):
        """Compute guidance command decomposed into pitch and yaw planes.

        The decomposition uses LOS angular rates directly, matching standard
        autopilot channel conventions:

        Pitch channel (vertical plane):
            TPN/APN:  a_pitch = N * V_c * lam_dot_el + (N/2) * nT_el   [APN]
            PPN:      a_pitch = N * |v_M| * lam_dot_el

        Yaw channel (horizontal plane, azimuth):
            TPN/APN:  a_yaw   = N * V_c * lam_dot_az * cos(lam_el) + (N/2) * nT_az [APN]
            PPN:      a_yaw   = N * |v_M| * lam_dot_az * cos(lam_el)

        The cos(lam_el) factor projects the azimuth rate onto the horizontal
        guidance plane (consistent with the spherical coordinate Jacobian).

        Sign convention:
            a_pitch > 0  =>  pull-up (increase altitude in NED: -z direction)
            a_yaw   > 0  =>  turn right (positive y / East direction)

        Args:
            r_M    : missile position [3] (m)
            v_M    : missile velocity [3] (m/s)
            r_T    : target position  [3] (m)
            v_T    : target velocity  [3] (m/s)
            n_T_est: estimated target acceleration [3] (m/s²); used only for APN.

        Returns:
            a_pitch: pitch-plane acceleration command (m/s²)
            a_yaw  : yaw-plane acceleration command (m/s²)
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

        cos_el = float(np.cos(lam_el))

        if self.variant == 'PPN':
            V_M = float(np.linalg.norm(v_M))
            effective_speed = V_M if V_M >= 1e-6 else 0.0
        else:
            # Clamp to non-negative to prevent reversed commands
            effective_speed = max(V_c, 0.0)

        # Base PN terms
        a_pitch = self.N * effective_speed * lam_dot_el
        a_yaw = self.N * effective_speed * lam_dot_az * cos_el

        # APN augmentation (target acceleration feedforward)
        if self.variant == 'APN' and n_T_est is not None:
            n_T_est = np.asarray(n_T_est, dtype=float)
            # Decompose target acceleration into pitch/yaw channels.
            # In NED: x = North, y = East, z = Down
            #   elevation acceleration ~ -n_T[2]  (pull-up = -Down)
            #   azimuth acceleration   ~  n_T[1]  (turn right = East)
            # More precisely: project n_T onto the LOS elevation and azimuth
            # unit vectors in spherical coordinates.
            #
            # e_el  = [-sin(lam_el)*cos(lam_az),
            #          -sin(lam_el)*sin(lam_az),
            #          -cos(lam_el)]   (points in +elevation direction)
            # e_az  = [-sin(lam_az),
            #           cos(lam_az),
            #           0]             (points in +azimuth direction)
            lam_az = geo['lam_az']
            sin_el = float(np.sin(lam_el))
            cos_az = float(np.cos(lam_az))
            sin_az = float(np.sin(lam_az))

            e_el = np.array([
                -sin_el * cos_az,
                -sin_el * sin_az,
                -cos_el,
            ])
            e_az = np.array([
                -sin_az,
                cos_az,
                0.0,
            ])

            nT_el = float(np.dot(n_T_est, e_el))
            nT_az = float(np.dot(n_T_est, e_az))

            a_pitch += (self.N / 2.0) * nT_el
            a_yaw += (self.N / 2.0) * nT_az

        # Saturate each channel independently then enforce combined limit
        # (consistent with a 2D saturation ellipse; here we use total magnitude)
        a_vec = np.array([a_pitch, a_yaw])
        mag = float(np.linalg.norm(a_vec))
        if mag > self.a_max:
            scale = self.a_max / mag
            a_pitch *= scale
            a_yaw *= scale

        return float(a_pitch), float(a_yaw)
