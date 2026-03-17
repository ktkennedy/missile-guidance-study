"""LOS-Relative Engagement Dynamics for MPC Guidance.

Models the missile-target engagement in Line-of-Sight relative coordinates
using CasADi symbolic expressions for use in MPC optimization.

State vector (6):
    [R, V_c, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]

Control input (2):
    [a_pitch_cmd, a_yaw_cmd]

Parameters (4):
    [n_T_az, n_T_el, tau_ap, a_T_radial]

References:
    Zarchan, P. "Tactical and Strategic Missile Guidance", 7th Ed., AIAA, 2019
    Chapter 8: LOS-relative engagement dynamics
"""
import casadi as ca
import numpy as np


class LOSRelativeDynamics:
    """LOS-relative engagement dynamics with CasADi symbolic expressions.

    State vector (6):
        x[0] = R          : missile-target range (m)
        x[1] = V_c        : closing velocity (m/s), positive when approaching
        x[2] = lam_dot_az : LOS azimuth rate (rad/s)
        x[3] = lam_dot_el : LOS elevation rate (rad/s)
        x[4] = a_pitch_ach: achieved pitch acceleration (m/s^2)
        x[5] = a_yaw_ach  : achieved yaw acceleration (m/s^2)

    Control input (2):
        u[0] = a_pitch_cmd: commanded pitch acceleration (m/s^2)
        u[1] = a_yaw_cmd  : commanded yaw acceleration (m/s^2)

    Parameters (4):
        p[0] = n_T_az    : target lateral acceleration estimate, azimuth (m/s^2)
        p[1] = n_T_el    : target lateral acceleration estimate, elevation (m/s^2)
        p[2] = tau_ap    : autopilot time constant (s, default 0.05)
        p[3] = a_T_radial: target radial acceleration along LOS (m/s^2)

    Args:
        dt: integration time step (s, default 0.05)
        N : MPC prediction horizon steps (default 20)
    """

    def __init__(self, dt=0.05, N=20):
        self.dt = dt
        self.N = N
        self.n_x = 6
        self.n_u = 2
        self.n_p = 4  # [n_T_az, n_T_el, tau_ap, a_T_radial]

        self.a_max = 400.0   # 40g acceleration limit (m/s^2)
        self.R_min = 0.1     # singularity guard (m)
        self.tau_ap_default = 0.05  # default autopilot time constant (s)

        # CasADi symbolic variables
        self.sym_x = ca.SX.sym('x', self.n_x)
        self.sym_u = ca.SX.sym('u', self.n_u)
        self.sym_p = ca.SX.sym('p', self.n_p)

        # Build pre-compiled CasADi Function objects
        self._build_functions()

    def _build_functions(self):
        """Pre-build ca.Function objects for continuous/discrete dynamics and Jacobians."""
        xdot = self.derivatives(self.sym_x, self.sym_u, self.sym_p)
        x_next = self.f_d_rk4(self.sym_x, self.sym_u, self.sym_p)

        self.f_cont = ca.Function('f_los_cont',
            [self.sym_x, self.sym_u, self.sym_p], [xdot],
            ['x', 'u', 'p'], ['xdot'])
        self.f_disc = ca.Function('f_los_disc',
            [self.sym_x, self.sym_u, self.sym_p], [x_next],
            ['x', 'u', 'p'], ['x_next'])

        # Jacobians of continuous dynamics
        self.jac_x = ca.Function('jac_x',
            [self.sym_x, self.sym_u, self.sym_p],
            [ca.jacobian(xdot, self.sym_x)])
        self.jac_u = ca.Function('jac_u',
            [self.sym_x, self.sym_u, self.sym_p],
            [ca.jacobian(xdot, self.sym_u)])

    def derivatives(self, x, u, p):
        """LOS-relative engagement dynamics (Zarchan Ch. 8).

        Args:
            x: state vector ca.SX(6,)
            u: control input ca.SX(2,)
            p: parameter vector ca.SX(4,)

        Returns:
            xdot: state derivative ca.SX(6,)
        """
        R = x[0]
        Vc = x[1]
        lam_dot_az = x[2]
        lam_dot_el = x[3]
        a_p_ach = x[4]
        a_y_ach = x[5]

        a_p_cmd = u[0]
        a_y_cmd = u[1]

        n_T_az = p[0]
        n_T_el = p[1]
        tau_ap = p[2]
        a_T_radial = p[3]

        # Guard against singularity at R=0
        R_safe = ca.fmax(R, self.R_min)

        # Range rate: dR/dt = -V_c
        dR = -Vc

        # Closing velocity rate: centripetal term reduces V_c (Zarchan Ch. 8 sign convention)
        lam_dot_sq = lam_dot_az**2 + lam_dot_el**2
        dVc = -lam_dot_sq * R_safe - a_T_radial

        # LOS azimuth rate dynamics (Zarchan Ch.8 transverse equation)
        # R·λ̈ + 2·Ṙ·λ̇ = aT - aM  =>  λ̈ = (2·Vc·λ̇ + aT - aM) / R
        d_lam_dot_az = (2.0 * Vc * lam_dot_az + n_T_az - a_y_ach) / R_safe

        # LOS elevation rate dynamics
        d_lam_dot_el = (2.0 * Vc * lam_dot_el + n_T_el - a_p_ach) / R_safe

        # First-order autopilot lag: achieved tracks commanded
        da_p_ach = (a_p_cmd - a_p_ach) / tau_ap
        da_y_ach = (a_y_cmd - a_y_ach) / tau_ap

        return ca.vertcat(dR, dVc, d_lam_dot_az, d_lam_dot_el, da_p_ach, da_y_ach)

    def f_d_rk4(self, x, u, p):
        """Discrete dynamics via 4th-order Runge-Kutta integration.

        Args:
            x: state vector ca.SX(6,)
            u: control input ca.SX(2,)
            p: parameter vector ca.SX(4,)

        Returns:
            x_next: next state ca.SX(6,)
        """
        h = self.dt
        k1 = self.derivatives(x, u, p)
        k2 = self.derivatives(x + 0.5 * h * k1, u, p)
        k3 = self.derivatives(x + 0.5 * h * k2, u, p)
        k4 = self.derivatives(x + h * k3, u, p)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def ned_to_los_state(self, r_M, v_M, r_T, v_T, a_pitch_ach=0.0, a_yaw_ach=0.0):
        """Convert NED states to LOS-relative state vector (NumPy).

        Uses the same geometry formulas as compute_los_geometry() from
        proportional_navigation.py so that outputs are numerically consistent.

        Args:
            r_M        : missile position [3] (m, NED)
            v_M        : missile velocity [3] (m/s, NED)
            r_T        : target position  [3] (m, NED)
            v_T        : target velocity  [3] (m/s, NED)
            a_pitch_ach: initial achieved pitch acceleration (m/s^2, default 0)
            a_yaw_ach  : initial achieved yaw acceleration (m/s^2, default 0)

        Returns:
            np.array([R, V_c, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach])
        """
        r_M = np.asarray(r_M, dtype=float)
        v_M = np.asarray(v_M, dtype=float)
        r_T = np.asarray(r_T, dtype=float)
        v_T = np.asarray(v_T, dtype=float)

        R_vec = r_T - r_M
        R_dot_vec = v_T - v_M
        R = float(np.linalg.norm(R_vec))

        if R < 1e-6:
            return np.array([R, 0.0, 0.0, 0.0, a_pitch_ach, a_yaw_ach])

        R_hat = R_vec / R
        V_c = float(-np.dot(R_hat, R_dot_vec))

        # LOS angles in spherical coordinates (NED: x=North, y=East, z=Down)
        Rx, Ry, Rz = R_vec
        rho_h = float(np.sqrt(Rx**2 + Ry**2))  # horizontal range

        # LOS rates derived from spherical angle differentiation
        # (identical formulas to compute_los_geometry in proportional_navigation.py)
        Rdx, Rdy, Rdz = R_dot_vec
        R_SMALL = 1e-6

        Omega_LOS = np.cross(R_vec, R_dot_vec) / (R * R)

        if rho_h < R_SMALL:
            # Near-vertical LOS: azimuth rate undefined; use Omega_LOS components
            lam_dot_az = 0.0
            omega_h = float(np.sqrt(Omega_LOS[0]**2 + Omega_LOS[1]**2))
            if omega_h > 1e-12:
                sign = (np.sign(Omega_LOS[1])
                        if abs(Omega_LOS[1]) > abs(Omega_LOS[0])
                        else np.sign(-Omega_LOS[0]))
                lam_dot_el = omega_h * sign
            else:
                lam_dot_el = 0.0
        else:
            lam_dot_az = float((Rx * Rdy - Ry * Rdx) / (rho_h**2))
            lam_dot_el = float(
                (-Rdz * rho_h**2 + Rz * (Rx * Rdx + Ry * Rdy)) / (R**2 * rho_h)
            )

        return np.array([R, V_c, lam_dot_az, lam_dot_el,
                         float(a_pitch_ach), float(a_yaw_ach)])

    def los_to_zem(self, x, n_T_az=0.0, n_T_el=0.0):
        """Compute Zero-Effort Miss from LOS state (NumPy).

        Approximates ZEM using linear propagation of LOS rates:
            ZEM_az ~ R * lam_dot_az * t_go + 0.5 * n_T_az * t_go^2
            ZEM_el ~ R * lam_dot_el * t_go + 0.5 * n_T_el * t_go^2
        where t_go = R / V_c

        Args:
            x     : LOS state vector np.array(6,)
            n_T_az: target lateral acceleration estimate, azimuth (m/s^2)
            n_T_el: target lateral acceleration estimate, elevation (m/s^2)

        Returns:
            (ZEM_az, ZEM_el, t_go) as floats
        """
        R = x[0]
        Vc = x[1]
        lam_dot_az = x[2]
        lam_dot_el = x[3]

        # Guard against non-closing or very small closing velocity
        Vc_safe = Vc if Vc > 1.0 else 1.0
        t_go = R / Vc_safe

        ZEM_az = R * lam_dot_az * t_go + 0.5 * n_T_az * t_go**2
        ZEM_el = R * lam_dot_el * t_go + 0.5 * n_T_el * t_go**2

        return ZEM_az, ZEM_el, t_go

    def los_to_zem_casadi(self, x, p):
        """CasADi symbolic ZEM computation for use in MPC cost functions.

        Args:
            x: LOS state ca.SX(6,)
            p: parameter vector ca.SX(4,)  -- uses p[0]=n_T_az, p[1]=n_T_el

        Returns:
            ca.SX(2,) = [ZEM_az, ZEM_el]
        """
        R = x[0]
        Vc = ca.fmax(x[1], 1.0)  # guard against non-closing
        lam_dot_az = x[2]
        lam_dot_el = x[3]
        n_T_az = p[0]
        n_T_el = p[1]

        t_go = R / Vc
        ZEM_az = R * lam_dot_az * t_go + 0.5 * n_T_az * t_go**2
        ZEM_el = R * lam_dot_el * t_go + 0.5 * n_T_el * t_go**2

        return ca.vertcat(ZEM_az, ZEM_el)
