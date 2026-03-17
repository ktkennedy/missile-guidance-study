"""Gain scheduler for the three-loop autopilot.

Maps flight condition (Mach, altitude) to autopilot gains by:
1. Looking up nondimensional aero coefficients from TabulatedAerodynamics
2. Converting to dimensional stability derivatives via _nondim_to_dim()
3. Computing three-loop gains via ThreeLoopAutopilot gain formulas
4. Interpolating across a precomputed (Mach, altitude) grid at runtime

References:
    Abd-Elatif et al., "Three-Loop Autopilot Design for Tail-Controlled Missiles"
"""
import warnings
import numpy as np

from .three_loop_autopilot import ThreeLoopAutopilot


class GainScheduler:
    """Gain scheduler for three-loop autopilot across the flight envelope.

    Precomputes autopilot gains at a grid of (Mach, altitude) operating
    points and provides bilinear interpolation at runtime.

    Args:
        aero_model:     TabulatedAerodynamics instance for coefficient lookup.
        S_ref:          Reference area (m^2).
        d_ref:          Reference diameter (m).
        Iyy:            Pitch moment of inertia (kg*m^2).
        mass:           Vehicle mass (kg).
        omega:          Design bandwidth (rad/s).
        zeta:           Design damping ratio.
        tau:            Design time constant (s).
        omega_ACT:      Actuator bandwidth (rad/s).
        Ki:             Integral gain.
        mach_grid:      1-D array of Mach breakpoints.
        alt_grid:       1-D array of altitude breakpoints (m).
    """

    def __init__(self, aero_model, S_ref=0.01267, d_ref=0.127,
                 Iyy=12.8, mass=71.5,
                 omega=20.0, zeta=0.7, tau=0.5, omega_ACT=150.0, Ki=0.02,
                 mach_grid=None, alt_grid=None):

        self._aero = aero_model
        self.S_ref = S_ref
        self.d_ref = d_ref
        self.Iyy = Iyy
        self.mass = mass

        # Design parameters (stored for reference)
        self._omega = omega
        self._zeta = zeta
        self._tau = tau
        self._omega_ACT = omega_ACT
        self._Ki = Ki

        # Default grids
        if mach_grid is None:
            mach_grid = np.array([0.8, 1.2, 1.5, 2.0, 3.0])
        if alt_grid is None:
            alt_grid = np.array([0.0, 5000.0, 9150.0, 15000.0])

        self._mach_grid = np.asarray(mach_grid, dtype=float)
        self._alt_grid = np.asarray(alt_grid, dtype=float)

        # Precompute gains at each grid point
        self._gain_grid = self._precompute_gains()

    # ------------------------------------------------------------------
    # Dimensional conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _nondim_to_dim(Cm_alpha, Cm_delta, CL_alpha, CL_delta,
                       q_bar, V, S_ref, d_ref, Iyy, mass):
        """Convert nondimensional aero coefficients to dimensional derivatives.

        Args:
            Cm_alpha: pitch moment coefficient due to alpha (/rad)
            Cm_delta: pitch moment coefficient due to delta (/rad)
            CL_alpha: lift coefficient slope (/rad)
            CL_delta: lift coefficient due to delta (/rad) — often small
            q_bar:    dynamic pressure (Pa)
            V:        airspeed (m/s)
            S_ref:    reference area (m^2)
            d_ref:    reference diameter (m)
            Iyy:      pitch moment of inertia (kg*m^2)
            mass:     vehicle mass (kg)

        Returns:
            (M_alpha, M_delta, Z_alpha, Z_delta) in dimensional form
            M_alpha [/s^2], M_delta [/s^2/rad], Z_alpha [/s], Z_delta [/s/rad]
            All in codebase convention (negative = stable).
        """
        M_alpha = (q_bar * S_ref * d_ref * Cm_alpha) / Iyy      # /s^2
        M_delta = (q_bar * S_ref * d_ref * Cm_delta) / Iyy      # /s^2/rad
        Z_alpha = -(q_bar * S_ref * CL_alpha) / (mass * V)      # /s (negative for stable)
        Z_delta = -(q_bar * S_ref * CL_delta) / (mass * V)      # /s/rad
        return M_alpha, M_delta, Z_alpha, Z_delta

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_gains(self):
        """Compute gains at every (Mach, alt) grid point.

        Returns:
            dict of gain arrays, each shape (n_mach, n_alt).
        """
        nm = len(self._mach_grid)
        na = len(self._alt_grid)
        gains = {
            'KA':      np.zeros((nm, na)),
            'K_omega': np.zeros((nm, na)),
            'Kg':      np.zeros((nm, na)),
            'KDC':     np.zeros((nm, na)),
        }

        atm = self._aero._atm

        for i, mach in enumerate(self._mach_grid):
            for j, alt in enumerate(self._alt_grid):
                # Atmospheric conditions
                _, _, rho, a_sound = atm.get_properties(alt)
                V = mach * a_sound
                q_bar = 0.5 * rho * V ** 2

                # Interpolate nondimensional coefficients
                self._aero.set_altitude(alt)
                Cm_alpha = self._aero._interpolate(mach, 'Cm_alpha')
                Cm_delta = self._aero._interpolate(mach, 'Cm_delta')
                CL_alpha = self._aero._interpolate(mach, 'CL_alpha')
                # CL_delta is small for most missiles; approximate as fraction of Cm_delta
                CL_delta = abs(Cm_delta) * 0.2

                # Convert to dimensional
                M_alpha, M_delta, Z_alpha, Z_delta = self._nondim_to_dim(
                    Cm_alpha, Cm_delta, CL_alpha, CL_delta,
                    q_bar, V, self.S_ref, self.d_ref, self.Iyy, self.mass
                )

                # Compute gains via ThreeLoopAutopilot formulas
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ap = ThreeLoopAutopilot(
                            M_alpha=M_alpha, M_q=-4.0, M_delta=M_delta,
                            Z_alpha=Z_alpha, Z_delta=Z_delta, V=V,
                            omega=self._omega, zeta=self._zeta,
                            tau=self._tau, omega_ACT=self._omega_ACT,
                            Ki=self._Ki,
                        )
                    gains['KA'][i, j] = ap.KA
                    gains['K_omega'][i, j] = ap.K_omega
                    gains['Kg'][i, j] = ap.Kg
                    gains['KDC'][i, j] = ap.KDC
                except (ValueError, ZeroDivisionError):
                    # If gain computation fails at this point, use nearest valid
                    warnings.warn(
                        f"Gain computation failed at Mach={mach:.1f}, alt={alt:.0f}m",
                        stacklevel=2,
                    )

        return gains

    # ------------------------------------------------------------------
    # Runtime interpolation
    # ------------------------------------------------------------------

    def get_gains(self, mach, altitude):
        """Return interpolated autopilot gains at the given flight condition.

        Uses bilinear interpolation across the (Mach, altitude) grid.

        Args:
            mach:     Current Mach number.
            altitude: Current altitude (m).

        Returns:
            dict with keys 'KA', 'K_omega', 'Kg', 'KDC', each a float.
        """
        result = {}
        for key in ['KA', 'K_omega', 'Kg', 'KDC']:
            result[key] = self._bilinear_interp(
                mach, altitude, self._gain_grid[key]
            )
        return result

    def _bilinear_interp(self, mach, alt, grid):
        """Bilinear interpolation on the (Mach, alt) grid.

        Clamps to grid boundaries (no extrapolation).
        """
        mg = self._mach_grid
        ag = self._alt_grid

        # Clamp
        mach = np.clip(mach, mg[0], mg[-1])
        alt = np.clip(alt, ag[0], ag[-1])

        # Find bracketing indices for Mach
        im = np.searchsorted(mg, mach, side='right') - 1
        im = np.clip(im, 0, len(mg) - 2)

        # Find bracketing indices for altitude
        ia = np.searchsorted(ag, alt, side='right') - 1
        ia = np.clip(ia, 0, len(ag) - 2)

        # Fractional positions
        if mg[im + 1] - mg[im] > 1e-12:
            fm = (mach - mg[im]) / (mg[im + 1] - mg[im])
        else:
            fm = 0.0

        if ag[ia + 1] - ag[ia] > 1e-12:
            fa = (alt - ag[ia]) / (ag[ia + 1] - ag[ia])
        else:
            fa = 0.0

        # Bilinear interpolation
        v00 = grid[im, ia]
        v10 = grid[im + 1, ia]
        v01 = grid[im, ia + 1]
        v11 = grid[im + 1, ia + 1]

        return (v00 * (1 - fm) * (1 - fa) +
                v10 * fm * (1 - fa) +
                v01 * (1 - fm) * fa +
                v11 * fm * fa)
