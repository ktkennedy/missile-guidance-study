"""
US Standard Atmosphere 1976 model.

Reference: NOAA/NASA/USAF, U.S. Standard Atmosphere, 1976.

Covers sea level to 32 km altitude across three layers:
  - Troposphere   :  0 – 11 000 m  (lapse rate -6.5 K/km)
  - Tropopause    : 11 000 – 20 000 m  (isothermal at 216.65 K)
  - Lower Stratosphere: 20 000 – 32 000 m  (lapse rate +1.0 K/km)
"""

import numpy as np


class StandardAtmosphere1976:
    """US Standard Atmosphere 1976 model.

    Provides temperature, pressure, density and speed of sound as functions
    of geometric altitude.  Valid from sea level (h = 0) to 32 000 m.
    Altitudes outside this range are clamped to the nearest valid bound with
    a warning-free extrapolation at the boundary conditions.

    Constants are taken directly from the 1976 US Standard Atmosphere document.
    """

    # --------------------------------------------------------------------------
    # Physical constants
    # --------------------------------------------------------------------------
    R: float = 287.0528      # J/(kg·K)  specific gas constant for dry air
    g0: float = 9.80665      # m/s²      standard gravitational acceleration
    gamma_air: float = 1.4   # –         ratio of specific heats

    # --------------------------------------------------------------------------
    # Sea-level reference values (layer 0 base)
    # --------------------------------------------------------------------------
    T0: float = 288.15       # K
    P0: float = 101_325.0    # Pa
    rho0: float = 1.225      # kg/m³

    # --------------------------------------------------------------------------
    # Layer definitions
    # Each entry: (base_altitude_m, base_temperature_K, lapse_rate_K_per_m)
    # A lapse rate of 0 denotes an isothermal layer.
    # --------------------------------------------------------------------------
    _LAYERS = [
        (0.0,      288.15, -0.0065),   # Troposphere
        (11_000.0, 216.65,  0.0),      # Tropopause (isothermal)
        (20_000.0, 216.65,  0.001),    # Lower Stratosphere
        (32_000.0, 228.65,  None),     # Sentinel — marks upper boundary
    ]

    def __init__(self) -> None:
        # Pre-compute pressure at the base of each layer using the recurrence
        # relation from the standard atmosphere.
        self._base_pressures: list[float] = [self.P0]

        for i in range(len(self._LAYERS) - 1):
            h_b, T_b, lapse = self._LAYERS[i]
            h_top, _, _ = self._LAYERS[i + 1]
            P_b = self._base_pressures[i]

            if lapse == 0.0:
                # Isothermal layer: hydrostatic exponential
                P_top = P_b * np.exp(-self.g0 * (h_top - h_b) / (self.R * T_b))
            else:
                # Gradient layer: power-law relation
                T_top = T_b + lapse * (h_top - h_b)
                P_top = P_b * (T_top / T_b) ** (-self.g0 / (lapse * self.R))

            self._base_pressures.append(P_top)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def get_properties(self, altitude: float) -> tuple:
        """Return atmospheric properties at the given geometric altitude.

        Args:
            altitude: Geometric altitude above mean sea level (m).
                      Values below 0 are treated as 0; values above 32 000 m
                      are treated as 32 000 m.

        Returns:
            Tuple of:
                temperature      (K)
                pressure         (Pa)
                density          (kg/m³)
                speed_of_sound   (m/s)
        """
        # Clamp to the valid range of the model
        h = float(np.clip(altitude, 0.0, 32_000.0))

        # Identify which layer h falls in
        layer_idx = self._find_layer(h)
        h_b, T_b, lapse = self._LAYERS[layer_idx]
        P_b = self._base_pressures[layer_idx]

        dh = h - h_b

        if lapse == 0.0:
            # Isothermal layer
            T = T_b
            P = P_b * np.exp(-self.g0 * dh / (self.R * T_b))
        else:
            # Gradient layer
            T = T_b + lapse * dh
            P = P_b * (T / T_b) ** (-self.g0 / (lapse * self.R))

        rho = P / (self.R * T)
        a = np.sqrt(self.gamma_air * self.R * T)

        return T, P, rho, a

    def temperature(self, altitude: float) -> float:
        """Return temperature (K) at the given altitude (m)."""
        T, _, _, _ = self.get_properties(altitude)
        return T

    def pressure(self, altitude: float) -> float:
        """Return static pressure (Pa) at the given altitude (m)."""
        _, P, _, _ = self.get_properties(altitude)
        return P

    def density(self, altitude: float) -> float:
        """Return air density (kg/m³) at the given altitude (m)."""
        _, _, rho, _ = self.get_properties(altitude)
        return rho

    def speed_of_sound(self, altitude: float) -> float:
        """Return speed of sound (m/s) at the given altitude (m)."""
        _, _, _, a = self.get_properties(altitude)
        return a

    def dynamic_pressure(self, altitude: float, airspeed: float) -> float:
        """Return dynamic pressure q_bar = 0.5 * rho * V^2 (Pa).

        Args:
            altitude: Geometric altitude (m).
            airspeed: True airspeed (m/s).
        """
        rho = self.density(altitude)
        return 0.5 * rho * airspeed ** 2

    def mach(self, altitude: float, airspeed: float) -> float:
        """Return Mach number at the given altitude and airspeed.

        Args:
            altitude: Geometric altitude (m).
            airspeed: True airspeed (m/s).
        """
        a = self.speed_of_sound(altitude)
        return airspeed / max(a, 1e-6)

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _find_layer(self, h: float) -> int:
        """Return the index of the atmospheric layer containing altitude h."""
        # Walk from the top layer downward (h is already clamped)
        for i in range(len(self._LAYERS) - 2, -1, -1):
            if h >= self._LAYERS[i][0]:
                return i
        return 0  # Fallback – sea level
