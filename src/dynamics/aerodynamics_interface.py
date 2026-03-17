"""Abstract base class for missile aerodynamics models."""

from abc import ABC, abstractmethod
import numpy as np


class AerodynamicsInterface(ABC):
    """Interface that all aerodynamics models must satisfy."""

    @abstractmethod
    def get_forces_moments(self, alpha, beta, V, q_bar, p, q, r,
                           delta_e, delta_r, delta_a):
        """Return (forces [3], moments [3]) in body frame."""

    @abstractmethod
    def lift_coefficient(self, alpha: float) -> float:
        """Return lift coefficient at given angle of attack (rad)."""

    @abstractmethod
    def drag_coefficient(self, alpha: float) -> float:
        """Return drag coefficient at given angle of attack (rad)."""

    @abstractmethod
    def trim_alpha(self, normal_accel: float, q_bar: float, mass: float) -> float:
        """Return trim angle of attack for given normal acceleration."""
