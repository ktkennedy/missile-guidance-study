"""Abstract base class for missile autopilot implementations."""

from abc import ABC, abstractmethod


class AutopilotInterface(ABC):
    """Interface that all autopilot implementations must satisfy."""

    @abstractmethod
    def compute(self, a_cmd: float, a_measured: float, q_measured: float,
                dt: float) -> float:
        """Compute fin deflection command.

        Args:
            a_cmd:      Commanded acceleration (m/s²).
            a_measured: Measured/achieved acceleration (m/s²).
            q_measured: Measured pitch/yaw rate (rad/s).
            dt:         Integration timestep (s).

        Returns:
            Fin deflection command (rad).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all integrator and filter states."""
