from abc import ABC, abstractmethod
import numpy as np

from .plant import Plant


class ClosedLoop(ABC):
    def __init__(self, plant: Plant):

        self._plant = plant

        # Begrenzung des Stellwerts (z. B. 0–100 %)
        self._control_constraint = [0, 100]

        # Standard-Sollwert (Einheitssprung)
        self._set_point: float = 1.0

        # Zustände für Zeitbereichsberechnung
        self._last_time: float | None = None
        self._last_error: float = 0.0
        self._integral: float = 0.0
        self._filtered_d: float = 0.0
        self._last_u: float = 0.0

    # ToDo: __format__ für MATLAB
    @property
    def plant(self) -> Plant:
        return self._plant

    @abstractmethod
    def controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """Controller transfer function with derivative filter (Laplace domain)."""
        pass

    def closed_loop(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """Closed-loop transfer function."""
        C = self.controller(s)
        G = self._plant.system(s)
        return (C * G) / (1 + C * G)

    def response(self, t: np.ndarray) -> np.ndarray:
        """Closed-loop step response via inverse Laplace transform (Talbot)."""
        # ToDo: step response
        pass

    @abstractmethod
    def controller_time_step(self, t: float, y: float, set_point: float | None = None) -> float:
        """
        Compute control output in the time domain (time-constant form).
        Uses internal state memory between calls.

        Args:
            t (float): Current simulation time [s]
            y (float): Current measured process variable
            set_point (float, optional): Desired reference value. Defaults to 1.0.

        Returns:
            float: Control output u(t)
        """
        pass
