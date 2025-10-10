from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
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

    def __format__(self, format_spec: str) -> str:
        """
        Format the closed-loop system as a MATLAB-style transfer function string.

        This produces the symbolic closed-loop transfer function:
            G_cl(s) = (C(s) * G(s)) / (1 + C(s) * G(s))

        The controller and plant are formatted using their own __format__ methods:
            - Controller:  f"{self:controller}"
            - Plant:       f"{self._plant:plant}"

        Args:
            format_spec (str): Format type.
                - "cl": MATLAB-style closed-loop expression.

        Returns:
            str: MATLAB-formatted closed-loop transfer function string.

        Raises:
            NotImplementedError: If the format_spec is not supported.
        """
        # Formatstring bereinigen
        format_spec = format_spec.strip().lower()

        if format_spec == "cl":
            controller_str = format(self, "controller")
            plant_str = format(self._plant, "plant")

            num_str = f"{controller_str} * {plant_str}"
            den_str = f"1 + {controller_str} * {plant_str}"
            return f"({num_str}) / ({den_str})"
        else:
            raise NotImplementedError(f"Unsupported format specifier: '{format_spec}'")

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

    # ToDo: Integration Stoeruebertragungsfunktion

    def step_response(
            self,
            t0: float = 0,
            t1: float = 10,
            dt: float = 0.01,
            method: str = "RK23"
    ) -> tuple[np.ndarray, np.ndarray]:
        r = lambda t: 1
        return self.system_response(r, t0, t1, dt, method=method)

    def system_response(self, r: Callable[[float], float],
                        t0: float,
                        t1: float,
                        dt: float,
                        x0: np.ndarray | None = None,
                        y0: float = 0,
                        method: str = "RK23"
                        ) -> tuple[np.ndarray, np.ndarray]:
        if x0 is None:
            x0 = np.zeros(self._plant.get_system_order())

        t_eval = np.arange(t0, t1, dt)
        y_hist = []
        x = x0
        y = y0
        for t in t_eval:
            u = self.controller_time_step(t, y, set_point=r(t))
            if method == "RK4":
                x, y = self._plant.rk4_step(u, dt, x)
            else:
                x, y = self._plant.tf2ivp(u, t, t + dt, x)
            y_hist.append(y)

        return t_eval, np.array(y_hist)

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
