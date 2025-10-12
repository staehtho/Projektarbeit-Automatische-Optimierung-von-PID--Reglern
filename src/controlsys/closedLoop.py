from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
from .plant import Plant


class ClosedLoop(ABC):
    def __init__(self, plant: Plant):

        self._plant = plant

        # Standard-Sollwert (Einheitssprung)
        self._set_point: float = 1.0

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
        """
        Compute the controller transfer function with derivative filter in the Laplace domain.

        Args:
            s (complex | np.ndarray): Laplace variable.

        Returns:
            complex | np.ndarray: Complex transfer function value.
        """
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
        """
        Compute the step response of the system.

        This method generates a step input signal and computes the system response
        over the specified time interval using the specified numerical integration method.

        Args:
            t0 (float): Start time of the simulation. Defaults to 0.
            t1 (float): End time of the simulation. Defaults to 10.
            dt (float): Time step for the simulation. Defaults to 0.01.
            method (str): Numerical integration method to use. Options are "RK23" or "RK4".
                Defaults to "RK23".

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                - t_eval (np.ndarray): Array of time points.
                - y_hist (np.ndarray): Array of system output values corresponding to `t_eval`.
        """
        r = lambda t: 1
        return self.system_response(r, t0, t1, dt, method=method)

    def system_response(
            self,
            r: Callable[[float], float],
            t0: float,
            t1: float,
            dt: float,
            x0: np.ndarray | None = None,
            y0: float = 0,
            method: str = "RK23"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the system response to a given reference input signal.

        This method simulates the system over the specified time interval using the
        provided reference function `r(t)` and numerical integration method.

        Args:
            r (Callable[[float], float]): Reference input function of time.
            t0 (float): Start time of the simulation.
            t1 (float): End time of the simulation.
            dt (float): Time step for the simulation.
            x0 (np.ndarray | None, optional): Initial state of the system. If None, defaults
                to zero vector of system order. Defaults to None.
            y0 (float, optional): Initial output of the system. Defaults to 0.
            method (str, optional): Numerical integration method. Options are "RK23" or "RK4".
                Defaults to "RK23".

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - t_eval (np.ndarray): Array of time points.
                - y_hist (np.ndarray): Array of system output values corresponding to `t_eval`.
        """
        if x0 is None:
            x0 = np.zeros(self._plant.get_system_order())

        self._reset_controller_time_step()

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

    @abstractmethod
    def _reset_controller_time_step(self) -> None:
        pass
