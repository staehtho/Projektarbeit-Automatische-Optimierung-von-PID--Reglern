from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
from .plant import Plant


class ClosedLoop(ABC):
    def __init__(self, plant: Plant):

        self._plant = plant

    def __format__(self, format_spec: str) -> str:
        """
        Format the closed-loop system as a MATLAB-style transfer function string.

        This produces the symbolic closed-loop transfer function:
            G_cl(s) = (C(s) * G(s)) / (1 + C(s) * G(s))

        The controller and system are formatted using their own __format__ methods:
            - Controller:  f"{self:controller}"
            - Plant:       f"{self._plant:system}"

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
            system_str = format(self._plant, "system")

            num_str = f"{controller_str} * {system_str}"
            den_str = f"1 + {controller_str} * {system_str}"
            return f"({num_str}) / ({den_str})"
        else:
            raise NotImplementedError(f"Unsupported format specifier: '{format_spec}'")

    @property
    def system(self) -> Plant:
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

    def closed_loop_Z1(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """Closed-loop transfer function for Z1."""
        C = self.controller(s)
        G = self._plant.system(s)
        return G / (1 + C * G)

    def closed_loop_Z2(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """Closed-loop transfer function for Z1."""
        C = self.controller(s)
        G = self._plant.system(s)
        return 1 / (1 + C * G)

    def step_response(
            self,
            t0: float = 0,
            t1: float = 10,
            dt: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the step response of the system.

        This method generates a unit step input signal and computes the corresponding
        system response over a specified time interval using the defined numerical
        integration method.

        Args:
            t0 (float, optional): Start time of the simulation. Defaults to 0.
            t1 (float, optional): End time of the simulation. Defaults to 10.
            dt (float, optional): Time step for the simulation. Defaults to 1e-4.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - **t_eval** (*np.ndarray*): Array of time points.
                - **y_hist** (*np.ndarray*): Array of system output values corresponding to `t_eval`.

        Notes:
            The step input `r(t)` is defined as a constant signal equal to 1 for all `t >= 0`.

            This method internally calls :meth:`system_response`, which performs the actual
            system simulation given the reference signal.
        """
        r = lambda t: np.ones_like(t)
        return self.system_response(t0, t1, dt, r=r)

    def z1_step_response(
            self,
            t0: float = 0,
            t1: float = 10,
            dt: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the step response of the system to a disturbance at the plant input (Z1).

        This method generates a unit step disturbance applied to the plant input (Z1)
        and computes the corresponding system response over a specified time interval.

        Args:
            t0 (float, optional): Start time of the simulation. Defaults to 0.
            t1 (float, optional): End time of the simulation. Defaults to 10.
            dt (float, optional): Time step for the simulation. Defaults to 1e-4.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **t_eval** (*np.ndarray*): Array of time points.
                - **y_hist** (*np.ndarray*): Array of system output values corresponding to `t_eval`.
        """
        l = lambda t: np.ones_like(t)
        return self.system_response(t0, t1, dt, l=l)

    def z2_step_response(
            self,
            t0: float = 0,
            t1: float = 10,
            dt: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the step response of the system to a disturbance at the plant input (Z2).

        This method generates a unit step disturbance applied to the plant input (Z2)
        and computes the corresponding system response over a specified time interval.

        Args:
            t0 (float, optional): Start time of the simulation. Defaults to 0.
            t1 (float, optional): End time of the simulation. Defaults to 10.
            dt (float, optional): Time step for the simulation. Defaults to 1e-4.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **t_eval** (*np.ndarray*): Array of time points.
                - **y_hist** (*np.ndarray*): Array of system output values corresponding to `t_eval`.
        """
        n = lambda t: np.ones_like(t)
        return self.system_response(t0, t1, dt, n=n)

    @abstractmethod
    def system_response(
            self,
            t0: float,
            t1: float,
            dt: float,
            r: Callable[[np.ndarray], np.ndarray] | None = None,
            l: Callable[[np.ndarray], np.ndarray] | None = None,
            n: Callable[[np.ndarray], np.ndarray] | None = None,
            x0: np.ndarray | None = None,
            y0: float = 0
    ) -> tuple[np.ndarray, np.ndarray]:

        pass
