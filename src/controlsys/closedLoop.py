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
            method: str = "RK23",
            anti_windup: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            anti_windup (bool, optional): If True, activates integrator clamping in
                the controller to prevent windup when actuator saturation occurs.
                Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                - t_eval (np.ndarray): Array of time points.
                - y_hist (np.ndarray): Array of system output values corresponding to `t_eval`.
        """
        r = lambda t: 1
        return self.system_response(r, t0, t1, dt, method=method, anti_windup=anti_windup)

    def system_response(
            self,
            r: Callable[[float], float],
            t0: float,
            t1: float,
            dt: float,
            x0: np.ndarray | None = None,
            y0: float = 0,
            method: str = "RK23",
            anti_windup: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            anti_windup (bool, optional): If True, activates integrator clamping in
                the controller to prevent windup when actuator saturation occurs.
                Defaults to True.

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
        u_hist = []
        e_hist = []
        x = x0
        y = y0
        for t in t_eval:
            e_hist.append(r(t) - 0)
            u = self.controller_time_step(t, dt, y, set_point=r(t), anti_windup=anti_windup)
            if method == "RK4":
                x, y = self._plant.rk4_step(u, dt, x)
            else:
                x, y = self._plant.tf2ivp(u, t, t + dt, x)
            y_hist.append(y)
            u_hist.append(u)

        return t_eval, np.array(y_hist), np.array(u_hist), np.array(e_hist)

    @abstractmethod
    def controller_time_step(self,
                             t: float,
                             dt: float,
                             y: float,
                             set_point: float | None = None,
                             anti_windup: bool = True
                             ) -> float:
        """
        Compute control output in the time domain (time-constant form).
        Uses internal state memory between calls.

        This method implements a discrete-time controller (e.g., PID) that updates its
        internal states (such as integral and derivative terms) based on the current
        measurement `y`, desired set point `set_point`, and time step `dt`. The controller
        output `u(t)` can be clamped to prevent actuator saturation, and an optional
        anti-windup mechanism can limit integrator growth when clamping occurs.

        Args:
            t (float): Current simulation time [s].
            dt (float): Time step since the last controller update [s]. Used to scale
                the integral and derivative contributions correctly.
            y (float): Current measured process variable (feedback signal).
            set_point (float, optional): Desired reference value for the process variable.
                If None, the controller may use the last set point or a default value.
                Defaults to None.
            anti_windup (bool, optional): If True, activates integrator clamping in
                the controller to prevent windup when actuator saturation occurs.
                Defaults to True.

        Returns:
            float: Control output u(t), typically in the range supported by the actuator.
                The returned value may be limited to prevent saturation, and internal
                integrator states may be adjusted accordingly if anti-windup is active.

        Notes:
            - The controller maintains internal state between calls, so it must be called
              sequentially in simulation or real-time control.
            - Proper choice of `dt` is crucial for stable and accurate control behavior.
            - If `anti_windup` is enabled, the integrator term will be adjusted to
              prevent excessive overshoot caused by actuator limits.
        """
        pass

    @abstractmethod
    def _reset_controller_time_step(self) -> None:
        pass
