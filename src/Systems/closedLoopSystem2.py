from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any


class ClosedLoopSystem2(ABC):
    """
    Abstract base class for closed-loop systems with a PID controller.
    Concrete subclasses must implement the _system_dynamics method.
    """

    def __init__(self,
                 Kp: float,
                 Ki: float | None = None,
                 Kd: float | None = None,
                 Ti: float | None = None,
                 Td: float | None = None,
                 pos_sat: float | None = None,
                 neg_sat: float | None = None,
                 K: float = 1.0,
                 setPoint: float = 1.0,
                 tau_d: float = 0.01,
                 control: bool = True) -> None:
        """Initializes the closed-loop system with PID parameters."""
        self.myKp: float = Kp
        self.myKi: float = Ki if Ki is not None else (0.0 if Ti == 0 else Kp / Ti)
        self.myKd: float = Kd if Kd is not None else (Kp * Td if Td is not None else 0.0)
        self.tau_d: float = tau_d
        self.myPosSat: float | None = pos_sat
        self.myNegSat: float | None = neg_sat
        self.myK = K
        self.myIntegralError: float = 0.0
        self.myLastError: list[float] = [0.0, 0.0]
        self.mySetPoint: float = setPoint
        self.prev_d: float = 0.0
        self.control: bool = control
        self.filtered_d: float = 0.0
        self.tau: float = 0.01
        self.order_i: int = 1  # Index of the output variable to control

    @abstractmethod
    def _system_dynamics(self, t: float, y: list[float]) -> list[float]:
        """
        Abstract method for system dynamics.

        Must be implemented by subclasses to return dy/dt for the ODE solver.

        Args:
            t (float): Current time.
            y (list[float]): Current state vector.

        Returns:
            list[float]: Derivatives of the state vector.
        """
        pass

    def _pid_controller(self, t: float, y: list[float]) -> float:
        """Computes the PID control output."""
        error: float = self.mySetPoint - y[self.order_i]
        dt: float = t - self.myLastError[1]

        if dt > 0:
            error_diff: float = error - self.myLastError[0]
            derivative: float = error_diff / dt
            alpha: float = dt / (self.tau + dt)
            self.filtered_d = alpha * derivative + (1 - alpha) * self.filtered_d
            self.myIntegralError += error * dt

        P: float = self.myKp * error
        I: float = self.myKi * self.myIntegralError
        if self.myNegSat is not None and self.myPosSat is not None:
            I = np.clip(I, self.myNegSat, self.myPosSat)
        D: float = self.myKd * self.filtered_d
        PID: float = P + I + D
        if self.myNegSat is not None and self.myPosSat is not None:
            PID = np.clip(PID, self.myNegSat, self.myPosSat)

        self.myLastError = [error, t]
        return PID

    def response(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Simulates the system response to a step input."""
        T1_0: float = 20.0
        T2_0: float = 20.0
        y0: list[float] = [T1_0, T2_0]
        t_span: tuple[float, float] = (0.0, 200.0)

        sol = solve_ivp(self._system_dynamics, t_span, y0,
                        max_step=0.1, method="RK23", vectorized=False)

        timet: np.ndarray = sol.t
        step_response: np.ndarray = sol.y[self.order_i]
        itae_krit: float = self.__itae(timet, step_response, self.mySetPoint)
        return itae_krit, timet, step_response

    @staticmethod
    def __itae(t: np.ndarray, y: np.ndarray, setPoint: float) -> float:
        """Calculates ITAE criterion."""
        value: float = 0.0
        t_alt = 0.0
        for r in range(len(t)):
            delta_t = t[r] - t_alt
            value += t[r] * abs((setPoint - y[r]) * delta_t)
            t_alt = t[r]
        return value

    @classmethod
    def systemResponseX(cls, X: list[float], **kwargs: Any) -> float:
        """
        Unified interface for optimizers (e.g., PSO).
        Expects a parameter vector X = [Kp, Ti, Td].

        Args:
            X (list of float): Parameter vector for the controller.
            **kwargs: Additional arguments for the system initialization.

        Returns:
            float: ITAE (Integral of Time-weighted Absolute Error) value of the system response.
        """
        system = cls(Kp=X[0], Ti=X[1], Td=X[2], **kwargs)
        itae, _, _ = system.response()
        return itae

    @classmethod
    def systemResponseX2(cls, X: list[float], **kwargs: Any) -> float:
        """
        Unified interface for optimizers (e.g., PSO).
        Expects a parameter vector X = [Kp, Ki, Kd].

        Args:
            X (list of float): Parameter vector for the controller.
            **kwargs: Additional arguments for the system initialization.

        Returns:
            float: ITAE (Integral of Time-weighted Absolute Error) value of the system response.
        """
        system = cls(Kp=X[0], Ki=X[1], Kd=X[2], **kwargs)
        itae, _, _ = system.response()
        return itae
