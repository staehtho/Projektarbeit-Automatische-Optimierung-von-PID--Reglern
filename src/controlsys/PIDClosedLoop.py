from .plant import Plant
import numpy as np


class PIDClosedLoop:
    """
    Represents a closed-loop control system with a PID controller.

    The closed-loop system connects a plant with a PID controller defined by
    proportional, integral, and derivative terms. Controller parameters can
    be specified in two alternative ways:

    1. Direct form: kp, ki, kd
    2. Time-constant form: kp, tn, tv

    Exactly one form must be provided. Mixing parameters from both forms
    is not allowed. The derivative part is filtered using a PT1 filter
    with time constant tf = derivative_filter_ratio * plant.t1.
    """

    def __init__(
        self,
        plant: Plant,
        kp: float,
        *,
        ki: float = None,
        kd: float = None,
        tn: float = None,
        tv: float = None,
        derivative_filter_ratio: float = 0.01
    ) -> None:
        """
        Initialize a PID closed-loop controller.

        Args:
            plant (Plant): The plant to be controlled.
            kp (float): Proportional gain.
            ki (float, optional): Integral gain (direct form).
            kd (float, optional): Derivative gain (direct form).
            tn (float, optional): Integral time constant (time-constant form).
            tv (float, optional): Derivative time constant (time-constant form).
            derivative_filter_ratio (float, optional): Ratio to compute derivative filter Tf. Default 0.01.

        Raises:
            ValueError: If neither or both parameter sets are provided, or if mixed.

        Examples:
            Direct form:
            >>> plant = Plant(num=[1], den=[1, 2, 1]
            >>> loop = PIDClosedLoop(plant=plant, kp=1.0, ki=0.5, kd=0.1)

            Time-constant form:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> loop = PIDClosedLoop(plant=plant, kp=1.0, tn=2.0, tv=0.1)
        """
        self._plant = plant
        self._kp = kp

        if (ki is not None or kd is not None) and (tn is not None or tv is not None):
            raise ValueError("Use either (ki, kd) or (tn, tv), not both.")

        if (ki is None or kd is None) and (tn is None or tv is None):
            raise ValueError("Either provide both (ki, kd) or both (tn, tv).")

        # Direct form
        if ki is not None and kd is not None:
            self._ki = ki
            self._kd = kd
            self._tn = kp / ki
            self._tv = kd / kp
        # Time-constant form
        else:
            self._tn = tn
            self._tv = tv
            self._ki = kp / tn
            self._kd = kp * tv

        self._tf = self._plant.t1 * derivative_filter_ratio

    # Attributes
    @property
    def plant(self) -> Plant:
        """Plant being controlled."""
        return self._plant

    @property
    def kp(self) -> float:
        """Proportional gain."""
        return self._kp

    @property
    def ki(self) -> float:
        """Integral gain."""
        return self._ki

    @property
    def kd(self) -> float:
        """Derivative gain."""
        return self._kd

    @property
    def tn(self) -> float:
        """Integral time constant."""
        return self._tn

    @property
    def tv(self) -> float:
        """Derivative time constant."""
        return self._tv

    # Methods
    def pid_controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluate the PID controller transfer function with derivative filter (PID-T1).

        Args:
            s (complex or np.ndarray): Laplace variable (s = σ + jω). Scalar or array.

        Returns:
            complex or np.ndarray: PID controller evaluated at s.

        Examples:
            >>> PIDClosedLoop.pid_controller(1j * np.array([0.1, 1, 10]))
        """
        P = 1
        I = 1 / (self._tn * s)
        D = (self._tv * s) / (self._tf * s + 1)
        return self._kp * (P + I + D)

    def closed_loop(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluate the closed-loop transfer function of the system.

        T(s) = C(s) * P(s) / (1 + C(s) * P(s))

        Args:
            s (complex or np.ndarray): Laplace variable (s = σ + jω). Scalar or array.

        Returns:
            complex or np.ndarray: Closed-loop transfer function evaluated at s.

        Examples:
            >>> PIDClosedLoop.closed_loop(1j * np.array([0.1, 1, 10]))
        """
        C = self.pid_controller(s)
        P = self._plant.system(s)
        return (C * P) / (1 + C * P)
