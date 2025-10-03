from .plant import Plant
import numpy as np


class PIDClosedLoop:
    """
    Represents a closed-loop control system with a PID controller.

    The closed-loop system connects a plant with a PID controller defined by
    proportional, integral, and derivative terms. The controller parameters
    can be specified in two alternative ways:

    1. Direct form:
       - kp : proportional gain
       - ki : integral gain
       - kd : derivative gain

    2. Time-constant form:
       - kp : proportional gain
       - tn : integral time constant (ki = kp / tn)
       - tv : derivative time constant (kd = kp * tv)

    Exactly one of the two forms must be provided.
    Mixing parameters from both forms (e.g., `ki` with `tn`) is not allowed.

    Additionally, the derivative part of the PID controller is usually filtered.
    The filter time constant `tf` is automatically set proportional to the
    plant’s dominant time constant `t1`:

        tf = derivative_filter_ratio * plant.t1

    By default, `derivative_filter_ratio = 0.01`.

    Parameters
    ----------
    plant : Plant
        The plant or process to be controlled.
    kp : float
        Proportional gain of the PID controller.
    ki : float, optional
        Integral gain (only if using direct form).
    kd : float, optional
        Derivative gain (only if using direct form).
    tn : float, optional
        Integral time constant (only if using time-constant form).
    tv : float, optional
        Derivative time constant (only if using time-constant form).
    derivative_filter_ratio : float, optional, default=0.01
        Ratio used to compute the derivative filter constant `tf` relative to
        the plant’s dominant time constant `t1`.

    Raises
    ------
    ValueError
        If neither or both sets of parameters are provided,
        or if parameters from both forms are mixed.

    Example
    -------
    Direct form:
        plant = Plant(num=[1], den=[1, 2, 1])
        loop = PIDClosedLoop(plant=plant, kp=1.0, ki=0.5, kd=0.1)

    Time-constant form:
        plant = Plant(num=[1], den=[1, 2, 1])
        loop = PIDClosedLoop(plant=plant, kp=1.0, tn=2.0, tv=0.1)
    """

    def __init__(self,
                 plant: Plant,
                 kp: float,
                 *,
                 ki: float | None = None,
                 kd: float | None = None,
                 tn: float | None = None,
                 tv: float | None = None,
                 derivative_filter_ratio: float = 0.01
                 ) -> None:

        self._plant: Plant = plant
        self._kp: float = kp

        # Prüfen auf gemischte Eingaben
        if (ki is not None or kd is not None) and (tn is not None or tv is not None):
            raise ValueError("Use either (ki, kd) or (tn, tv), not both.")

        # Prüfen auf unvollständige Eingaben
        if (ki is None or kd is None) and (tn is None or tv is None):
            raise ValueError("Either provide both (ki, kd) or both (tn, tv).")

        # Direkte Form
        if ki is not None and kd is not None:
            self._ki: float = ki
            self._kd: float = kd
            self._tn: float = kp / ki
            self._tv: float = kd / kp

        # Zeitkonstanten-Form
        elif tn is not None and tv is not None:
            self._tn: float = tn
            self._tv: float = tv
            self._ki: float = kp / tn
            self._kd: float = kp * tv

        self._tf = self._plant.t1 * derivative_filter_ratio

    # ******************************
    # Attributes
    # ******************************

    @property
    def plant(self) -> Plant:
        """The plant (system) being controlled."""
        return self._plant

    @property
    def kp(self) -> float:
        """Proportional gain of the PID controller."""
        return self._kp

    @property
    def ki(self) -> float:
        """Integral gain of the PID controller."""
        return self._ki

    @property
    def kd(self) -> float:
        """Derivative gain of the PID controller."""
        return self._kd

    def __format__(self, format_spec: str) -> str:
        format_spec = format_spec.replace(" ", "")
        if format_spec == "pid":
            p_str = "1"
            i_str = f"1 / ({self._tn} * s)"
            d_str = f"({self._tv} * s) / ({self._tf} * s + 1)"
            return f"{self._kp} * ({p_str} + {i_str} + {d_str})"

        elif format_spec == "mat":
            return f"({self: pid} * {self._plant: mat}) / (1 + {self: pid} * {self._plant: mat})"

        else:
            raise NotImplementedError

    # ******************************
    # Methods
    # ******************************

    def pid_controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Returns the PID controller transfer function with derivative filter (PID-T1).

        The PID controller is defined as:

            C(s) = Kp * ( 1 + 1/(Tn*s) + (Tv*s)/(Tf*s + 1) )

        where:
            - Kp : proportional gain
            - Tn : integral time constant
            - Tv : derivative time constant
            - Tf : derivative filter time constant (PT1 filter for D-term)

        Parameters
        ----------
        s : complex or np.ndarray
            The Laplace variable (s = σ + jω). Can be a scalar or array of complex numbers.

        Returns
        -------
        complex or np.ndarray
            Value of the PID controller transfer function at the given s.
            Returns an array if s is an array, or a scalar if s is a single value.

        Notes
        -----
        - The derivative term is filtered using a first-order low-pass (Tf) to reduce
          high-frequency noise.
        - This corresponds to a PID-T1 controller (PID with PT1-filtered derivative).
        - The method supports both scalar and vectorized evaluation of s.
        """
        P = 1
        I = 1 / (self._tn * s)
        D = (self._tv * s) / (self._tf * s + 1)
        return self._kp * (P + I + D)

    def closed_loop(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluate the closed-loop transfer function of the system.

        The closed-loop transfer function is defined as:

            T(s) = C(s) * P(s) / (1 + C(s) * P(s))

        where:
            - C(s): PID controller transfer function
            - P(s): plant (process) transfer function

        Parameters
        ----------
        s : complex or np.ndarray
            The Laplace variable (s = σ + jω). Can be a scalar or an array
            of complex frequencies.

        Returns
        -------
        complex or np.ndarray
            The closed-loop transfer function evaluated at s.
            Returns a scalar if s is a single value, or an array if s is an array.

        Notes
        -----
        - The closed-loop represents the transfer function from reference input
          to system output.
        - This is useful for stability analysis, Bode plots, and step-response simulations.
        """
        return (self.pid_controller(s) * self._plant.system(s)) / (1 + self.pid_controller(s) * self._plant.system(s))

