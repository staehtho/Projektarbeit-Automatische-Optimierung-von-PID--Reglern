from .plant import Plant
from .closedLoop import ClosedLoop
import numpy as np


class PIDClosedLoop(ClosedLoop):
    """
    Represents a closed-loop control system with a PID controller in international form.

    The PID controller can be parameterized either in **gain form** or in **time-constant form**.

    Gain form:
        Kp, Ki, Kd

    Time-constant form:
        Kp, Ti, Td

    The derivative term is filtered using a first-order (PT1) filter with time constant:
        Tf = derivative_filter_ratio * plant.t1

    Only one parameterization method should be used per initialization.
    Conversion between forms is handled automatically.

    PID Transfer Function:
        Gc(s) = Kp * (1 + 1/(Ti*s) + (Td*s)/(Tf*s + 1))
    """

    def __init__(self,
                 plant: Plant,
                 *,
                 # Gain form
                 Kp: float = None,
                 Ki: float = None,
                 Kd: float = None,
                 # Time-constant form
                 Ti: float = None,
                 Td: float = None,
                 derivative_filter_ratio: float = 0.01,
                 control_constraint: list[float] = None
                 ) -> None:
        """
        Initialize a PID closed-loop controller.

        Args:
            plant (Plant): The controlled plant instance.
            Kp (float, optional): Proportional gain (gain form).
            Ki (float, optional): Integral gain (gain form).
            Kd (float, optional): Derivative gain (gain form).
            Ti (float, optional): Integral time constant (time form).
            Td (float, optional): Derivative time constant (time form).
            derivative_filter_ratio (float, optional): Ratio to compute derivative filter time constant.
                Tf = derivative_filter_ratio * plant.t1 (default: 0.01).
            control_constraint (list[float], optional): [u_min, u_max] saturation limits. Defaults to [-5, 5].

        Raises:
            ValueError: If both or neither of the parameter sets (gain form, time form) are provided.
        """
        super().__init__(plant)

        # --- Parameter Validation ---
        gain_form = all(v is not None for v in (Kp, Ki, Kd))
        time_form = all(v is not None for v in (Kp, Ti, Td))

        if gain_form and time_form:
            raise ValueError("Use either (Kp, Ki, Kd) or (Kp, Ti, Td), not both.")
        if not (gain_form or time_form):
            raise ValueError("You must provide either the gain form or the time-constant form.")

        # --- Assign Parameters and Convert if Needed ---
        self._kp = Kp

        if gain_form:
            # Gain → Time conversion
            self._ki = Ki
            self._kd = Kd
            self._ti = Kp / Ki
            self._td = Kd / Kp
        else:
            # Time → Gain conversion
            self._ti = Ti
            self._td = Td
            self._ki = Kp / Ti
            self._kd = Kp * Td

        # Derivative filter time constant
        self._tf = self._plant.t1 * derivative_filter_ratio

        # Control output constraints
        self._control_constraint = control_constraint or [-5.0, 5.0]

        # Internal states for time-domain simulation
        self._last_time: float | None = None
        self._last_error: float = 0.0
        self._integral: float = 0.0
        self._filtered_d: float = 0.0
        self._last_u: float = 0.0

    # -------------------- Properties --------------------

    @property
    def Kp(self) -> float:
        """Proportional gain."""
        return self._kp

    @property
    def Ki(self) -> float:
        """Integral gain."""
        return self._ki

    @property
    def Kd(self) -> float:
        """Derivative gain."""
        return self._kd

    @property
    def Ti(self) -> float:
        """Integral time constant."""
        return self._ti

    @property
    def Td(self) -> float:
        """Derivative time constant."""
        return self._td

    @property
    def Tf(self) -> float:
        """Derivative filter time constant."""
        return self._tf

    # -------------------- Frequency Domain --------------------

    def controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Compute the PID controller transfer function with derivative filter in the Laplace domain.

        Args:
            s (complex | np.ndarray): Laplace variable.

        Returns:
            complex | np.ndarray: Complex transfer function value.
        """
        P = 1
        I = 1 / (self._ti * s)
        D = (self._td * s) / (self._tf * s + 1)
        return self._kp * (P + I + D)

    def __format__(self, format_spec: str) -> str:
        """
        Format the PID-T1 controller as a MATLAB transfer function string.

        Example:
            >>> format(pid, "controller")
            'tf([Td*Ti, Ti, 1], [Ti*Tf, Ti, 0]) * Kp'

        Args:
            format_spec (str): Format specifier ('controller' for MATLAB-style).

        Returns:
            str: Formatted MATLAB transfer function string.
        """
        format_spec = format_spec.strip().lower()
        if format_spec == "controller":
            num = f"[{self._ti * self._td} {self._ti} 1]"
            den = f"[{self._ti * self._tf} {self._ti} 0]"
            return f"tf({num}, {den}) * {self._kp}"
        return super().__format__(format_spec)

    # -------------------- Time Domain --------------------

    def controller_time_step(self, t: float, y: float, set_point: float | None = None) -> float:
        """
        Compute PID control output in the time domain using the time-constant form.

        The method keeps internal state between calls and supports anti-windup and derivative filtering.

        Args:
            t (float): Current simulation time [s].
            y (float): Current measured process variable.
            set_point (float, optional): Desired reference value. Defaults to `self._set_point`.

        Returns:
            float: Control output u(t).
        """
        if set_point is None:
            set_point = self._set_point

        # Compute control error
        error = set_point - y

        # First call initialization
        if self._last_time is None:
            self._last_time = t
            self._last_error = error
            return self._last_u

        # Time difference
        dt = t - self._last_time
        if dt <= 0:
            return self._last_u

        # Proportional term
        P = self._kp * error

        # Integral term (Euler integration)
        self._integral += error * dt
        I = self._kp / self._ti * self._integral

        # Derivative term with PT1 filter
        derivative = (error - self._last_error) / dt
        alpha = dt / (self._tf + dt)
        self._filtered_d = alpha * derivative + (1 - alpha) * self._filtered_d
        D = self._kp * self._td * self._filtered_d

        # Control output with saturation
        u = P + I + D
        # u = float(np.clip(u, self._control_constraint[0], self._control_constraint[1]))
        # TODO: Anti-windup in Simulink und Python aktivieren
        # Anti-windup (Integrator clamping)
        # if ((u >= self._control_constraint[1] and error > 0) or
        #         (u <= self._control_constraint[0] and error < 0)):
        #     self._integral -= error * dt

        # Update states
        self._last_time = t
        self._last_error = error
        self._last_u = u

        return u

    def _reset_controller_time_step(self) -> None:
        # Internal states for time-domain simulation
        self._last_time: float | None = None
        self._last_error: float = 0.0
        self._integral: float = 0.0
        self._filtered_d: float = 0.0
        self._last_u: float = 0.0
        pass
