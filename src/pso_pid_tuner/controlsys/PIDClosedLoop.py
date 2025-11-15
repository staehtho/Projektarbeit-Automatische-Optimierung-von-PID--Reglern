from typing import Callable

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
        Tf = derivative_filter_ratio * system.t1

    Only one parameterization method should be used per initialization.
    Conversion between forms is handled automatically.

    PID Transfer Function:
        Gc(s) = Kp * (1 + 1/(Ti*s) + (Td*s)/(Tf*s + 1))
    """

    def __init__(self,
                 system: Plant,
                 *,
                 # Gain form
                 Kp: float = None,
                 Ki: float = None,
                 Kd: float = None,
                 # Time-constant form
                 Ti: float = None,
                 Td: float = None,
                 control_constraint: list[float] = None,
                 anti_windup_method: str = "clamping"
                 ) -> None:
        """
        Initialize a PID closed-loop controller.

        Args:
            system (Plant): The controlled system instance.
            Kp (float, optional): Proportional gain (gain form).
            Ki (float, optional): Integral gain (gain form).
            Kd (float, optional): Derivative gain (gain form).
            Ti (float, optional): Integral time constant (time form).
            Td (float, optional): Derivative time constant (time form).
                Tf = derivative_filter_ratio * system.t1 (default: 0.01).
            control_constraint (list[float], optional): [u_min, u_max] saturation limits. Defaults to [-5, 5].

        Raises:
            ValueError: If both or neither of the parameter sets (gain form, time form) are provided.
        """
        super().__init__(system)

        self._kp: float = 0
        self._ki: float = 0
        self._kd: float = 0

        self._ti: float = 0
        self._td: float = 0

        self.set_pid_param(Kp=Kp, Ki=Ki, Kd=Kd, Ti=Ti, Td=Td)

        # filter time constant
        self._tf: float = 0.01

        # Control output constraints
        self._control_constraint = control_constraint or [-5.0, 5.0]

        self._anti_windup_method: str = anti_windup_method

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

    @property
    def control_constraint(self) -> list[float]:
        return self._control_constraint

    @property
    def anti_windup_method(self) -> str:
        return self._anti_windup_method

    @anti_windup_method.setter
    def anti_windup_method(self, anti_windup_method) -> None:
        self._anti_windup_method = anti_windup_method

    def set_filter(self, Tf):
        self._tf = Tf

    def set_pid_param(self,
                      *,
                      # Gain form
                      Kp: float = None,
                      Ki: float = None,
                      Kd: float = None,
                      # Time-constant form
                      Ti: float = None,
                      Td: float = None):

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
        elif format_spec == "tf_num":
            return "[1 0];"
        elif format_spec == "tf_den":
            return f"[{self._tf} 1];"
        return super().__format__(format_spec)

    # -------------------- Time Domain --------------------

    def system_response(
            self,
            t0: float,
            t1: float,
            dt: float,
            r: Callable[[np.ndarray], np.ndarray] | None = None,
            d1: Callable[[np.ndarray], np.ndarray] | None = None,
            d2: Callable[[np.ndarray], np.ndarray] | None = None,
            x0: np.ndarray | None = None,
            y0: float = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the system response for a given reference signal.

        This method simulates the closed-loop response of a system with a PID controller
        over a specified time interval. It numerically integrates the system dynamics
        based on the system’s state-space representation (A, B, C, D) and applies the
        selected anti-windup strategy.

        Args:
            t0 (float): Start time of the simulation.
            t1 (float): End time of the simulation.
            dt (float): Time step for numerical integration.
            r (Callable[[np.ndarray], np.ndarray] | None, optional):
                Reference (setpoint) function as a function of time.
                Must accept a NumPy array of time values and return an array of the same shape.
                If None, a zero vector is used. Defaults to None.
            d1 (Callable[[np.ndarray], np.ndarray] | None, optional):
                Disturbance at the plant input (Z1) as a function of time.
                If None, zero disturbance is assumed. Defaults to None.
            d2 (Callable[[np.ndarray], np.ndarray] | None, optional):
                Disturbance at the measurement/output (Z2) as a function of time.
                If None, zero disturbance is assumed. Defaults to None.
            x0 (np.ndarray | None, optional): Initial state vector of the system. If None, a zero vector of appropriate
                dimension is used. Defaults to None.
            y0 (float, optional): Initial output value. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - **t_eval** (*np.ndarray*): Array of time points over the simulation interval.
                - **y** (*np.ndarray*): Plant output values corresponding to `t_eval`.

        Raises:
            NotImplementedError:
                If an unsupported anti-windup method is specified.

        Notes:
            - The system dynamics are obtained from the state-space matrices (A, B, C, D).
            - Supported anti-windup methods are:
                - `"conditional"`: Update the integrator only when output is within limits or reduces saturation.
                - `"clamping"`: Clamp the integrator term when the actuator saturates.
            - Internally calls the compiled function `pid_system_response()` for performance.
        """
        from .pso_system_optimization import pid_system_response

        t_eval = np.arange(t0, t1 + dt, dt)

        if r is None:
            r = lambda t: np.zeros_like(t)

        if d1 is None:
            d1 = lambda t: np.zeros_like(t)

        if d2 is None:
            d2 = lambda t: np.zeros_like(t)

        r_eval = r(t_eval)
        d1_eval = d1(t_eval)
        d2_eval = d2(t_eval)

        if x0 is None:
            x0 = np.zeros(self._system.get_plant_order())

        if self._anti_windup_method == "conditional":
            anti_windup = 0
        elif self._anti_windup_method == "clamping":
            anti_windup = 1
        else:
            raise NotImplementedError(f"Unsupported anti windup method: '{self._anti_windup_method}'")

        A, B, C, D = self._system.get_ABCD()

        A = np.ascontiguousarray(A, dtype=np.float64)
        # SISO → (n x 1)
        B = B.flatten()
        B = np.ascontiguousarray(B, dtype=np.float64)
        # SISO → (1 x n) wird aber in ein (n x 1) umgeschrieben (Performance)
        C = C.flatten()
        C = np.ascontiguousarray(C, dtype=np.float64)
        # SISO → D ist ein skalar
        D = float(D[0, 0])

        y = pid_system_response(Kp=self._kp, Ti=self._ti, Td=self._td,
                                Tf=self._tf, t_eval=t_eval, dt=dt,
                                r_eval=r_eval, d1_eval=d1_eval, d2_eval=d2_eval,
                                x=x0, control_constraint=np.array(self._control_constraint, dtype=np.float64),
                                anti_windup_method=anti_windup,
                                A=A, B=B, C=C, D=D)
        return t_eval, y
