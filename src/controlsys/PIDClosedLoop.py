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
        # TODO: Achtung hier für die Verifikation von Büchi diese gleich wie in Simulink implementieren
        # self._tf = self._plant.t1 * derivative_filter_ratio
        self._tf = 0.01

        # Control output constraints
        self._control_constraint = control_constraint or [-5.0, 5.0]

        # Internal states for time-domain simulation
        self._e_prev: float = 0.0
        self._e_prev2: float = 0.0
        self._filtered_prev: float = 0.0
        self._integral: float = 0.0

        self._P_hist = []
        self._I_hist = []
        self._D_hist = []

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
    def P_hist(self) -> np.ndarray:
        return np.ndarray(self._P_hist)

    @property
    def I_hist(self) -> np.ndarray:
        return np.ndarray(self._I_hist)

    @property
    def D_hist(self) -> np.ndarray:
        return np.ndarray(self._D_hist)

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

    def controller_time_step(self,
                             t: float,
                             dt: float,
                             y: float,
                             set_point: float | None = None,
                             anti_windup_method: str = "clamping"
                             ) -> float:
        """
        Compute the controller output in discrete time (time-domain formulation).

        This method updates the internal states of a continuous-time PID (or similar)
        controller using the current measurement `y`, the desired set point, and
        the elapsed time step `dt`. It supports multiple anti-windup mechanisms to
        prevent integrator windup when actuator limits are reached.

        Args:
            t (float):
                Current simulation or control time in seconds.

            dt (float):
                Time step since the previous controller update in seconds.
                Used to correctly scale the integral and derivative terms.

            y (float):
                Current measured process variable (feedback signal).

            set_point (float, optional):
                Desired reference value for the process variable. If `None`, the
                last internally stored set point (`self._set_point`) is used.
                Defaults to `None`.

            anti_windup_method (str, optional):
                Method used to prevent integrator windup when actuator limits are hit.
                Must be one of:

                - `"clamping"`: Directly clamps the integral term to actuator limits.
                - `"conditional"`: Updates the integrator only if the output is within
                  actuator bounds or in a direction that reduces saturation.

                Defaults to `"clamping"`.

        Returns:
            float:
                Controller output signal `u(t)`, typically representing the actuator input.
                The value is saturated according to the internal control constraints
                (`self._control_constraint`).

        Raises:
            NotImplementedError:
                If `anti_windup_method` is not one of the supported options.

        Notes:
            - The controller keeps its internal state between calls.
              It should therefore be called sequentially in a real-time loop or simulation.
            - Proper choice of `dt` is critical for stability and accuracy.
            - Derivative action uses a first-order (PT1) filter with time constant `self._tf`.
            - Internal histories (`P_hist`, `I_hist`, `D_hist`, `U_temp_hist`) are updated
              at each step for logging or plotting purposes.
            - The proportional (`P`), integral (`I`), and derivative (`D`) components are
              combined as:

                  u(t) = P + I + D

              before output saturation and anti-windup correction.

        Example:
            >>> # Assume controller instance `pid` has been properly initialized
            >>> u = pid.controller_time_step(
            ...     t=0.1,
            ...     dt=0.01,
            ...     y=2.5,
            ...     set_point=3.0,
            ...     anti_windup_method="conditional"
            ... )
            >>> print(u)
            0.742
        """

        # ------------------------------
        # Setpoint and Error
        # ------------------------------
        r = self._set_point if set_point is None else set_point
        e = r - y

        # --- Proportional term ---
        P = self._kp * e

        # --- Derivative term (mit PT1-Filter) ---
        de = (e - self._e_prev) / dt
        alpha = dt / (self._tf + dt)
        d_filtered = (1 - alpha) * self._filtered_prev + alpha * de
        D = self._kp * self._td * d_filtered

        if anti_windup_method == "conditional":
            # Integral term
            I = self._kp * (1 / self._ti) * self._integral

            # --- Conditional Integration Logik ---
            # Integrator nur updaten, wenn keine Sättigung ODER Entlastungsrichtung
            u_temp = P + I + D

            u_min, u_max = self._control_constraint
            if (u_max > u_temp > u_min) or \
                    (u_temp >= u_max and e < 0) or \
                    (u_temp <= u_min and e > 0):
                self._integral += e * dt
                # Integral term
                I = self._kp * (1 / self._ti) * self._integral
        elif anti_windup_method == "clamping":
            # Integral term
            self._integral += e * dt
            I = self._kp * (1 / self._ti) * self._integral
            I = float(np.clip(I, *self._control_constraint))

        else:
            raise NotImplementedError(f"Unsupported anti windup method: '{anti_windup_method}'")

        # --- Gesamtausgang berechnen ---
        u_unsat = P + I + D

        # --- Stellgrößenbegrenzung am Reglerausgang ---
        u_sat = float(np.clip(u_unsat, *self._control_constraint))

        # Save filtered derivative for next step
        self._filtered_prev = d_filtered

        # Save histories for debugging / plotting
        self._P_hist.append(P)
        self._I_hist.append(I)
        self._D_hist.append(D)

        # Update previous error values
        self._e_prev2 = self._e_prev
        self._e_prev = e

        return u_sat

    def _reset_controller_time_step(self) -> None:
        """Reset PID internal states before simulation."""
        self._e_prev = 0.0
        self._e_prev2 = 0.0
        self._filtered_prev = 0.0
        self._integral = 0.0
        self._P_hist.clear()
        self._I_hist.clear()
        self._D_hist.clear()
