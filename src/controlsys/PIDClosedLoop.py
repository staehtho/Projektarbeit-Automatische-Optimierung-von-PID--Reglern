from .plant import Plant
from .closedLoop import ClosedLoop
import numpy as np


class PIDClosedLoop(ClosedLoop):
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
        """
        super().__init__(plant)

        self._kp = kp

        # Parameter-Validierung
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

        # Filterzeitkonstante für D-Anteil
        self._tf = self._plant.t1 * derivative_filter_ratio

    def __format__(self, format_spec: str) -> str:
        """
        Format the PID-T1 controller as a MATLAB transfer function string.

        This uses the normalized PID-T1 form:
            G(s) = (Tn*Tv*s^2 + Tn*s + 1) / (Tn*Tf*s^2 + Tn*s)

        Args:
            format_spec (str): Format type.
                - "mat": MATLAB-style string representation.

        Returns:
            str: PID-T1 controller formatted as MATLAB transfer function string.

        Raises:
            NotImplementedError: If the format_spec is not supported.

        Examples:
            >>> pid = PIDClosedLoop(Tn=1.0, Tv=0.2, Tf=0.05)
            >>> format(pid, "controller")
            'tf([0.2 1.0 1],[0.05 1.0 0])'
        """
        # Whitespace entfernen und auf Kleinbuchstaben prüfen
        format_spec = format_spec.strip().lower()

        if format_spec == "controller":
            num_str = f"[{self._tn * self._tv} {self._tn} 1]"
            dec_str = f"[{self._tn * self._tf} {self._tn} 0]"
            return f"tf({num_str}, {dec_str})"
        else:
            return super().__format__(format_spec)

    # -------------------- Properties --------------------

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def ki(self) -> float:
        return self._ki

    @property
    def kd(self) -> float:
        return self._kd

    @property
    def tn(self) -> float:
        return self._tn

    @property
    def tv(self) -> float:
        return self._tv

    # -------------------- Frequency Domain --------------------

    def controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """PID controller transfer function with derivative filter (Laplace domain)."""
        P = 1
        I = 1 / (self._tn * s)
        D = (self._tv * s) / (self._tf * s + 1)
        return self._kp * (P + I + D)

    # -------------------- Time Domain --------------------

    def controller_time_step(self, t: float, y: float, set_point: float | None = None) -> float:
        """
        Compute PID control output in the time domain (time-constant form).
        Uses internal state memory between calls.

        Args:
            t (float): Current simulation time [s]
            y (float): Current measured process variable
            set_point (float, optional): Desired reference value. Defaults to 1.0.

        Returns:
            float: Control output u(t)
        """
        if set_point is None:
            set_point = self._set_point

        # Fehler
        error = set_point - y

        # Erstaufruf: Zustände initialisieren
        if self._last_time is None:
            self._last_time = t
            self._last_error = error
            return self._last_u

        # Zeitdifferenz
        dt = t - self._last_time
        if dt <= 0:
            return self._last_u

        # P-Anteil
        P = self._kp * error

        # I-Anteil (Euler-Integration)
        self._integral += error * dt
        I = self._kp / self._tn * self._integral

        # D-Anteil mit PT1-Filter
        derivative = (error - self._last_error) / dt
        alpha = dt / (self._tf + dt)
        self._filtered_d = alpha * derivative + (1 - alpha) * self._filtered_d
        D = self._kp * self._tv * self._filtered_d

        # Gesamtausgang
        u: float = P + I + D
        u = float(np.clip(u, self._control_constraint[0], self._control_constraint[1]))

        # Anti-Windup (Integrator-Clamping)
        if (u >= self._control_constraint[1] and error > 0) or \
           (u <= self._control_constraint[0] and error < 0):
            self._integral -= error * dt

        # Zustände aktualisieren
        self._last_time = t
        self._last_error = error
        self._last_u = u

        return u
