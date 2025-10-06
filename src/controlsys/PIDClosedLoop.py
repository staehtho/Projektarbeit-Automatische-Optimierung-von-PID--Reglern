from .plant import Plant
import numpy as np
import mpmath as mp


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
        """
        self._plant = plant
        self._kp = kp

        # Begrenzung des Stellwerts (z. B. 0–100 %)
        self._control_constraint = [0, 100]

        # Standard-Sollwert (Einheitssprung)
        self._set_point: float = 1.0

        # Zustände für Zeitbereichsberechnung
        self._last_time: float | None = None
        self._last_error: float = 0.0
        self._integral: float = 0.0
        self._filtered_d: float = 0.0
        self._last_u: float = 0.0

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

    # -------------------- Properties --------------------

    @property
    def plant(self) -> Plant:
        return self._plant

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

    def pid_controller(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """PID controller transfer function with derivative filter (Laplace domain)."""
        P = 1
        I = 1 / (self._tn * s)
        D = (self._tv * s) / (self._tf * s + 1)
        return self._kp * (P + I + D)

    def closed_loop(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """Closed-loop transfer function."""
        C = self.pid_controller(s)
        G = self._plant.system(s)
        return (C * G) / (1 + C * G)

    def response(self, t: np.ndarray) -> np.ndarray:
        """Closed-loop step response via inverse Laplace transform (Talbot)."""
        F = lambda s: self.closed_loop(s) * 1 / s
        y = np.array([mp.invertlaplace(F, float(tt), method='talbot') for tt in t], dtype=float)
        return y

    # -------------------- Time Domain --------------------

    def pid_time_step(self, t: float, y: float, set_point: float | None = None) -> float:
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
        u = P + I + D
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
