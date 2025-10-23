from numba import njit, prange
import numpy as np

from src.controlsys.plant import Plant


class PsoFunc:
    """Wrapper für die PSO-Optimierung eines PID-Reglers."""

    def __init__(self, plant: Plant, t0: float, t1: float, dt: float, Tf: float,
                 control_constraint: list[float] | np.ndarray, anti_windup_method: str, swarm_size: int):
        """
        Args:
            t0, t1, dt (float): Simulationszeitparameter.
            A, B, C, D (np.ndarray): Systemmatrizen des Zustandsraummodells.
            system_order (int): Ordnung des Systems.
            Tf (float): Filterzeitkonstante für D-Anteil.
            control_constraint (list[float] | np.ndarray): Stellgrößenbeschränkungen [u_min, u_max].
            anti_windup_method (str): Anti-Windup-Methode.
        """
        self.t0 = t0
        self.t1 = t1
        self.dt = dt

        A, B, C, D = plant.get_ABCD()
        self.A = A
        # SISO → (n x 1)
        self.B = B.flatten()
        # SISO → (1 x n) wird aber in ein (n x 1) umgeschrieben (Performance)
        self.C = C.flatten()
        # SISO → D ist ein skalar
        self.D = float(D[0, 0])

        self.system_order = plant.get_system_order()
        self.Tf = Tf
        self.control_constraint = np.array(control_constraint, dtype=np.float64)

        self.swarm_size = swarm_size

        if anti_windup_method == "conditional":
            self.anti_windup = 0
        elif anti_windup_method == "clamping":
            self.anti_windup = 1
        else:
            raise NotImplementedError(f"Unsupported anti windup method: '{anti_windup_method}'")

        self.anti_windup_method = anti_windup_method

        # Vorab-Komplilierung von Numba für schnellere spätere Ausführung
        _pso_func_jit(
            np.array([[1.0, 1.0, 0.1]], dtype=np.float64),
            self.t0, self.t1, self.dt, self.A, self.B, self.C, self.D, self.system_order, self.Tf,
            self.control_constraint, self.anti_windup, 1
        )

    def __call__(self, X) -> np.ndarray:
        """Ermöglicht die Instanz als Funktionsobjekt.

        Args:
            X (list | np.ndarray): PID-Parameter [Kp, Ti, Td].

        Returns:
            float: ITAE-Kostenwert.
        """
        # Sicherstellen, dass X 2D ist
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        val = _pso_func_jit(
            X, self.t0, self.t1, self.dt, self.A, self.B, self.C, self.D,
            self.system_order, self.Tf, self.control_constraint,
            self.anti_windup, self.swarm_size
        )
        return val


@njit(parallel=True)
def _pso_func_jit(X: np.ndarray, t0: float, t1: float, dt: float,
                  A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float,
                  system_order: int, Tf: float, control_constraint: np.ndarray, anti_windup_method: int,
                  swarm_size: int) -> np.ndarray:
    """Berechnet die ITAE-Kosten für einen PID-Regler mit gegebenen Parametern.

    Args:
        X (np.ndarray): PID-Parameter [Kp, Ti, Td].
        t0 (float): Startzeit.
        t1 (float): Endzeit.
        dt (float): Zeitschritt für die Simulation.
        A, B, C, D (np.ndarray): Systemmatrizen des Zustandsraummodells.
        system_order (int): Ordnung des Systems.
        Tf (float): Filterzeitkonstante für D-Anteil.
        control_constraint (np.ndarray): [u_min, u_max] für die Stellgröße.
        anti_windup_method (int): Methode des Anti-Windup (0 oder 1).

    Returns:
        float: ITAE-Kostenwert für die PID-Parameter.
    """

    t_eval = np.arange(t0, t1 + dt, dt)  # Zeitvektor
    r_eval = np.ones_like(t_eval)  # Referenzsignal (Sprung)
    itae_val = np.zeros(swarm_size)

    for i in prange(swarm_size):
        # Simulation des PID-regulierten Systems
        Kp = float(X[i, 0])
        Ti = float(X[i, 1])
        Td = float(X[i, 2])

        x = np.zeros(system_order, dtype=np.float64)  # Anfangszustand

        y = pid_system_response_RK4(Kp, Ti, Td, Tf, t_eval, dt, r_eval,
                                    x, control_constraint, anti_windup_method,
                                    A, B, C, D)

        # ITAE-Kosten berechnen
        itae_val[i] = itae(t_eval, y, 1)

    return itae_val


@njit
def pid_system_response_RK4(Kp: float, Ti: float, Td: float, Tf: float,
                            t_eval: np.ndarray, dt: float, r_eval: np.ndarray,
                            x: np.ndarray, control_constraint: np.ndarray,
                            anti_windup_method: int, A: np.ndarray, B: np.ndarray,
                            C: np.ndarray, D: float) -> np.ndarray:
    """Simuliert ein System mit PID-Regler und Anti-Windup mittels RK4.

    Args:
        Kp (float): PID-Parameter.
        Ti (float): PID-Parameter.
        Td (float): PID-Parameter.
        Tf (float): Filterzeitkonstante für D-Anteil.
        t_eval (np.ndarray): Zeitpunkte der Simulation.
        dt (float): Zeitschritt.
        r_eval (np.ndarray): Referenzsignal.
        x (np.ndarray): Anfangszustand.
        control_constraint (np.ndarray): [u_min, u_max].
        anti_windup_method (int): Anti-Windup-Methode (0 oder 1).
        A (np.ndarray): Systemmatrix
        B (np.ndarray): Systemmatrix
        C (np.ndarray): Systemmatrix
        D (float): Es wird ein SISO simuliert -> D ist ein Skalar

    Returns:
        np.ndarray: Ausgangssignal y(t) des Systems.
    """
    e_prev = 0.0
    filtered_prev = 0.0
    integral = 0.0

    u_min: float = float(control_constraint[0])
    u_max: float = float(control_constraint[1])

    n_steps = len(t_eval)
    y_hist = np.zeros(n_steps)

    # erster Ausgang
    y = C @ x

    for i in range(n_steps):
        # Ausgang berechnen

        # Fehler
        e = r_eval[i] - y

        # PID-Anteile
        P = Kp * e
        de = (e - e_prev) / dt
        alpha = dt / (Tf + dt)
        d_filtered = (1 - alpha) * filtered_prev + alpha * de
        D_term = Kp * Td * d_filtered
        I_term = Kp / Ti * integral
        u_temp = P + I_term + D_term

        # Anti-Windup
        if anti_windup_method == 0:
            if (u_min < u_temp < u_max) or \
                    (u_temp >= u_max and e < 0.0) or \
                    (u_temp <= u_min and e > 0.0):
                integral += e * dt
                I_term = Kp / Ti * integral
            u = min(max(P + I_term + D_term, u_min), u_max)

        elif anti_windup_method == 1:
            integral += e * dt
            I_term = Kp / Ti * integral
            I_term = min(max(I_term, u_min), u_max)
            u = min(max(P + I_term + D_term, u_min), u_max)

        else:
            # Nicht implementierte Methode
            return y_hist

        # RK4 Integration
        temp = np.zeros_like(x)

        k1 = matvec_auto(A, x) + B * u

        temp[:] = x + 0.5 * dt * k1
        k2 = matvec_auto(A, temp) + B * u

        temp[:] = x + 0.5 * dt * k2
        k3 = matvec_auto(A, temp) + B * u

        temp[:] = x + dt * k3
        k4 = matvec_auto(A, temp) + B * u

        x += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ausgang mit dem neuen x berechnen
        y = C @ x
        y_hist[i] = y + D * u

        # PID-Zustände aktualisieren
        e_prev = e
        filtered_prev = d_filtered

    return y_hist


@njit
def matvec_auto(A, x):
    n = A.shape[0]

    y = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x[j]
        y[i] = acc
    return y


@njit
def itae(t: np.ndarray, y: np.ndarray, set_point: float) -> float:
    """Compute the Integral of Time-weighted Absolute Error (ITAE).

    The ITAE criterion is defined as:

        ITAE = ∫ t * |set_point - y(t)| dt

    It penalizes errors that persist over time, emphasizing fast
    settling and minimal steady-state error.

    Args:
        t (np.ndarray): Time vector [s].
        y (np.ndarray): System output corresponding to `t`.
        set_point (float): Desired reference value.

    Returns:
        float: The computed ITAE value.
    """
    # berechnet delta t, beginnend mit t[0] - t[0]
    val = 0
    for i in range(1, t.shape[0]):
        val += t[i] * np.abs(set_point - y[i]) * (t[i] - t[i - 1])

    return val
