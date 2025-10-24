import numpy as np
import time
from numba import njit, prange
from src.controlsys import ClosedLoop, PIDClosedLoop, pid_system_response, itae


class PsoFunc:
    """Wrapper for PSO optimization of a PID controller.

    This class prepares a PID controller for Particle Swarm Optimization (PSO)
    by providing a unified interface to calculate the ITAE (Integral of Time-weighted Absolute Error)
    cost for a given set of PID parameters. It handles system matrix extraction,
    PID parameter management (including anti-windup settings), and pre-compiles
    the Numba-accelerated simulation function for performance.

    Attributes:
        _controller (ClosedLoop): The PID controller to be optimized.
        t0 (float): Start time of the simulation.
        t1 (float): End time of the simulation.
        dt (float): Time step for the simulation.
        A (np.ndarray): State-space A matrix of the plant.
        B (np.ndarray): State-space B vector of the plant (flattened).
        C (np.ndarray): State-space C vector of the plant (flattened).
        D (float): State-space D scalar of the plant.
        system_order (int): Order of the system.
        controller_param (dict[str, str | float | np.ndarray]): Controller-specific parameters
            such as filter time constant, control constraints, and anti-windup method.
        swarm_size (int): Number of particles in the swarm.

    Example:
        >>> controller = PIDClosedLoop(plant, Kp=1.0, Ti=1.0, Td=0.1)
        >>> pso_func = PsoFunc(controller, t0=0.0, t1=10.0, dt=0.01, swarm_size=50)
        >>> X = np.array([[1.0, 1.0, 0.1]])
        >>> itae_val = pso_func(X)
    """

    def __init__(self, controller: ClosedLoop, t0: float, t1: float, dt: float, swarm_size: int):

        self._controller = controller

        self.t0 = t0
        self.t1 = t1
        self.dt = dt

        A, B, C, D = self._controller.plant.get_ABCD()
        self.A = A
        # SISO → (n x 1)
        self.B = B.flatten()
        # SISO → (1 x n) wird aber in ein (n x 1) umgeschrieben (Performance)
        self.C = C.flatten()
        # SISO → D ist ein skalar
        self.D = float(D[0, 0])

        self.system_order = self._controller.plant.get_system_order()

        self.controller_param: dict[str, str | float | np.ndarray]

        self.swarm_size = swarm_size

        if isinstance(self._controller, PIDClosedLoop):

            pid: PIDClosedLoop = self._controller

            if pid.anti_windup_method == "conditional":
                anti_windup = 0
            elif pid.anti_windup_method == "clamping":
                anti_windup = 1
            else:
                raise NotImplementedError(f"Unsupported anti windup method: '{pid.anti_windup_method}'")

            self.controller_param = {
                "Tf": pid.Tf,
                "control_constraint": np.array(pid.control_constraint, dtype=np.float64),
                "anti_windup": anti_windup
            }

        else:
            raise NotImplementedError(f"Unsupported controller type: '{type(controller)}'")

        # Vorab-Komplilierung von Numba für schnellere spätere Ausführung
        self.__call__(np.array([[1.0, 1.0, 0.1]], dtype=np.float64))

    def __call__(self, X) -> np.ndarray:

        # Sicherstellen, dass X 2D ist
        start = time.time()
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        itae_val = []
        if isinstance(self._controller, PIDClosedLoop):
            itae_val = _pid_pso_func(X, self.t0, self.t1, self.dt, self.A, self.B, self.C, self.D,
                                     self.system_order, self.controller_param["Tf"],
                                     self.controller_param["control_constraint"], self.controller_param["anti_windup"],
                                     self.swarm_size)

        end = time.time()
        print(f"{end - start:0.3f} sec")
        return itae_val


@njit(parallel=True)
def _pid_pso_func(X: np.ndarray, t0: float, t1: float, dt: float,
                  A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float,
                  system_order: int, Tf: float, control_constraint: np.ndarray, anti_windup_method: int,
                  swarm_size: int) -> np.ndarray:
    """Compute ITAE cost values for a PID controller with given parameters using parallel simulation.

    This function evaluates the PID-regulated system response for multiple sets of PID parameters
    and calculates the ITAE (Integral of Time-weighted Absolute Error) for each particle in the swarm.
    It is optimized with Numba for parallel execution.

    Args:
        X (np.ndarray): Array of PID parameters [Kp, Ti, Td] for each particle, shape (swarm_size, 3).
        t0 (float): Start time of the simulation.
        t1 (float): End time of the simulation.
        dt (float): Time step for the simulation.
        A (np.ndarray): State-space A matrix of the plant.
        B (np.ndarray): State-space B vector of the plant (flattened).
        C (np.ndarray): State-space C vector of the plant (flattened).
        D (float): State-space D scalar of the plant.
        system_order (int): Order of the plant system.
        Tf (float): Filter time constant for the derivative term.
        control_constraint (np.ndarray): Array [u_min, u_max] defining control signal limits.
        anti_windup_method (int): Anti-windup strategy (0 = conditional, 1 = clamping).
        swarm_size (int): Number of particles in the swarm.

    Returns:
        np.ndarray: ITAE cost values for each particle, shape (swarm_size,).
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

        y = pid_system_response(Kp, Ti, Td, Tf, t_eval, dt, r_eval,
                                x, control_constraint, anti_windup_method,
                                A, B, C, D)

        # ITAE-Kosten berechnen
        itae_val[i] = itae(t_eval, y, 1)

    return itae_val
