import numpy as np
import time
from numba import njit, prange, types, float64, int64
from .closedLoop import ClosedLoop
from .PIDClosedLoop import PIDClosedLoop
from typing import Callable


class PsoFunc:
    """
    Wrapper class for Particle Swarm Optimization (PSO) of a PID controller.

    This class prepares a PID controller for optimization using PSO by providing
    a unified interface to compute the ITAE (Integral of Time-weighted Absolute Error)
    for a given set of PID parameters. It pre-compiles Numba functions for speed.

    The class allows simulation of a SISO system under PID control with optional
    disturbances at the plant input (Z1) and at the measurement/output (Z2),
    as well as a reference (setpoint) trajectory. This enables PSO to optimize
    PID parameters considering both reference tracking and disturbance rejection.

    Attributes:
        controller (ClosedLoop): The PID controller instance.
        t0 (float): Simulation start time.
        t1 (float): Simulation end time.
        dt (float): Simulation time step.
        t_eval (np.ndarray): Array of time points for simulation.
        r_eval (np.ndarray): Evaluated reference trajectory over t_eval.
        d1_eval (np.ndarray): Evaluated disturbance trajectory at plant input (Z1) over t_eval.
        d2_eval (np.ndarray): Evaluated disturbance trajectory at measurement/output (Z2) over t_eval.
        A (np.ndarray): State-space system matrix A (contiguous for Numba).
        B (np.ndarray): State-space input vector B (contiguous for Numba).
        C (np.ndarray): State-space output vector C (contiguous for Numba).
        D (float): State-space scalar D.
        system_order (int): Order of the system (number of states).
        controller_param (dict[str, str | float | np.ndarray]): PID-specific parameters
            including derivative filter time Tf, control constraints, and anti-windup method.
        swarm_size (int): Number of particles in the PSO swarm.

    Args:
        controller (ClosedLoop): PID controller instance to be optimized.
        t0 (float): Simulation start time.
        t1 (float): Simulation end time.
        dt (float): Simulation time step.
        r (Callable[[np.ndarray], np.ndarray] | None, optional): Reference (setpoint)
            function defining the desired output over time. If None, a zero vector is used.
        d1 (Callable[[np.ndarray], np.ndarray] | None, optional): Disturbance function
            at the plant input (Z1). If None, zero disturbance is assumed.
        d2 (Callable[[np.ndarray], np.ndarray] | None, optional): Disturbance function
            at the measurement/output (Z2). If None, zero disturbance is assumed.
        swarm_size (int, optional): Number of particles in the swarm for PSO. Defaults to 40.

    Notes:
        - The class pre-compiles internal Numba functions on initialization
          to accelerate repeated evaluations of the PID controller response.
        - For purely disturbance response simulations, set r to None or a zero function.
        - The ITAE cost computed by this class can be used directly by PSO
          algorithms to optimize PID parameters.
    """
    def __init__(self, controller: ClosedLoop, t0: float, t1: float, dt: float,
                 r: Callable[[np.ndarray], np.ndarray] | None = None,
                 d1: Callable[[np.ndarray], np.ndarray] | None = None,
                 d2: Callable[[np.ndarray], np.ndarray] | None = None,
                 swarm_size: int = 40) -> None:

        self.controller = controller

        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.t_eval = np.arange(t0, t1 + dt, dt)

        if r is None:
            r = lambda t: np.zeros_like(t)

        if d1 is None:
            d1 = lambda t: np.zeros_like(t)

        if d2 is None:
            d2 = lambda t: np.zeros_like(t)

        self.r_eval = r(self.t_eval)
        self.d1_eval = d1(self.t_eval)
        self.d2_eval = d2(self.t_eval)

        # Extract state-space matrices and ensure they are contiguous for Numba
        A, B, C, D = self.controller.system.get_ABCD()
        self.A = A
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        # SISO → (n x 1)
        self.B = B.flatten()
        self.B = np.ascontiguousarray(self.B, dtype=np.float64)
        # SISO → (1 x n) wird aber in ein (n x 1) umgeschrieben (Performance)
        self.C = C.flatten()
        self.C = np.ascontiguousarray(self.C, dtype=np.float64)
        # SISO → D ist ein skalar
        self.D = float(D[0, 0])

        self.system_order = self.controller.system.get_system_order()

        self.controller_param: dict[str, str | float | np.ndarray]

        self.swarm_size = swarm_size

        # Extract PID-specific parameters
        if isinstance(self.controller, PIDClosedLoop):

            pid: PIDClosedLoop = self.controller

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

        # Pre-compile Numba functions
        start = time.time()
        X = np.array([[10, 9.6, 0.3] for _ in range(swarm_size)], dtype=np.float64)
        self.__call__(X)
        end = time.time()
        print(f"Pre-compiling: {end - start:0.6f} sec", flush=True)
        time.sleep(0.05)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the ITAE criterion for a batch of PID parameter sets.

        Args:
            X (np.ndarray): PID parameter matrix of shape (swarm_size, 3).

        Returns:
            np.ndarray: ITAE values for each particle.
        """
        start = time.time()
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if isinstance(self.controller, PIDClosedLoop):
            itae_val = _pid_pso_func(X, self.t_eval, self.dt, self.r_eval, self.d1_eval, self.d2_eval, self.A, self.B,
                                     self.C, self.D, self.system_order, self.controller_param["Tf"],
                                     self.controller_param["control_constraint"], self.controller_param["anti_windup"],
                                     self.swarm_size)

        else:
            raise NotImplementedError(f"Unsupported controller type: '{type(self.controller)}'")

        end = time.time()
        # print(f"{end - start:0.3f} sec")
        return itae_val


# =============================================================================
# Helper Functions
# =============================================================================
@njit(float64[:](float64[:, :], float64[:]), cache=True, inline="always")
def _matvec_auto(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Perform matrix-vector multiplication manually for Numba.

    Args:
        A (np.ndarray): Square matrix.
        x (np.ndarray): Vector.

    Returns:
        np.ndarray: Result of A @ x.
    """
    n = A.shape[0]
    y = np.zeros(n)

    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += A[i][j] * x[j]
        y[i] = acc

    return y


@njit(float64(float64[:], float64[:]), cache=True, inline="always")
def dot1D(x: np.ndarray, y: np.ndarray) -> float:
    """Compute dot product of two 1D vectors manually for Numba.

    Args:
        x (np.ndarray): Vector x.
        y (np.ndarray): Vector y.

    Returns:
        float: Dot product result.
    """
    acc = 0.0
    for i in range(x.shape[0]):
        acc += x[i] * y[i]
    return acc


# =============================================================================
# PID Update
# =============================================================================
@njit(
    types.UniTuple(float64, 3)(
        float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64
    ),
    cache=True, inline="always"
)
def pid_update(e: float, e_prev: float, filtered_prev: float, integral: float,
               Kp: float, Ti: float, Td: float, Tf: float, dt: float, u_min: float, u_max: float,
               anti_windup_method: int) -> tuple[float, float, float]:
    """Perform a single PID update with anti-windup.

    Args:
        e (float): Current error.
        e_prev (float): Previous error.
        filtered_prev (float): Previous filtered derivative term.
        integral (float): Current integral term.
        Kp (float): Proportional gain.
        Ti (float): Integral time constant.
        Td (float): Derivative time constant.
        Tf (float): Derivative filter time constant.
        dt (float): Simulation time step.
        u_min (float): Lower control limit.
        u_max (float): Upper control limit.
        anti_windup_method (int): Anti-windup strategy (0=Conditional, 1=Clamping).

    Returns:
        tuple[float, float, float]: Updated (control signal, integral, derivative).
    """
    # PID terms
    P = Kp * e
    de = (e - e_prev) / dt
    alpha = dt / (Tf + dt)
    d_filtered = (1 - alpha) * filtered_prev + alpha * de
    D_term = Kp * Td * d_filtered
    I_term = Kp / Ti * integral
    u_temp = P + I_term + D_term

    # --- Anti-windup handling ---
    if anti_windup_method == 0:  # Conditional
        if (u_min < u_temp < u_max) or \
                (u_temp >= u_max and e < 0.0) or \
                (u_temp <= u_min and e > 0.0):
            integral += e * dt
            I_term = Kp / Ti * integral
        u = min(max(P + I_term + D_term, u_min), u_max)

    elif anti_windup_method == 1:  # Clamping
        integral += e * dt
        I_term = Kp / Ti * integral
        I_term = min(max(I_term, u_min), u_max)
        u = min(max(P + I_term + D_term, u_min), u_max)

    else:
        u = 0.0  # Sicherheit

    return u, integral, d_filtered


# =============================================================================
# ODE Solver
# =============================================================================
@njit(float64[:](float64[:, :], float64[:], float64[:], float64, float64), cache=True, inline="always")
def rk4(A: np.ndarray, B: np.ndarray, x: np.ndarray, u: float, dt: float) -> np.ndarray:
    """Perform a single Runge–Kutta 4th order (RK4) integration step.

    Args:
        A (np.ndarray): System matrix.
        B (np.ndarray): Input matrix.
        x (np.ndarray): Current state.
        u (float): Control input.
        dt (float): Integration time step.

    Returns:
        np.ndarray: Updated state vector.
    """
    Bu = B * u
    k1 = _matvec_auto(A, x) + Bu
    k2 = _matvec_auto(A, x + 0.5 * dt * k1) + Bu
    k3 = _matvec_auto(A, x + 0.5 * dt * k2) + Bu
    k4 = _matvec_auto(A, x + dt * k3) + Bu
    x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


# =============================================================================
# ITAE Cost Function
# =============================================================================
@njit(float64(float64[:], float64[:], float64[:]), cache=True, inline="always")
def itae(t: np.ndarray, y: np.ndarray, r: np.ndarray) -> float:
    """Compute the Integral of Time-weighted Absolute Error (ITAE).

    The ITAE criterion is defined as:

        ITAE = ∫ t * |r(t) - y(t)| dt

    It penalizes long-lasting errors, promoting fast settling and
    small steady-state deviation.

    Args:
        t (np.ndarray): Time vector [s].
        y (np.ndarray): System output trajectory.
        r (np.ndarray): Reference trajectory.

    Returns:
        float: Computed ITAE value.
    """
    # berechnet delta t, beginnend mit t[0] - t[0]
    val = 0
    for i in range(1, t.shape[0]):
        val += t[i] * np.abs(r[i] - y[i]) * (t[i] - t[i - 1])

    return val


# =============================================================================
# System Response
# =============================================================================
@njit(float64[:](
    float64[:], float64, float64[:], float64[:],
    float64[:, :], float64[:], float64[:], float64
),
    cache=True, inline="always"
)
def system_response(t_eval: np.ndarray, dt: float, u_eval: np.ndarray,
                    x: np.ndarray, A: np.ndarray, B: np.ndarray,
                    C: np.ndarray, D: float) -> np.ndarray:
    """Simulate the open-loop (uncontrolled) response of a SISO system.

    This function computes the output of a single-input single-output (SISO)
    linear time-invariant (LTI) system described in state-space form:

        dx/dt = A * x + B * r
        y = C * x + D * r

    Args:
        t_eval (np.ndarray): Time vector for simulation.
        dt (float): Integration time step.
        u_eval (np.ndarray): Input trajectory (e.g., step input).
        x (np.ndarray): Initial state vector.
        A (np.ndarray): System matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (float): Feedthrough term.

    Returns:
        np.ndarray: System output trajectory y(t).
    """
    n_steps = len(t_eval)
    y_hist = np.zeros(n_steps)

    for i in range(n_steps):
        u = float(u_eval[i])

        # Zustand aktualisieren (numerische Integration)
        x = rk4(A, B, x, u, dt)

        # Ausgang berechnen
        y = dot1D(C, x)
        y_hist[i] = y + D * u

    return y_hist


@njit(float64[:](
    float64, float64, float64, float64, float64[:], float64,
    float64[:], float64[:], float64[:], float64[:], float64[:],
    int64, float64[:, :], float64[:], float64[:], float64
), cache=True, inline="always")
def pid_system_response(Kp: float, Ti: float, Td: float, Tf: float,
                        t_eval: np.ndarray, dt: float,
                        r_eval: np.ndarray, d1_eval: np.ndarray, d2_eval: np.ndarray,
                        x: np.ndarray, control_constraint: np.ndarray,
                        anti_windup_method: int,
                        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float) -> np.ndarray:
    """
    Simulate a SISO system under PID control with reference input and two disturbance inputs (Z1, Z2).

    Args:
        Kp (float): PID parameter.
        Ti (float): PID parameter.
        Td (float): PID parameter.
        Tf:(float): PID parameter.
        t_eval (np.ndarray): Time vector.
        dt (float): Simulation time step.
        r_eval (np.ndarray): Reference trajectory.
        d1_eval (np.ndarray): Disturbance at plant input (affects process input) → Z1.
        d2_eval (np.ndarray): Disturbance at measurement/output (affects feedback signal) → Z2.
        x (np.ndarray): Initial state vector.
        control_constraint (np.ndarray): Control limits [u_min, u_max].
        anti_windup_method (int): Anti-windup strategy (0=Conditional, 1=Clamping).
        A (np.ndarray): System matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (float): Feedthrough scalar.

    Returns:
        np.ndarray: Output trajectory y(t).
    """
    e_prev = 0.0
    filtered_prev = 0.0
    integral = 0.0

    u_min = float(control_constraint[0])
    u_max = float(control_constraint[1])

    n_steps = len(t_eval)
    y_hist = np.zeros(n_steps)
    y = dot1D(C, x)

    for i in range(n_steps):
        r = float(r_eval[i])
        d1 = float(d1_eval[i])  # Störung am Eingang (Z1)
        d2 = float(d2_eval[i])  # Störung im Messpfad (Z2)

        # Messsignal, das der Regler sieht (mit Messrauschen)
        y_meas = y + d2

        # Reglerfehler
        e = r - y_meas

        # PID-Regler
        u, integral, filtered_prev = pid_update(
            e, e_prev, filtered_prev, integral, Kp, Ti, Td, Tf,
            dt, u_min, u_max, anti_windup_method
        )

        # Systemzustand aktualisieren (Störung additiv zum Prozessinput)
        x = rk4(A, B, x, u + d1, dt)

        # Ausgang berechnen (reales y ohne Messrauschen)
        y = dot1D(C, x)
        y_hist[i] = y + D * (u + d1)

        e_prev = e

    return y_hist


# =============================================================================
# PSO Function
# =============================================================================
@njit(parallel=True, cache=True)
def _pid_pso_func(X: np.ndarray, t_eval: np.ndarray, dt: float, r_eval: np.ndarray, d1_eval: np.ndarray,
                  d2_eval: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float,
                  system_order: int, Tf: float, control_constraint: np.ndarray, anti_windup_method: int,
                  swarm_size: int) -> np.ndarray:
    """Compute ITAE values for multiple PID parameter sets in parallel.

    Args:
        X (np.ndarray): PID parameters (Kp, Ti, Td) for each particle, shape (swarm_size, 3).
        t_eval (np.ndarray): Time vector.
        dt (float): Simulation time step.
        r_eval (np.ndarray): Reference trajectory.
        A (np.ndarray): System matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (float): Feedthrough scalar.
        system_order (int): Order of the system.
        Tf (float): Derivative filter time constant.
        control_constraint (np.ndarray): Control limits [u_min, u_max].
        anti_windup_method (int): Anti-windup strategy (0=Conditional, 1=Clamping).
        swarm_size (int): Number of particles.

    Returns:
        np.ndarray: ITAE values for each particle.
    """
    itae_val = np.zeros(swarm_size)

    for i in prange(swarm_size):
        # Simulation des PID-regulierten Systems
        Kp = float(X[i, 0])
        Ti = float(X[i, 1])
        Td = float(X[i, 2])

        x = np.zeros(system_order, dtype=np.float64)  # Anfangszustand

        y = pid_system_response(Kp, Ti, Td, Tf, t_eval, dt, r_eval, d1_eval, d2_eval, x, control_constraint,
                                anti_windup_method, A, B, C, D)

        # ITAE-Kosten berechnen
        itae_val[i] = itae(t_eval, y, r_eval)

    return itae_val
