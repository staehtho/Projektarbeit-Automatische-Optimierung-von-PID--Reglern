from numba import njit
import numpy as np


@njit
def pid_system_response(Kp: float, Ti: float, Td: float, Tf: float,
                        t_eval: np.ndarray, dt: float, r_eval: np.ndarray,
                        x: np.ndarray, control_constraint: np.ndarray,
                        anti_windup_method: int, A: np.ndarray, B: np.ndarray,
                        C: np.ndarray, D: float) -> np.ndarray:
    """Simulate a SISO system with a PID controller including anti-windup using RK4 integration.

    This function computes the time response of a system under PID control.
    The PID controller includes proportional, integral, and derivative terms,
    with optional anti-windup strategies. The system is integrated using the
    Runge-Kutta 4th order method (RK4).

    Args:
        Kp (float): Proportional gain.
        Ti (float): Integral time constant.
        Td (float): Derivative time constant.
        Tf (float): Filter time constant for derivative action.
        t_eval (np.ndarray): Time vector for simulation.
        dt (float): Time step for integration.
        r_eval (np.ndarray): Reference signal vector.
        x (np.ndarray): Initial state vector of the plant.
        control_constraint (np.ndarray): Control limits [u_min, u_max].
        anti_windup_method (int): Anti-windup strategy (0 = conditional, 1 = clamping).
        A (np.ndarray): State-space A matrix of the system.
        B (np.ndarray): State-space B vector of the system.
        C (np.ndarray): State-space C vector of the system.
        D (float): Feedthrough term (scalar for SISO system).

    Returns:
        np.ndarray: Output signal y(t) of the system at each time step.
    """
    # Initialize PID states
    e_prev = 0.0
    filtered_prev = 0.0
    integral = 0.0

    u_min: float = float(control_constraint[0])
    u_max: float = float(control_constraint[1])

    n_steps = len(t_eval)
    y_hist = np.zeros(n_steps)

    # Compute first output
    y = C @ x

    for i in range(n_steps):
        # Compute error
        e = r_eval[i] - y

        # PID calculation
        P = Kp * e
        de = (e - e_prev) / dt
        alpha = dt / (Tf + dt)
        d_filtered = (1 - alpha) * filtered_prev + alpha * de
        D_term = Kp * Td * d_filtered
        I_term = Kp / Ti * integral
        u_temp = P + I_term + D_term

        # Anti-windup handling
        if anti_windup_method == 0:  # Conditional integration
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
            # Unsupported method, return zeros
            return y_hist

        # RK4 integration of the plant state
        temp = np.zeros_like(x)

        k1 = _matvec_auto(A, x) + B * u
        temp[:] = x + 0.5 * dt * k1
        k2 = _matvec_auto(A, temp) + B * u
        temp[:] = x + 0.5 * dt * k2
        k3 = _matvec_auto(A, temp) + B * u
        temp[:] = x + dt * k3
        k4 = _matvec_auto(A, temp) + B * u

        x += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Compute output
        y = C @ x
        y_hist[i] = y + D * u

        # Update PID internal states
        e_prev = e
        filtered_prev = d_filtered

    return y_hist


@njit
def _matvec_auto(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Perform a matrix-vector multiplication manually (optimized for Numba).

    Args:
        A (np.ndarray): Square matrix.
        x (np.ndarray): Vector to multiply.

    Returns:
        np.ndarray: Result of the multiplication y = A @ x.
    """
    n = A.shape[0]
    y = np.zeros(n)

    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x[j]
        y[i] = acc

    return y
