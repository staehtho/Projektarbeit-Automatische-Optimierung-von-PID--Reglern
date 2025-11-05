import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss
from scipy.integrate import solve_ivp
from typing import Any


def tf_to_ivp_step_with_ABCD(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, u: float, t0: float, t1: float, x0: float):
    def dx_dt(t, x):
        return (A @ x + B.flatten() * u).flatten()

    sol: Any = solve_ivp(dx_dt, (t0, t1), x0, t_eval=[t1], method="RK23", max_step=0.1)
    y = (C @ sol.y[:, -1] + D * u).item()
    return sol.y[:, -1], y


# --- Systemdefinition ---
num = [1, 2, 3, 4, 5]
den = [1, 2, 3, 4, 5, 6]
A: np.ndarray
B: np.ndarray
C: np.ndarray
D: np.ndarray
A, B, C, D = tf2ss(num, den)
x = np.zeros(A.shape[0])

# --- Simulationseinstellungen ---
t_span = (0, 10)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)

# --- Eingangssignal ---
u = lambda t: 1

# --- Initialisierung ---
y_hist = []
u_hist = []

# --- Zeitschleife ---
for t in t_eval:
    uu = u(t)
    x, y = tf_to_ivp_step_with_ABCD(A, B, C, D, uu, t, t + dt, x)
    y_hist.append(y)

    u_hist.append(uu)

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(t_eval, y_hist, label="Systemausgang y(t)")
plt.plot(t_eval, u_hist, "--", label="Eingang u(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
