import numpy as np
from scipy.signal import tf2ss
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def tf_to_ivp(num, den, u_func, t_span, x0=None, t_eval=None):
    A, B, C, D = tf2ss(num, den)
    n_states = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n_states)

    def dxdt(t, x):
        u = u_func(t)
        return (A @ x + B.flatten() * u).flatten()

    sol = solve_ivp(dxdt, t_span, x0, t_eval=t_eval)
    y = C @ sol.y + D * np.array([u_func(ti) for ti in sol.t])
    return sol.t, y.squeeze(), sol.y

# Beispiel:
num = [1, 2, 3, 4, 5]
den = [1, 2, 3, 4, 5, 6]
u = lambda t: 1.0
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)

t, y, x = tf_to_ivp(num, den, u, t_span, t_eval=t_eval)

plt.plot(t, y, label="y(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Ausgang")
plt.legend()
plt.grid(True)
plt.show()


from src.Matlab import MatlabInterface
