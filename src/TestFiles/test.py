import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from src.Matlab import MatlabInterface

# Parameter
num = [3, 1]
den = [1, 0.6, 1]

# Ordnung des Systems
n = len(den) - 1


def state_derivative(t, x, num, den, u_func):
    """
    x : Zustandsvektor [y1, y2, ..., yn]
    u_func : Funktion der Zeit, liefert den Eingang u(t)
    """
    n = len(den) - 1
    x = np.array(x, dtype=float)
    y_dot = np.zeros(n)

    # dy1dt = y2, dy2dt = y3, ...
    for i in range(n - 1):
        y_dot[i] = x[i + 1]

    # letzte DGL: y^(n) = -a_{n-1} y^{(n-1)} - ... - a0 y + b0*u + ...
    den = np.array(den) / den[0]
    num = np.array(num) / den[0]
    y_coeffs = -den[1:]
    u_coeffs = np.zeros(n + 1)
    m = len(num) - 1
    u_coeffs[-(m + 1):] = num

    # Beitrag von y
    last = sum(c * x_i for c, x_i in zip(y_coeffs[::-1], x))

    # Beitrag von u(t) – hier einfache Version: nur u^0
    u_t = u_func(t)
    last += u_coeffs[-1] * u_t

    y_dot[-1] = last
    return y_dot


# Eingang u(t) = Sprung
def u_step(t):
    return 1.0

# Anfangszustände
x0 = np.zeros(n)

# Zeitvektor
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Simulation
sol = solve_ivp(state_derivative, t_span, x0, t_eval=t_eval, args=(num, den, u_step))

# Ausgabe
y = sol.y[0]  # y1 = Ausgang

#with MatlabInterface() as mat:
#    s = "tf('s');"
#    G = "tf([3, 1], [1, 0.6, 1]);"
#    mat.run_simulation("stepresponse", "yout", s=s, G=G)
#    mat.plot_simulation("test", "Step response")

plt.plot(sol.t, y)
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.title('Sprungantwort des Systems')
plt.grid(True)
plt.show()
