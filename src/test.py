import numpy as np
from scipy.signal import tf2ss
from scipy.integrate import solve_ivp

def laplace_to_rhs(num_coeffs, den_coeffs):
    """
    num_coeffs: Zählerkoeffizienten [b0, b1, ...] (höchster Grad zuerst)
    den_coeffs: Nennerkoeffizienten [a0, a1, ...] (höchster Grad zuerst)

    Gibt eine Funktion rhs(t, y_vec, x_vec) zurück:
        - y_vec = [y, y', y'', ...] Länge = Ordnung des Systems
        - x_vec = [x, x', x'', ...] Länge = Zählergrad +1
        - rhs(t, y_vec, x_vec) gibt dy_n/dt = y^(n) zurück
    """
    # Systemordnung
    n = len(den_coeffs) - 1
    m = len(num_coeffs) - 1

    # Koeffizienten normalisieren (y'' + ...)
    a_coeffs = np.array(den_coeffs, dtype=float) / den_coeffs[0]
    b_coeffs = np.array(num_coeffs, dtype=float) / den_coeffs[0]

    def rhs(t, y_vec, x_vec):
        """
        y_vec: [y, y', ..., y^(n-1)]
        x_vec: [x, x', ..., x^(m)]
        Gibt dy_n/dt zurück
        """
        # Linke Seite: y^(n) = - (a_{n-1} y^(n-1) + ... + a0 y) + RHS
        y_terms = -np.sum(a_coeffs[1:] * y_vec[::-1])

        # Rechte Seite: b0 x^(m) + ... + b_m x
        # Pad x_vec falls zu kurz
        if len(x_vec) < len(b_coeffs):
            x_vec_padded = np.pad(x_vec, (len(b_coeffs) - len(x_vec), 0), 'constant')
        else:
            x_vec_padded = x_vec[-len(b_coeffs):]

        x_terms = np.sum(b_coeffs * x_vec_padded[::-1])

        return y_terms + x_terms

    return rhs


# ===========================
# Beispiel: H(s) = 1 / (s+1)^2
num_coeffs = [1]  # Zähler: 1
den_coeffs = [1, 2, 1]  # Nenner: s^2 +2 s +1

rhs_func = laplace_to_rhs(num_coeffs, den_coeffs)

# Test: y_vec = [y, y'], x_vec = [x, x']
y_vec = np.array([0.0, 0.0])  # y, y'
x_vec = np.array([1.0, 0.0])  # x, x'

dy_n_dt = rhs_func(0.0, y_vec, x_vec)
print("dy_n/dt =", dy_n_dt)

y0 = [0.0, 0.0]  # Anfangswerte
# Zeitbereich
t_span = (0, 10)
sol = solve_ivp(rhs_func, t_span, y0, max_step=0.1, method='RK23', vectorized=False)
