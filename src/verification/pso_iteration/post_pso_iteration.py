import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from typing import Callable


def exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def log_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.log(b * x + 1) + c


def pot_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x ** b + c


def estimate_x_for_fraction_global(model: Callable, a: float, b: float, c: float, fraction: float,
                                   x_guess: int = 10000):
    """
    Schätzt die Anzahl PSO-Läufe x für eine gegebene fraction der globalen Verbesserung.
    x_guess dient als numerische Approximation für x->∞ bei log- und pot-Fit.
    """
    y_start = model(1, a, b, c)

    # Bestimme theoretisches Minimum
    if model == exp_model:
        y_min = c
    else:  # log und pot: numerische Approximation
        y_min = model(x_guess, a, b, c)

    y_target = y_start - fraction * (y_start - y_min)

    # Invertiere je nach Modell
    if model == log_model:
        x_est = (np.exp((y_target - c) / a) - 1) / b
    elif model == exp_model:
        x_est = np.log((y_target - c) / a) / b
    elif model == pot_model:
        x_est = ((y_target - c) / a) ** (1 / b)
    else:
        raise ValueError("Modell unbekannt")
    return x_est, y_min


def fraction_after_x_iterations_global(model: Callable, a: float, b: float, c: float, x: float, y_min: float):
    """Berechnet den Prozentsatz der globalen Verbesserung nach x Iterationen."""
    y_start = model(1, a, b, c)
    y_current = model(x, a, b, c)
    fraction = (y_start - y_current) / (y_start - y_min)
    return fraction * 100  # in Prozent


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def main():
    data = np.load("pso_iteration_100.npy")
    performance_index = data[:, 0]
    time_vec = data[:, -1]

    time_mean = np.mean(time_vec)
    time_min = np.min(time_vec)
    time_max = np.max(time_vec)

    performance_index = np.sort(performance_index)[::-1]
    performance_index = performance_index[1::]
    x = np.arange(performance_index.shape[0])
    y_norm = minmax(performance_index)

    fit_dic = {
        "exp_fit": {"a": y_norm[0] - y_norm[-1], "b": -0.005, "c": y_norm[-1], "model": exp_model},
        "log_fit": {"a": y_norm[0] - y_norm[-1], "b": 0.05, "c": y_norm[-1], "model": log_model},
        "pot_fit": {"a": y_norm[0] - y_norm[-1], "b": 2, "c": y_norm[-1], "model": pot_model},
    }

    plt.figure(figsize=(8, 5))
    fraction = 0.95
    x_given = 50

    result_dict = {
        "time_statistics": {
            "mean_sec": float(time_mean),
            "min_sec": float(time_min),
            "max_sec": float(time_max)
        }
    }

    for key, val in fit_dic.items():
        model = val["model"]
        a, b, c = val["a"], val["b"], val["c"]

        param = curve_fit(model, x, y_norm, p0=[a, b, c])
        a, b, c = param[0]

        y_fit = model(x, a, b, c)
        plt.plot(x, y_fit, label=key)

        # Schätzung 95% bezogen auf globales Minimum
        x_est, y_min = estimate_x_for_fraction_global(model, a, b, c, fraction)
        perc = fraction_after_x_iterations_global(model, a, b, c, x_given, y_min)

        time_given = x_given * time_mean
        time_95 = x_est * time_mean

        result_dict[key] = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "x_95_percent_global": float(x_est),
            f"percent_after_{x_given}_iterations_global": float(perc),
            f"time_for_{x_given}_iterations_sec": float(time_given),
            f"time_for_95_percent_sec": float(time_95)
        }

    plt.plot(x, y_norm, 'k.', markeredgewidth=0.01, label='Daten')
    plt.xlabel("PSO Iterationen")
    plt.ylabel("Performance Index")
    plt.legend()
    plt.grid()
    plt.title("PSO Konvergenz der Fit-Modelle")
    plt.savefig("plot_global_fit", dpi=600)
    plt.show()

    with open("pso_fit_results.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    print("Ergebnisse wurden in 'pso_fit_results_global.json' gespeichert.")


if __name__ == "__main__":
    main()
