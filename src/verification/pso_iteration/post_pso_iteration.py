import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

def log_model(x, a, b, c):
    return a * np.log(b * x + 1) + c

def pot_model(x, a, b, c):
    return a * x**b + c

def main():

    data = np.load("pso_iteration.npy")

    performance_index = data[:, 0]
    time_vec = data[:, -1]

    time_mean = np.mean(time_vec)
    print(f"Durchschnittliche Laufzeit: {time_mean:0.3f} sec")
    print(f"Max Laufzeit: {np.max(time_vec):0.3f} sec")
    print(f"Min Laufzeit: {np.min(time_vec):0.3f} sec")

    # Daten sortieren
    performance_index = np.sort(performance_index)[::-1]

    # Daten ausdünnen
    performance_index = performance_index[::1]

    # x-Vektor
    x = np.arange(performance_index.shape[0])
    y = performance_index

    fit_dic = {
        "exp_fit": {
            "a": y[0] - y[-1],
            "b": -0.5,
            "c": y[-1],
            "model": exp_model
        },
        "log_fit": {
            "a": y[0] - y[-1],
            "b": 0.5,
            "c": y[-1],
            "model": log_model
        },
        "pot_fit": {
            "a": y[0] - y[-1],
            "b": 0.5,
            "c": y[-1],
            "model": pot_model
        }
    }

    for key, val in fit_dic.items():
        model = val["model"]
        a = val["a"]
        b = val["b"]
        c = val["c"]

        param = curve_fit(model, x, y, p0=[a, b, c])
        a = param[0][0]
        b = param[0][1]
        c = param[0][2]

        print(f"Model: {key}, a = {a: 0.3f}, b={b: 0.3f}, c={c: 0.3f}")

        y_fit = model(x, a, b, c)
        plt.plot(x, y_fit, label=key)

    plt.plot(x, y, 'k.', markeredgewidth=0.01, label='Daten')
    plt.legend()

    plt.grid()
    plt.show()

    plt.savefig("plot")


if __name__ == "__main__":
    main()
