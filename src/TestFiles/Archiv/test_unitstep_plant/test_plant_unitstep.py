from src.controlsys import System, itae
from src.Matlab import MatlabInterface
import matplotlib.pyplot as plt


def main():
    num = [1]
    den = [1, 0.6, 1]
    system: System = System(num, den)

    with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{system: system};"
        mat.write_in_workspace(s=s, G=G)
        mat.run_simulation("system_model", "yout", stop_time=20, max_step=0.001)
        mat.plot_simulation("1", "Test", show=False)
        itae_mat = itae(mat.t, mat.values['value_y']['value'], 1)
        print(f"Matlab ITAE: {itae_mat}")

    t_cl, y_cl = system.step_response(t0=0, t1=20, dt=0.001, method="RK23")
    itae_py = itae(t_cl, y_cl, 1)
    print(f"Python ITAE: {itae_py}")
    plt.plot(t_cl, y_cl, label="python cl")
    plt.legend()
    plt.show()

    print(
        f"ITAE der Schrittantwort einer Beispiels-PT2-Strecke unterscheidet sich um {abs(itae_mat - itae_py)} "
        f"zwischen Matlab und Python (ITAE in Python gerechnet)")


if __name__ == "__main__":
    main()
