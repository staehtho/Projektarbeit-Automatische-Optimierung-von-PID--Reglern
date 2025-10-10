import numpy as np
from src.controlsys import Plant, PIDClosedLoop, bode_plot
from src.Matlab import MatlabInterface
import matplotlib.pyplot as plt


def main():
    num = [1, 2, 3, 4, 5]
    den = [1, 2, 3, 4, 5, 6]
    plant: Plant = Plant(num, den)

    t, y = plant.step_response()

    plt.figure()
    plt.plot(t, y)
    plt.show()

    '''pid: PIDClosedLoop = PIDClosedLoop(plant, kp=4.92, tn=19.2e-3, tv=5.7e-3, derivative_filter_ratio=0.001)
    print(f"{pid: cl}")
    bode_plot({"test": plant.system})'''

    '''with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{pid: cl};"
        mat.write_in_workspace(s=s, G=G)
        mag, phase, omega = mat.bode("G", -2, 3, 1000)

        bode_plot({
            "Plant": plant.system,
            "ClosedLoop": pid.closed_loop,
            "Matlab": (omega, mag, phase),
        })
        Kp = pid.kp
        Td = pid.tn
        Ti = pid.tn
        F = f"s / (0.01 * {plant.t1} * s + 1);"
        mat.write_in_workspace(Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("model", "yout", stop_time=5)
        mat.plot_simulation("1", "Test", show=True)'''


if __name__ == "__main__":
    main()
