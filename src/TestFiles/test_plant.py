import numpy as np
from src.controlsys import Plant, PIDClosedLoop, bode_plot
from src.Matlab import MatlabInterface
import matplotlib.pyplot as plt


def main():
    num = [1, 2]
    den = [1, 3, 2]
    plant: Plant = Plant(num, den)

    pid: PIDClosedLoop = PIDClosedLoop(plant, kp=4.92, tn=19.2e-3, tv=5.7e-3, derivative_filter_ratio=0.01)

    with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{plant: plant};"
        mat.write_in_workspace(s=s, G=G)
        # mag, phase, omega = mat.bode("G", -2, 3, 1000)

        # bode_plot({
        #    "Plant": plant.system,
        #    "ClosedLoop": pid.closed_loop,
        #    "Matlab": (omega, mag, phase),
        # })
        Kp = pid.kp
        Td = pid.tv
        #Ti = pid.tv
        Ti = pid.tn
        #F = f"s / (0.01 * {plant.t1} * s + 1);"
        F = f"s / (0.001 * s + 1);"
        mat.write_in_workspace(Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("model", "yout", stop_time=10)
        mat.plot_simulation("1", "Test", show=False)

        # t, y = plant.step_response(method="RK4")
        t_cl, y_cl = pid.step_response(t0=1, method="RK23")
        plt.plot(t_cl, y_cl, label="python cl")
        plt.legend()
        # plt.plot(t, y)
        plt.show()


if __name__ == "__main__":
    main()
