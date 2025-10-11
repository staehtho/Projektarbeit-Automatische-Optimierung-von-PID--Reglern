import numpy as np
from src.controlsys import Plant, PIDClosedLoop, bode_plot
from src.Matlab import MatlabInterface
import matplotlib.pyplot as plt


def main():
    num = [1, 2]
    den = [1, 10, 2]
    plant: Plant = Plant(num, den)

    pid: PIDClosedLoop = PIDClosedLoop(plant, Kp=4.92, Ti=19.2e-3, Td=5.7e-3, derivative_filter_ratio=0.01)

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
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        F = f"s / ({pid.Tf} * s + 1);"
        mat.write_in_workspace(Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("model", "yout", stop_time=10, max_step=0.001)
        mat.plot_simulation("1", "Test", show=False)

        # t, y = plant.step_response(method="RK4")
        t_cl, y_cl = pid.step_response(t0=1, dt=0.001, method="RK4")
        plt.plot(t_cl, y_cl, label="python cl")
        plt.legend()
        # plt.plot(t, y)
        plt.show()


if __name__ == "__main__":
    main()
