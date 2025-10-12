import numpy as np
from src.controlsys import Plant, PIDClosedLoop, bode_plot, itae
from src.Matlab import MatlabInterface
import matplotlib.pyplot as plt


def main():
    num = [1]
    den = [1, 0.6, 1]
    plant: Plant = Plant(num, den)

    pid: PIDClosedLoop = PIDClosedLoop(plant, Kp=10, Ti=3, Td=0.8, derivative_filter_ratio=0.01)

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
        mat.run_simulation("model", "yout", stop_time=20, max_step=0.001)
        mat.plot_simulation("1", "Test", show=False)
        print(f"Matlab ITEA: {itae(mat.t, mat.values['value_y']['value'], 1)}")

        # t, y = plant.step_response(method="RK4")
        t_cl, y_cl = pid.step_response(t0=0, t1=20, dt=0.001, method="RK23")
        print(f"Python ITEA: {itae(t_cl, y_cl, 1)}")
        plt.plot(t_cl, y_cl, label="python cl")
        plt.legend()
        # plt.plot(t, y)
        plt.show()


if __name__ == "__main__":
    main()
