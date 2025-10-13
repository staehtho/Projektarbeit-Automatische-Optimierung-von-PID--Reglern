from src.Matlab import MatlabInterface
from src.controlsys import Plant, PIDClosedLoop, itae
import matplotlib.pyplot as plt
import numpy as np


def main():
    num = [1]
    den = [1, 0.6, 1]
    plant: Plant = Plant(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(plant=plant, Kp=10, Ti= 3, Td=0.8)

    with MatlabInterface() as mat:
        G = f"{plant: plant}"
        mat.write_in_workspace(G=G)
        mat.run_simulation("", "yout", stop_time=20, max_step=0.001)
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']

    itae_mat = itae(t_mat, y_mat, 1)
    print(f"Matlab ITAE: {itae_mat}")

    # **************************************************************
    # Open Loop
    # **************************************************************
    t0 = 0
    t1 = 20
    dt = 0.01

    x = np.zeros(pid.plant.get_system_order())
    t_py = np.arange(t0, t1, dt)
    y_py = []

    for t in t_py:
        u = pid.controller_time_step(t, 1, 1, anti_windup=False)
        _, y = pid.plant.tf2ivp(u, t, t + dt, x, method="RK23")
        y_py.append(y)

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, legend="Matlab")
    plt.plot(t_py, y_py, legend="Python")


if __name__ == "__main__":
    main()
