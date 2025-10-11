from src.controlsys import Plant, PIDClosedLoop, itae
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.figure()

    num = [1]
    den = [1, 0.6, 1]
    plant: Plant = Plant(num, den)

    pid: PIDClosedLoop = PIDClosedLoop(plant, Kp=10, Ti=3, Td=0.8, derivative_filter_ratio=0.01)
    t, y = pid.step_response(dt=0.001, method="RK23", t1=20)
    print(itae(t, y, 1))
    plt.plot(t, y)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
