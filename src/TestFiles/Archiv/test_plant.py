import numpy as np
from src.pso_pid_tuner.controlsys import System, PIDClosedLoop, itae
import matplotlib.pyplot as plt
import time


def main():
    num = [1]
    den = [1, 6, 15, 20, 15, 6, 1]
    system: System = System(num, den)

    pid: PIDClosedLoop = PIDClosedLoop(system, Kp=1.1, Ti=5.3, Td=1.7, control_constraint=[-10, 10], derivative_filter_ratio=0.01)

    start = time.time()
    t, y = pid.step_response(t1=20)
    itae_val = itae(t, y, np.ones_like(t))
    end = time.time()
    print(f"ITAE: {itae_val: 0.3f}, Duration: {(end - start):0.3f}s")

    t1, y1 = system.step_response(t1=20)

    plt.figure()
    plt.plot(t, y, label="Closed Loop")
    plt.plot(t1, y1, label="System")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
