from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, AntiWindup, MySolver, PerformanceIndex, PsoFunc
from pso_pid_tuner.PSO import Swarm

import numpy as np
from tqdm import tqdm
from time import time


def main():
    plant = Plant([1], [1, 2, 1])
    pid = PIDClosedLoop(plant, Kp=10, Ti=3, Td=0.8, Tf=0.1,
                        control_constraint=[-5, 5],
                        anti_windup_method=AntiWindup.CLAMPING)

    r = lambda t: np.ones_like(t)

    swarm_size = 40

    obj_func = PsoFunc(pid, t0=0, t1=10, dt=1e-4, r=r,
                       solver=MySolver.RK4,
                       performance_index=PerformanceIndex.ITAE,
                       swarm_size=swarm_size, pre_compiling=False)

    bounds = [[0, 0.1, 0], [10, 10, 10]]

    performance_index = []

    pso_iteration = 1000

    for _ in tqdm(range(pso_iteration)):
        swarm = Swarm(obj_func, swarm_size, 3, bounds)

        start_time = time()
        param, performance_idx = swarm.simulate_swarm()
        end_time = time()

        performance_index.append([performance_idx, param[0], param[1], param[2], end_time - start_time])

    performance_index = np.array(performance_index)
    np.save("pso_iteration", performance_index)


if __name__ == '__main__':
    main()
