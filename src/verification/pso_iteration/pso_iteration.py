from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, AntiWindup, MySolver, PerformanceIndex, PsoFunc
from pso_pid_tuner.PSO import Swarm

import numpy as np
from tqdm import tqdm
from time import time


def pascal(n: int):
    triangle = [[1]]  # Erste Zeile
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(1)
        triangle.append(row)
    return triangle[n - 1]


def main():
    dens = np.arange(1, 7)  # Pasacal-Polynome
    constrains = [2, 3, 5, 10]

    pso_iteration = 200
    swarm_size = 40
    bounds = [[0, 0.1, 0], [10, 10, 10]]
    r = lambda t: np.ones_like(t)

    # PT2-Dämpfungen
    xis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    # ====================
    # PTn-Systeme
    # ====================
    for den in dens:
        for constrain in constrains:
            plant = Plant([1], pascal(den + 1))
            pid = PIDClosedLoop(
                plant,
                Kp=10, Ti=3, Td=0.8, Tf=0.1,
                control_constraint=[-constrain, constrain],
                anti_windup_method=AntiWindup.CLAMPING
            )

            obj_func = PsoFunc(
                pid, t0=0, t1=10, dt=1e-4, r=r,
                solver=MySolver.RK4,
                performance_index=PerformanceIndex.ITAE,
                swarm_size=swarm_size, pre_compiling=False
            )

            performance_index = []
            for _ in tqdm(range(pso_iteration), f"Pascal: {den=}, {constrain=}"):
                swarm = Swarm(obj_func, swarm_size, 3, bounds)
                start_time = time()
                param, performance_idx = swarm.simulate_swarm()
                end_time = time()
                performance_index.append([performance_idx, *param, end_time - start_time])

            performance_index = np.array(performance_index)
            np.save(f"pso_iteration_pt{den}_{constrain}", performance_index)

    # ====================
    # PT2-Systeme
    # ====================
    for xi in xis:
        for constrain in constrains:
            # PT2-Übertragungsfunktion: 1 / (s^2 + 2*xi*s + 1)
            plant = Plant([1], [1, 2*xi, 1])
            pid = PIDClosedLoop(
                plant,
                Kp=10, Ti=3, Td=0.8, Tf=0.1,
                control_constraint=[-constrain, constrain],
                anti_windup_method=AntiWindup.CLAMPING
            )

            obj_func = PsoFunc(
                pid, t0=0, t1=10, dt=1e-4, r=r,
                solver=MySolver.RK4,
                performance_index=PerformanceIndex.ITAE,
                swarm_size=swarm_size, pre_compiling=False
            )

            performance_index = []
            for _ in tqdm(range(pso_iteration), f"PT2: {xi=}, {constrain=}"):
                swarm = Swarm(obj_func, swarm_size, 3, bounds)
                start_time = time()
                param, performance_idx = swarm.simulate_swarm()
                end_time = time()
                performance_index.append([performance_idx, *param, end_time - start_time])

            performance_index = np.array(performance_index)
            np.save(f"pso_iteration_pt2_xi{xi}_{constrain}", performance_index)


if __name__ == '__main__':
    main()
