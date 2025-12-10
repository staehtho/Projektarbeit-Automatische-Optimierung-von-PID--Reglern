from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, AntiWindup, MySolver, PerformanceIndex, PsoFunc
from pso_pid_tuner.PSO import Swarm

import numpy as np
import sys

epsilon = 0.05


def test_Pid_pso():
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

    min_performance_index = sys.float_info.max

    for _ in range(10):
        swarm = Swarm(obj_func, swarm_size, 3, bounds)
        swarm_result, performance_index = swarm.simulate_swarm()

        if performance_index < min_performance_index:
            min_performance_index = performance_index

    assert abs(min_performance_index - 0.239) < epsilon
