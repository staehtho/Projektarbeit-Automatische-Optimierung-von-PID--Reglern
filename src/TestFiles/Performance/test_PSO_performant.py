import sys

from src.controlsys import System, PIDClosedLoop, PsoFunc
import time

from src.PSO import SwarmNew


def main():
    swarm_size = 40
    bounds = [[0, 0.1, 0], [10, 10, 10]]

    system: System = System([1], [1, 0.6, 1])

    pid: PIDClosedLoop = PIDClosedLoop(system, Kp=10, Ti=5, Td=3)

    pid.anti_windup_method = "clamping"
    obj_func = PsoFunc(pid, 0, 10, 1e-4, swarm_size=swarm_size)

    average = 0
    min_Kp = 0
    min_Ti = 0
    min_Td = 0
    min_itae = sys.float_info.max
    min_iterations = 0
    n = 25

    for i in range(n):
        # Swarm-Optimierung
        start = time.time()
        swarm = SwarmNew(obj_func, swarm_size, 3, bounds)
        terminated_swarm = swarm.simulate_swarm()
        end = time.time()

        print(f"{end - start:0.2f} sec")

        average += (end - start)

        # Best parameters from the swarm
        Kp = terminated_swarm.gBest.p_best_position[0]
        Ti = terminated_swarm.gBest.p_best_position[1]
        Td = terminated_swarm.gBest.p_best_position[2]
        itae = terminated_swarm.gBest.p_best_cost
        iterations = terminated_swarm.iterations

        if itae < min_itae:
            min_itae = itae
            min_Kp = Kp
            min_Ti = Ti
            min_Td = Td
            min_iterations = iterations

        print(f"Best swarm results: {i=} {Kp=:0.2f}, {Ti=:0.2f}, {Td=:0.2f}, {itae=:0.4f}, {iterations=}")
    average /= n
    print(f"{average=:0.2f} sec, {n=}")

    print(f"Min swarm results: {min_Kp=:0.2f}, {min_Ti=:0.2f}, {min_Td=:0.2f}, {min_itae=:0.4f}, {min_iterations=}")


if __name__ == "__main__":
    main()
