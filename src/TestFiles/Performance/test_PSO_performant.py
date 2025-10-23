from concurrent.futures import ProcessPoolExecutor
from src.controlsys import PsoFunc, Plant, PIDClosedLoop
import random
import time
import timeit

from src.PSO import SwarmNew
from src.controlsys.performance import _pso_func_jit


def main():
    swarm_size = 40
    bounds = [[0, 0.1, 0], [100, 10000, 10]]

    plant: Plant = Plant([1], [1, 2, 1])

    pid: PIDClosedLoop = PIDClosedLoop(plant, Kp=10, Ti=5, Td=3)

    obj_func = PsoFunc(plant, 0, 10, 1e-4, 0.01, pid.control_constraint, "clamping", swarm_size)

    average = 0
    av_Kp = 0
    av_Ti = 0
    av_Td = 0
    av_itae = 0
    av_iterations = 0
    n = 20

    for i in range(n):
        # Swarm-Optimierung
        start = time.time()
        swarm = SwarmNew(obj_func, swarm_size, bounds)
        terminated_swarm = swarm.simulate_swarm()
        end = time.time()

        print(f"{end - start:0.2f} sec")

        average += (end - start)

        # Best parameters from the swarm
        Kp = terminated_swarm.gBest.pBest_position[0]
        Ti = terminated_swarm.gBest.pBest_position[1]
        Td = terminated_swarm.gBest.pBest_position[2]
        itae = terminated_swarm.gBest.pBest_cost
        iterations = terminated_swarm.iterations
        maxStall = terminated_swarm.maxStall
        space_precision = terminated_swarm.spaceFactor
        stall_precision = terminated_swarm.convergenceFactor

        av_Kp += Kp
        av_Ti += Ti
        av_Td += Td
        av_itae += itae
        av_iterations += iterations

        print(f"Best swarm results: {i=} {Kp=:0.2f}, {Ti=:0.2f}, {Td=:0.2f}, {itae=:0.4f}, {iterations=}")
    average /= n
    print(f"{average=:0.2f} sec, {n=}")

    av_Kp /= n
    av_Ti /= n
    av_Td /= n
    av_itae /= n
    av_iterations /= n
    print(f"Average best swarm results: {n=} {av_Kp=:0.2f}, {av_Ti=:0.2f}, {av_Td=:0.2f}, {av_itae=:0.4f}, {av_iterations=}")

    print(_pso_func_jit.nopython_signatures)
    print(len(_pso_func_jit.nopython_signatures))


if __name__ == "__main__":
    main()
