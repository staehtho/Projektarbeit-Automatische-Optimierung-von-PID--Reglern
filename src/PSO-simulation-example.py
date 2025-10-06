# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:12:35 2024

@author: lukas
"""

# import the system to simulate
from SystemsOld import *

from PSO import Swarm


def outputSwarm(swarm, particle_space):
    print()
    print("Position")
    print(swarm.gBest.pBest_position)
    print("ITAE")
    print(swarm.gBest.pBest_cost)
    print("W")
    print(swarm.options[0])
    print("Particle Space in %")
    print(particle_space)


def main():
    swarm_size = 40
    bounds = [[0, 0.1, 0], [100, 10000, 10]]
    obj_func = MotorSystem.systemResponseX

    # Swarm-Optimierung
    swarm = Swarm(obj_func, swarm_size, bounds)
    terminated_swarm = swarm.simulate_swarm(outputSwarm)

    # Best parameters from the swarm
    Kp = terminated_swarm.gBest.pBest_position[0]
    Ki = terminated_swarm.gBest.pBest_position[1]
    Kd = terminated_swarm.gBest.pBest_position[2]
    itae = terminated_swarm.gBest.pBest_cost
    iterations = terminated_swarm.iterations
    maxStall = terminated_swarm.maxStall
    space_precision = terminated_swarm.spaceFactor
    stall_precision = terminated_swarm.convergenceFactor
    values = (Kp, Ki, Kd, itae, iterations, maxStall, space_precision, stall_precision)
    print("Best swarm results:", values)

    # ------------------------------
    # 1️⃣ Plot ohne Optimierung
    # ------------------------------
    print("Plot ohne Optimierung:")
    MotorSystem.plotSystemResponseX([59.8449, 2043.5, 5.72668], pos_sat=500, neg_sat=20, control=False)

    # ------------------------------
    # 2️⃣ Plot mit Swarm-Optimierung
    # ------------------------------
    print("Plot mit Swarm-Optimierung:")
    MotorSystem.plotSystemResponseX([Kp, Ki, Kd], pos_sat=500, neg_sat=20, control=True)


if __name__ == '__main__':
    main()
