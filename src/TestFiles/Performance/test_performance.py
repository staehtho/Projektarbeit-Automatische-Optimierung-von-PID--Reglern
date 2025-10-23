import timeit
from src.controlsys import Plant, PIDClosedLoop, PsoFunc
import numpy as np


def main():

    n = 10
    num = [1]
    den = [1, 2, 1]

    t0, t1 = 0, 10
    dt = 1e-4
    swarm_size = 40
    plant = Plant(num, den)
    pid = PIDClosedLoop(plant, Kp=10, Ti=9.6, Td=0.3)
    Tf = pid.Tf
    
    func = PsoFunc(plant, t0, t1, dt, Tf, pid.control_constraint, "clamping", swarm_size)

    average = timeit.timeit(lambda: func(np.array([[10, 9.6, 0.3] for _ in range(swarm_size)], dtype=np.float64)), number=n) / n
    print(f"Average with jit: {average} sec")
    print(func(np.array([[10, 9.6, 0.3] for _ in range(swarm_size)], dtype=np.float64)))

    '''Average without jit: 17.804126444999127 sec
    Average with jit: 0.1514863559999503 sec
    Average with jit (Mat multiplikation neu): 0.0732363700051792 sec'''


if __name__ == "__main__":
    main()
