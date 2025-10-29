import timeit
from src.controlsys import System, PIDClosedLoop, PsoFunc
import numpy as np


def main():

    n = 10
    num = [1]
    den = [1, 2, 1]

    t0, t1 = 0, 10
    dt = 1e-4
    swarm_size = 40
    system = System(num, den)
    pid = PIDClosedLoop(system, Kp=10, Ti=9.6, Td=0.3, control_constraint=[-3, 3])
    pid.anti_windup_method = "clamping"
    
    func = PsoFunc(pid, t0, t1, dt, swarm_size=swarm_size)
    X = np.array([[10, 9.6, 0.3] for _ in range(swarm_size)], dtype=np.float64)
    average = timeit.timeit(lambda: func(X), number=n) / n
    print(f"Average with jit: {average: 0.6f} sec")
    print(func(np.array([[10, 7.3, 0.3] for _ in range(swarm_size)], dtype=np.float64)))

    '''Average without jit: 17.804126444999127 sec
    Average with jit: 0.1514863559999503 sec
    Average with jit (Mat multiplikation neu): 0.0732363700051792 sec'''

    '''Average mit Funktionsaufruf swarm_size = 40, n = 20: 0.973579 sec'''
    '''Average mit system response und itae integriert swarm_size = 40, n = 20: 0.355098 sec'''
    '''Average alle Funktionen in einem Modul und inline="always" swarm_size = 40, n = 20: 0.051706 sec'''
    '''Average eigenes Skalarprodukt swarm_size = 40, n = 20: 0.039476 sec'''


if __name__ == "__main__":
    main()
