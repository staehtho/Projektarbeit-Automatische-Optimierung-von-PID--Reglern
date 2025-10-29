import time

import numpy as np
import pandas as pd

from src.controlsys import System, PIDClosedLoop, PsoFunc, itae
from src.PSO import SwarmNew


class PidParameter:
    def __init__(self, num: list[float], den: list[float], Kp: float, Ti: float, Td: float,
                 t0: float, t1: float, dt: float, control_constraint: list[float],
                 identifier_name: str = "n", identifier_value: float | int = 0):

        self._system = System(num, den)
        self._pid_verification = PIDClosedLoop(self._system, Kp=Kp, Ti=Ti, Td=Td, control_constraint=control_constraint)
        self._pid_verification.anti_windup_method = "clamping"

        self._kp = Kp
        self._ti = Ti
        self._td = Td
        self._control_constraint = control_constraint
        self._identifier_name = identifier_name
        self._identifier_value = identifier_value

        t, y = self._pid_verification.step_response(t0, t1, dt)
        self._itae = itae(t, y, np.ones_like(t))

        self._itae_pso = []
        self._kp_pso = []
        self._ti_pso = []
        self._td_pso = []
        self._min_itae = 0
        self._min_kp = 0
        self._min_ti = 0
        self._min_td = 0

    def simulate_swarm(self, n: int, t0: float, t1: float, dt: float, bounds: list[list[float]] | None = None, swarm_size: int = 40) -> int:
        if bounds is None:
            bounds = [[0, 0.1, 0], [10, 10, 10]]

        obj_func = PsoFunc(self._pid_verification, t0, t1, dt, swarm_size=swarm_size)

        iterations = 0

        for _ in range(n):
            swarm = SwarmNew(obj_func, swarm_size, bounds)
            terminated_swarm = swarm.simulate_swarm()

            self._kp_pso.append(terminated_swarm.gBest.pBest_position[0])
            self._ti_pso.append(terminated_swarm.gBest.pBest_position[1])
            self._td_pso.append(terminated_swarm.gBest.pBest_position[2])
            self._itae_pso.append(terminated_swarm.gBest.pBest_cost)

            iterations += terminated_swarm.iterations

        self._min_itae = np.min(np.array(self._itae_pso))
        min_idx = np.argmin(np.array(self._itae_pso))
        self._min_kp = self._kp_pso[min_idx]
        self._min_ti = self._ti_pso[min_idx]
        self._min_td = self._td_pso[min_idx]

        return iterations

    def to_dict(self):
        """Gibt alle relevanten Ergebnisse als Dictionary zurück."""
        return {
            self._identifier_name: self._identifier_value,
            "LowerLimit": self._control_constraint[0],
            "UpperLimit": self._control_constraint[1],
            "Kp_ref": self._kp,
            "Kp_PSO": self._min_kp,
            "Ti_ref": self._ti,
            "Ti_PSO": self._min_ti,
            "Td_ref": self._td,
            "Td_PSO": self._min_td,
            "ITAE_ref": self._itae,
            "ITAE_PSO": self._min_itae
        }



def pascal(n: int):
    triangle = [[1]]  # Erste Zeile

    for i in range(1, n):
        row = [1]  # Jede Zeile beginnt mit 1
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(1)  # Jede Zeile endet mit 1
        triangle.append(row)
    return triangle[n - 1]


def pso_vs_brute_force(pid_params: list[dict], number_pso: int, swarm_size: int, t0: float, t1: float, dt: float,
                       filename: str, identifier_name: str) -> int:
    results = []
    tot_iterations = 0

    for params in pid_params:
        num = params.get("num")
        den = params.get("den")
        identifier_value = params.get(identifier_name)

        start = time.time()
        pid_parameter = PidParameter(
            num, den, params.get("Kp"), params.get("Ti"), params.get("Td"),
            t0, t1, dt, params.get("constraint"),
            identifier_name=identifier_name, identifier_value=identifier_value
        )

        tot_iterations += pid_parameter.simulate_swarm(n=number_pso, t0=t0, t1=t1, dt=dt, swarm_size=swarm_size)
        end = time.time()

        results.append(pid_parameter.to_dict())
        print(results[-1])
        print(f"{(end - start): 0.3f} sec")

    df = pd.DataFrame(results)
    df.to_csv(filename + ".csv", index=False)
    print(f"Ergebnisse in {filename}.csv gespeichert.")

    return tot_iterations


def main():
    control_constraints = [[-2, 2], [-3, 3], [-5, 5], [-10, 10]]

    ptn_pid_params = [
        # n = 1
        {"n": 1, "num": [1], "den": pascal(1 + 1), "Kp": 9.3, "Ti": 2.9, "Td": 0, "constraint": control_constraints[0]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "Kp": 9.5, "Ti": 1.9, "Td": 0, "constraint": control_constraints[1]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "Kp": 9.1, "Ti": 1.2, "Td": 0, "constraint": control_constraints[2]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "Kp": 10, "Ti": 1.0, "Td": 0, "constraint": control_constraints[3]},

        # n = 2
        {"n": 2, "num": [1], "den": pascal(2 + 1), "Kp": 10, "Ti": 9.6, "Td": 0.3, "constraint": control_constraints[0]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "Kp": 10, "Ti": 7.3, "Td": 0.3, "constraint": control_constraints[1]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "Kp": 9.6, "Ti": 5.4, "Td": 0.3, "constraint": control_constraints[2]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "Kp": 9.8, "Ti": 4.7, "Td": 0.3, "constraint": control_constraints[3]},

        # n = 3
        {"n": 3, "num": [1], "den": pascal(3 + 1), "Kp": 5.4, "Ti": 9.4, "Td": 0.7, "constraint": control_constraints[0]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "Kp": 7.0, "Ti": 10.0, "Td": 0.7, "constraint": control_constraints[1]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "Kp": 8.2, "Ti": 9.6, "Td": 0.7, "constraint": control_constraints[2]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "Kp": 10.0, "Ti": 9.7, "Td": 0.7, "constraint": control_constraints[3]},

        # n = 4
        {"n": 4, "num": [1], "den": pascal(4 + 1), "Kp": 1.9, "Ti": 5.0, "Td": 1.1, "constraint": control_constraints[0]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "Kp": 2.4, "Ti": 5.9, "Td": 1.2, "constraint": control_constraints[1]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "Kp": 2.3, "Ti": 5.7, "Td": 1.2, "constraint": control_constraints[2]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "Kp": 2.1, "Ti": 5.0, "Td": 1.1, "constraint": control_constraints[3]},

        # n = 5
        {"n": 5, "num": [1], "den": pascal(5 + 1), "Kp": 1.4, "Ti": 5.3, "Td": 1.4, "constraint": control_constraints[0]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "Kp": 1.4, "Ti": 5.2, "Td": 1.4, "constraint": control_constraints[1]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "Kp": 1.4, "Ti": 5.2, "Td": 1.4, "constraint": control_constraints[2]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "Kp": 1.4, "Ti": 5.0, "Td": 1.4, "constraint": control_constraints[3]},

        # n = 6
        {"n": 6, "num": [1], "den": pascal(6 + 1), "Kp": 1.1, "Ti": 5.5, "Td": 1.7, "constraint": control_constraints[0]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "Kp": 1.1, "Ti": 5.5, "Td": 1.7, "constraint": control_constraints[1]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "Kp": 1.1, "Ti": 5.4, "Td": 1.7, "constraint": control_constraints[2]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "Kp": 1.1, "Ti": 5.3, "Td": 1.7, "constraint": control_constraints[3]},
    ]

    D = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    pt2_pid_params = [
        # D = D[0]
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "Kp": 10, "Ti": 9.6, "Td": 0.3, "constraint": control_constraints[0]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "Kp": 10, "Ti": 6.8, "Td": 0.3, "constraint": control_constraints[1]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "Kp": 9.6, "Ti": 5.4, "Td": 0.3, "constraint": control_constraints[2]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "Kp": 9.8, "Ti": 4.7, "Td": 0.3, "constraint": control_constraints[3]},

        # D = D[1]
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "Kp": 10, "Ti": 8.6, "Td": 0.35, "constraint": control_constraints[0]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "Kp": 10, "Ti": 6.8, "Td": 0.35, "constraint": control_constraints[1]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "Kp": 10, "Ti": 5.4, "Td": 0.35, "constraint": control_constraints[2]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "Kp": 9.9, "Ti": 4.6, "Td": 0.35, "constraint": control_constraints[3]},

        # D = D[2]
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "Kp": 9.8, "Ti": 8.3, "Td": 0.4, "constraint": control_constraints[0]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "Kp": 10, "Ti": 6.9, "Td": 0.4, "constraint": control_constraints[1]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "Kp": 10, "Ti": 5.2, "Td": 0.35, "constraint": control_constraints[2]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "Kp": 9.9, "Ti": 4.9, "Td": 0.4, "constraint": control_constraints[3]},

        # D = D[3]
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "Kp": 9.9, "Ti": 8.1, "Td": 0.4, "constraint": control_constraints[0]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "Kp": 9.8, "Ti": 6.5, "Td": 0.4, "constraint": control_constraints[1]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "Kp": 9.8, "Ti": 5.3, "Td": 0.4, "constraint": control_constraints[2]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "Kp": 9.9, "Ti": 4.7, "Td": 0.4, "constraint": control_constraints[3]},

        # D = D[4]
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "Kp": 9.7, "Ti": 7.6, "Td": 0.4, "constraint": control_constraints[0]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "Kp": 10, "Ti": 6.4, "Td": 0.4, "constraint": control_constraints[1]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "Kp": 10, "Ti": 5.2, "Td": 0.4, "constraint": control_constraints[2]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "Kp": 9.9, "Ti": 4.5, "Td": 0.4, "constraint": control_constraints[3]},

        # D = D[5]
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "Kp": 9.4, "Ti": 7.3, "Td": 0.45, "constraint": control_constraints[0]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "Kp": 9.7, "Ti": 6.3, "Td": 0.45, "constraint": control_constraints[1]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "Kp": 9.9, "Ti": 5.4, "Td": 0.45, "constraint": control_constraints[2]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "Kp": 9.9, "Ti": 4.8, "Td": 0.45, "constraint": control_constraints[3]},

        # D = D[6]
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "Kp": 9.7, "Ti": 7.3, "Td": 0.45, "constraint": control_constraints[0]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "Kp": 9.9, "Ti": 6.2, "Td": 0.45, "constraint": control_constraints[1]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "Kp": 9.9, "Ti": 5.2, "Td": 0.45, "constraint": control_constraints[2]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "Kp": 9.9, "Ti": 4.6, "Td": 0.45, "constraint": control_constraints[3]},

        # D = D[7]
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "Kp": 9.9, "Ti": 7.5, "Td": 0.5, "constraint": control_constraints[0]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "Kp": 9.8, "Ti": 6.3, "Td": 0.5, "constraint": control_constraints[1]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "Kp": 10, "Ti": 5.5, "Td": 0.5, "constraint": control_constraints[2]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "Kp": 9.9, "Ti": 4.9, "Td": 0.5, "constraint": control_constraints[3]},

        # D = D[8]
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "Kp": 10, "Ti": 7.3, "Td": 0.5, "constraint": control_constraints[0]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "Kp": 10, "Ti": 6.2, "Td": 0.5, "constraint": control_constraints[1]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "Kp": 10, "Ti": 5.3, "Td": 0.5, "constraint": control_constraints[2]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "Kp": 9.9, "Ti": 4.7, "Td": 0.5, "constraint": control_constraints[3]},
    ]

    swarm_size = 40
    n = 20
    t0, t1, dt = 0, 20, 1e-4

    start = time.time()
    tot_iterations = pso_vs_brute_force(ptn_pid_params, n, swarm_size, t0, t1, dt, "ptn_pid_results", identifier_name="n")
    end = time.time()
    total_odes = tot_iterations * swarm_size * (t1 - t0) / dt
    print(f"PTn: Total ODEs solved ≈ {total_odes:.3e} in {(end - start):.2f} sec")
    '''8 Kernen und 4.2 GHZ PTn: Total ODEs solved ≈ 9.070e+10 in 885.15 sec'''

    start = time.time()
    tot_iterations = pso_vs_brute_force(pt2_pid_params, n, swarm_size, t0, t1, dt, "pt2_pid_results", identifier_name="D")
    end = time.time()
    total_odes = tot_iterations * swarm_size * (t1 - t0) / dt
    print(f"PT2: Total ODEs solved ≈ {total_odes:.3e} in {(end - start):.2f} sec")
    '''8 Kernen und 4.2 GHZ PT2: Total ODEs solved ≈ 1.511e+11 in 899.76 sec'''

if __name__ == "__main__":
    main()
