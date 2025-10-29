import time

import numpy as np
import pandas as pd

from src.controlsys import System, PIDClosedLoop, PsoFunc, itae
from src.PSO import SwarmNew


class PidParameter:
    def __init__(self, n: int, num: list[float], den: list[float], Kp: float, Ti: float, Td: float,
                 t0: float, t1: float, dt: float, control_constraint: list[float]):

        self._n = n

        self._system = System(num, den)
        self._pid_verification = PIDClosedLoop(self._system, Kp=Kp, Ti=Ti, Td=Td, control_constraint=control_constraint)
        self._pid_verification.anti_windup_method = "clamping"

        self._kp = Kp
        self._ti = Ti
        self._td = Td
        self._control_constraint = control_constraint
        t, y = self._pid_verification.step_response(t0, t1, dt)
        self._itae = itae(t, y, np.ones_like(t))

        self._itae_pso = []
        self._min_itae = 0

        self._kp_pso = []
        self._min_kp = 0

        self._ti_pso = []
        self._min_ti = 0

        self._td_pso = []
        self._min_td = 0

    def simulate_swarm(self, n: int = 15, bounds: list[list[float]] | None = None, swarm_size: int = 40):

        if bounds is None:
            bounds = [[0, 0.1, 0], [10, 10, 10]]

        obj_func = PsoFunc(self._pid_verification, 0, 10, 1e-4, swarm_size=swarm_size)

        for _ in range(n):
            swarm = SwarmNew(obj_func, swarm_size, bounds)
            terminated_swarm = swarm.simulate_swarm()

            self._kp_pso.append(terminated_swarm.gBest.pBest_position[0])
            self._ti_pso.append(terminated_swarm.gBest.pBest_position[1])
            self._td_pso.append(terminated_swarm.gBest.pBest_position[2])
            self._itae_pso.append(terminated_swarm.gBest.pBest_cost)

        self._min_itae = np.min(np.array(self._itae_pso))
        min_idx = np.argmin(np.array(self._itae_pso))
        self._min_kp = self._kp_pso[min_idx]
        self._min_ti = self._ti_pso[min_idx]
        self._min_td = self._td_pso[min_idx]

    def to_dict(self):
        """Gibt alle relevanten Ergebnisse als Dictionary zurück."""
        return {
            "n": self._n,
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


def main():
    control_constraints = [[-2, 2], [-3, 3], [-5, 5], [-10, 10]]

    pid_params = [
        # n = 1
        {"n": 1, "Kp": 9.3, "Ti": 2.9, "Td": 0, "control_constraint": control_constraints[0]},
        {"n": 1, "Kp": 9.5, "Ti": 1.9, "Td": 0, "control_constraint": control_constraints[1]},
        {"n": 1, "Kp": 9.1, "Ti": 1.2, "Td": 0, "control_constraint": control_constraints[2]},
        {"n": 1, "Kp": 10, "Ti": 1.0, "Td": 0, "control_constraint": control_constraints[3]},

        # n = 2
        {"n": 2, "Kp": 10, "Ti": 9.6, "Td": 0.3, "control_constraint": control_constraints[0]},
        {"n": 2, "Kp": 10, "Ti": 7.3, "Td": 0.3, "control_constraint": control_constraints[1]},
        {"n": 2, "Kp": 9.6, "Ti": 5.4, "Td": 0.3, "control_constraint": control_constraints[2]},
        {"n": 2, "Kp": 9.8, "Ti": 4.7, "Td": 0.3, "control_constraint": control_constraints[3]},

        # n = 3
        {"n": 3, "Kp": 5.4, "Ti": 9.4, "Td": 0.7, "control_constraint": control_constraints[0]},
        {"n": 3, "Kp": 7.0, "Ti": 10.0, "Td": 0.7, "control_constraint": control_constraints[1]},
        {"n": 3, "Kp": 8.2, "Ti": 9.6, "Td": 0.7, "control_constraint": control_constraints[2]},
        {"n": 3, "Kp": 10.0, "Ti": 9.7, "Td": 0.7, "control_constraint": control_constraints[3]},

        # n = 4
        {"n": 4, "Kp": 1.9, "Ti": 5.0, "Td": 1.1, "control_constraint": control_constraints[0]},
        {"n": 4, "Kp": 2.4, "Ti": 5.9, "Td": 1.2, "control_constraint": control_constraints[1]},
        {"n": 4, "Kp": 2.3, "Ti": 5.7, "Td": 1.2, "control_constraint": control_constraints[2]},
        {"n": 4, "Kp": 2.1, "Ti": 5.0, "Td": 1.1, "control_constraint": control_constraints[3]},

        # n = 5
        {"n": 5, "Kp": 1.4, "Ti": 5.3, "Td": 1.4, "control_constraint": control_constraints[0]},
        {"n": 5, "Kp": 1.4, "Ti": 5.2, "Td": 1.4, "control_constraint": control_constraints[1]},
        {"n": 5, "Kp": 1.4, "Ti": 5.2, "Td": 1.4, "control_constraint": control_constraints[2]},
        {"n": 5, "Kp": 1.4, "Ti": 5.0, "Td": 1.4, "control_constraint": control_constraints[3]},

        # n = 6
        {"n": 6, "Kp": 1.1, "Ti": 5.5, "Td": 1.7, "control_constraint": control_constraints[0]},
        {"n": 6, "Kp": 1.1, "Ti": 5.5, "Td": 1.7, "control_constraint": control_constraints[1]},
        {"n": 6, "Kp": 1.1, "Ti": 5.4, "Td": 1.7, "control_constraint": control_constraints[2]},
        {"n": 6, "Kp": 1.1, "Ti": 5.3, "Td": 1.7, "control_constraint": control_constraints[3]},
    ]

    results = []

    for params in pid_params:
        n = params.get("n")

        num = [1]
        den = pascal(n + 1)
        start = time.time()
        pid_parameter = PidParameter(n, num, den, params.get("Kp"), params.get("Ti"), params.get("Td"),
                                     0, 20, 1e-4, params.get("control_constraint"))
        pid_parameter.simulate_swarm(n=5)
        end = time.time()

        results.append(pid_parameter.to_dict())
        print(results[-1])

        del pid_parameter

        print(f"{(end - start): 0.3f} sec")

    # Am Ende: als CSV speichern
    df = pd.DataFrame(results)
    df.to_csv("pid_results.csv", index=False)
    print("Ergebnisse in pid_results.csv gespeichert.")


if __name__ == "__main__":
    main()
