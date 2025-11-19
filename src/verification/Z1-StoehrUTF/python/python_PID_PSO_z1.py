import time
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.pso_pid_tuner.controlsys import Plant, PIDClosedLoop, PsoFunc, itae
from src.pso_pid_tuner.PSO import Swarm


class PidParameter:
    def __init__(self, num: list[float], den: list[float], Kp: float, Ti: float, Td: float,
                 t0: float, t1: float, dt: float, control_constraint: list[float],
                 identifier_name: str = "n", identifier_value: float | int = 0):

        self._plant = Plant(num, den)
        self._pid_verification = PIDClosedLoop(self._plant, Kp=Kp, Ti=Ti, Td=Td, control_constraint=control_constraint)
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

    def simulate_swarm(self, n: int, t0: float, t1: float, dt: float,
                       bounds: list[list[float]] | None = None, swarm_size: int = 40) -> int:
        if bounds is None:
            bounds = [[0, 0.1, 0], [10, 10, 10]]
        l = lambda t: np.ones_like(t)
        obj_func = PsoFunc(self._pid_verification, t0, t1, dt, l=l, swarm_size=swarm_size, pre_compiling=False)

        iterations = 0

        for _ in range(n):
            swarm = Swarm(obj_func, swarm_size, 3, bounds)
            terminated_swarm = swarm.simulate_swarm()

            self._kp_pso.append(terminated_swarm.gBest.p_best_position[0])
            self._ti_pso.append(terminated_swarm.gBest.p_best_position[1])
            self._td_pso.append(terminated_swarm.gBest.p_best_position[2])
            self._itae_pso.append(terminated_swarm.gBest.p_best_cost)

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
            "Kp_PSO": self._min_kp,
            "Ti_PSO": self._min_ti,
            "Td_PSO": self._min_td,
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

    for params in tqdm(pid_params, desc=identifier_name, unit="step"):
        num = params.get("num")
        den = params.get("den")
        identifier_value = params.get(identifier_name)

        pid_parameter = PidParameter(
            num, den, 10, 3, 1,
            t0, t1, dt, params.get("constraint"),
            identifier_name=identifier_name, identifier_value=identifier_value
        )

        tot_iterations += pid_parameter.simulate_swarm(n=number_pso, t0=t0, t1=t1, dt=dt, swarm_size=swarm_size)

        results.append(pid_parameter.to_dict())

    df = pd.DataFrame(results)
    df.to_csv(filename + ".csv", index=False)
    print(f"Ergebnisse in {filename}.csv gespeichert.")

    return tot_iterations


def main():
    control_constraints = [[-2, 2], [-3, 3], [-5, 5], [-10, 10]]

    ptn_pid_params = [
        # n = 1
        {"n": 1, "num": [1], "den": pascal(1 + 1), "constraint": control_constraints[0]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "constraint": control_constraints[1]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "constraint": control_constraints[2]},
        {"n": 1, "num": [1], "den": pascal(1 + 1), "constraint": control_constraints[3]},

        # n = 2
        {"n": 2, "num": [1], "den": pascal(2 + 1), "constraint": control_constraints[0]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "constraint": control_constraints[1]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "constraint": control_constraints[2]},
        {"n": 2, "num": [1], "den": pascal(2 + 1), "constraint": control_constraints[3]},

        # n = 3
        {"n": 3, "num": [1], "den": pascal(3 + 1), "constraint": control_constraints[0]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "constraint": control_constraints[1]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "constraint": control_constraints[2]},
        {"n": 3, "num": [1], "den": pascal(3 + 1), "constraint": control_constraints[3]},

        # n = 4
        {"n": 4, "num": [1], "den": pascal(4 + 1), "constraint": control_constraints[0]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "constraint": control_constraints[1]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "constraint": control_constraints[2]},
        {"n": 4, "num": [1], "den": pascal(4 + 1), "constraint": control_constraints[3]},

        # n = 5
        {"n": 5, "num": [1], "den": pascal(5 + 1), "constraint": control_constraints[0]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "constraint": control_constraints[1]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "constraint": control_constraints[2]},
        {"n": 5, "num": [1], "den": pascal(5 + 1), "constraint": control_constraints[3]},

        # n = 6
        {"n": 6, "num": [1], "den": pascal(6 + 1), "constraint": control_constraints[0]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "constraint": control_constraints[1]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "constraint": control_constraints[2]},
        {"n": 6, "num": [1], "den": pascal(6 + 1), "constraint": control_constraints[3]},
    ]

    D = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    pt2_pid_params = [
        # D = D[0]
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "constraint": control_constraints[0]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "constraint": control_constraints[1]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "constraint": control_constraints[2]},
        {"D": D[0], "num": [1], "den": [1, 2*D[0], 1], "constraint": control_constraints[3]},

        # D = D[1]
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "constraint": control_constraints[0]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "constraint": control_constraints[1]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "constraint": control_constraints[2]},
        {"D": D[1], "num": [1], "den": [1, 2*D[1], 1], "constraint": control_constraints[3]},

        # D = D[2]
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "constraint": control_constraints[0]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "constraint": control_constraints[1]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "constraint": control_constraints[2]},
        {"D": D[2], "num": [1], "den": [1, 2*D[2], 1], "constraint": control_constraints[3]},

        # D = D[3]
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "constraint": control_constraints[0]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "constraint": control_constraints[1]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "constraint": control_constraints[2]},
        {"D": D[3], "num": [1], "den": [1, 2*D[3], 1], "constraint": control_constraints[3]},

        # D = D[4]
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "constraint": control_constraints[0]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "constraint": control_constraints[1]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "constraint": control_constraints[2]},
        {"D": D[4], "num": [1], "den": [1, 2*D[4], 1], "constraint": control_constraints[3]},

        # D = D[5]
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "constraint": control_constraints[0]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "constraint": control_constraints[1]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "constraint": control_constraints[2]},
        {"D": D[5], "num": [1], "den": [1, 2*D[5], 1], "constraint": control_constraints[3]},

        # D = D[6]
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "constraint": control_constraints[0]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "constraint": control_constraints[1]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "constraint": control_constraints[2]},
        {"D": D[6], "num": [1], "den": [1, 2*D[6], 1], "constraint": control_constraints[3]},

        # D = D[7]
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "constraint": control_constraints[0]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "constraint": control_constraints[1]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "constraint": control_constraints[2]},
        {"D": D[7], "num": [1], "den": [1, 2*D[7], 1], "constraint": control_constraints[3]},

        # D = D[8]
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "constraint": control_constraints[0]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "constraint": control_constraints[1]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "constraint": control_constraints[2]},
        {"D": D[8], "num": [1], "den": [1, 2*D[8], 1], "constraint": control_constraints[3]},
    ]

    swarm_size = 40
    n = 20
    t0, t1, dt = 0, 20, 1e-4

    start = time.time()
    tot_iterations = pso_vs_brute_force(ptn_pid_params, n, swarm_size, t0, t1, dt, "ptn_pid_results", identifier_name="n")
    end = time.time()
    total_odes = tot_iterations * swarm_size * (t1 - t0) / dt
    print(f"PTn: Total ODEs solved ≈ {total_odes:.3e} in {(end - start):.2f} sec")
    '''12 Kernen und 1.2 GHZ PTn: Total ODEs solved ≈ 7.743e+10 in 2786.46 sec'''

    start = time.time()
    tot_iterations = pso_vs_brute_force(pt2_pid_params, n, swarm_size, t0, t1, dt, "pt2_pid_results", identifier_name="D")
    end = time.time()
    total_odes = tot_iterations * swarm_size * (t1 - t0) / dt
    print(f"PT2: Total ODEs solved ≈ {total_odes:.3e} in {(end - start):.2f} sec")
    '''12 Kernen und 1.2 GHZ PT2: Total ODEs solved ≈ 1.455e+11 in 2300.98 sec'''


if __name__ == "__main__":
    main()
