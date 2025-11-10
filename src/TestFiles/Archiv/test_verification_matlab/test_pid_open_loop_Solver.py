import numpy as np

from src.Matlab import MatlabInterface
from src.pso_pid_tuner.controlsys import System, PIDClosedLoop, itae
import matplotlib.pyplot as plt


def main():
    num = [1]
    den = [1, 0.6, 1]
    system: System = System(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(system=system, Kp=10, Ti=3, Td=0.8)

    with MatlabInterface() as mat:
        G_num = f"{pid.system: num}"
        G_den = f"{pid.system: den}"
        F_num = f"{pid: tf_num}"
        F_den = f"{pid: tf_den}"
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        mat.write_in_workspace(G_num=G_num, G_den=G_den, F_num=F_num, F_den=F_den, Kp=Kp, Td=Td, Ti=Ti)
        mat.run_simulation("closedloop_model_ClampingWindup", "yout")
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']

    # **************************************************************
    # Closed Loop Python
    # **************************************************************

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, label="y (Matlab)")

    itae_mat = itae(t_mat, y_mat, np.ones_like(t_mat))
    print(f"ITAE Matlab: {itae_mat}")

    t_py, y_py = pid.step_response(t0=0, t1=10, dt=1e-4)

    plt.plot(t_py, y_py, label=f"y (Python)")

    # **************************************************************
    # ITAE
    # **************************************************************
    itae_py = itae(t_py, y_py, np.ones_like(t_py))

    print(f"ITAE Python: {itae_py}")

    '''print(
        f"ITAE der Schrittantwort einer Beispiels-PT2-Strecke unterscheidet sich um {abs(100 * (itae_py - itae_mat) / itae_mat)} % "
        f"zwischen Matlab und Python (ITAE in Python gerechnet)")'''

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
