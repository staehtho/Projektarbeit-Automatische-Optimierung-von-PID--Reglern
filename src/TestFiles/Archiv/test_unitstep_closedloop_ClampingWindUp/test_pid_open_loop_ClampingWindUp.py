from src.Matlab import MatlabInterface
from src.controlsys import System, PIDClosedLoop, itae
import matplotlib.pyplot as plt


def main():

    num = [1]
    den = [1, 2, 1]
    system: System = System(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(system=system, Kp=10, Ti=9.6, Td=0.3)

    with MatlabInterface() as mat:
        G_num = f"{pid.system: num}"
        G_den = f"{pid.system: den}"
        F_num = "[1 0];"
        F_den = "[0.01 1];"
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        mat.write_in_workspace(G_num=G_num, G_den=G_den, F_num=F_num, F_den=F_den, Kp=Kp, Td=Td, Ti=Ti)
        mat.run_simulation("closedloop_model_ClampingWindup", "yout")
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']
        '''np.savetxt('C:/Users/Flo/Desktop/pythonexporty.csv',y_mat, delimiter=',')
        np.savetxt('C:/Users/Flo/Desktop/pythonexportt.csv', t_mat, delimiter=',')'''

    # **************************************************************
    # Closed Loop Python
    # **************************************************************
    t_py, y_py, u_py, e_py = pid.step_response(t0=0, t1=10, dt=1e-4, method="ODE45")
    P_py = pid.P_hist
    I_py = pid.I_hist
    D_py = pid.D_hist

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, label="y (Matlab)")
    plt.plot(t_py, y_py, label="y (Python)")
    plt.legend()

    # **************************************************************
    # ITAE
    # **************************************************************
    itae_mat = itae(t_mat, y_mat, 1)
    itae_py = itae(t_py, y_py, 1)

    print(f"ITAE Matlab: {itae_mat}")
    print(f"ITAE Python: {itae_py}")

    print(
        f"ITAE der Schrittantwort einer Beispiels-PT2-Strecke unterscheidet sich um {abs(100 * (itae_py - itae_mat) / itae_mat)} % "
        f"zwischen Matlab und Python (ITAE in Python gerechnet)")

    plt.show()


if __name__ == "__main__":
    main()
