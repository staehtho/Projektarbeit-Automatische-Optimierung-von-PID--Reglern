from src.Matlab import MatlabInterface
from src.controlsys import Plant, PIDClosedLoop, itae
import matplotlib.pyplot as plt

from src.SystemsOld import SecondOrderSystem


def main():
    num = [1]
    den = [1, 0.6, 1]
    plant: Plant = Plant(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(plant=plant, Kp=10, Ti=3, Td=0.8)


    with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{plant: plant};"
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        F = f"s / ({pid.Tf} * s + 1);"
        mat.write_in_workspace(s=s, G=G, Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("closedloop_model_no_antiwindup", "yout", stop_time=20, max_step=0.001)
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']

    # **************************************************************
    # Closed Loop Python
    # **************************************************************
    t_py, y_py, u_py, e_py = pid.step_response(t0=0, t1=20, dt=0.01, method="RK23", anti_windup=False)
    P_py = pid.P_hist
    I_py = pid.I_hist
    D_py = pid.D_hist

    # t_sec, y_sec, _ = sec_sys.response()

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, label="y (Matlab)")
    plt.plot(t_py, y_py, label="y (Python)")
    # plt.plot(t_sec, y_sec, label="y (Lukas)")
    plt.legend()

    # **************************************************************
    # ITAE
    # **************************************************************
    itae_mat = itae(t_mat, y_mat, 1)
    itae_py = itae(t_py, y_py, 1)
    # itae_sec = itae(t_sec, y_sec, 1)

    print(f"ITAE Matlab: {itae_mat}")
    print(f"ITAE Python: {itae_py}")
    # print(f"ITAE Lukas: {itae_sec}")

    print(
        f"ITAE der Schrittantwort einer Beispiels-PT2-Strecke unterscheidet sich um {abs(itae_py - itae_mat)} "
        f"zwischen Matlab und Python (ITAE in Python gerechnet)")

    plt.show()


if __name__ == "__main__":
    main()
