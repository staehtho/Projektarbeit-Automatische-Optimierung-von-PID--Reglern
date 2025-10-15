from src.Matlab import MatlabInterface
from src.controlsys import Plant, PIDClosedLoop, itae
import matplotlib.pyplot as plt

from src.SystemsOld import SecondOrderSystem


def main():
    num = [1, 2, 4, 5, 10]
    den = [1, 1, 30, 2, 6, 8, 10, 0.6, 10]
    plant: Plant = Plant(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(plant=plant, Kp=10, Ti=3, Td=0.8)

    sec_sys = SecondOrderSystem(Kp=10, Ti=3, Td=0.8, pos_sat=5, neg_sat=-5)

    with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{plant: plant};"
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        F = f"s / ({pid.Tf} * s + 1);"
        mat.write_in_workspace(s=s, G=G, Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("closedloop_model_nolimit", "yout", stop_time=20, max_step=0.001)
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']
        u_mat = mat.values['value_u']['value']
        e_mat = mat.values['value_r']['value']
        P_mat = mat.values['value_P']['value']
        I_mat = mat.values['value_I']['value']
        D_mat = mat.values['value_D']['value']

    # **************************************************************
    # Closed Loop Python
    # **************************************************************
    t_py, y_py, u_py, e_py = pid.step_response(t0=0, t1=20, dt=0.01, method="RK23", anti_windup=True)
    P_py = pid.P_hist
    I_py = pid.I_hist
    D_py = pid.D_hist

    # t_sec, y_sec, _ = sec_sys.response()

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, label="y (Matlab)")
    plt.plot(t_py, y_py, label="y (Python)")
    # plt.plot(t_sec, y_sec, label="y (Lukas)")
    plt.legend()

    plt.figure("Signal u")
    plt.plot(t_mat, u_mat, label="u (Matlab)")
    plt.plot(t_py, u_py, label="u (Python)")
    plt.legend()

    plt.figure("Signal r")
    plt.plot(t_mat, e_mat, label="e (Matlab)")
    plt.plot(t_py, e_py, label="e (Python)")
    plt.legend()

    plt.figure("Signal P")
    plt.plot(t_mat, P_mat, label="P (Matlab)")
    plt.plot(t_py, P_py, label="P (Python)")
    plt.legend()

    plt.figure("Signal I")
    plt.plot(t_mat, I_mat, label="I (Matlab)")
    plt.plot(t_py, I_py, label="I (Python)")
    plt.legend()

    plt.figure("Signal D")
    plt.plot(t_mat, D_mat, label="D (Matlab)")
    plt.plot(t_py, D_py, label="D (Python)")
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
        f"ITAE der Schrittantwort einer Beispiels-PT2-Strecke unterscheidet sich um {itae_py / itae_mat} "
        f"zwischen Matlab und Python (ITAE in Python gerechnet)")

    plt.show()


if __name__ == "__main__":
    main()
