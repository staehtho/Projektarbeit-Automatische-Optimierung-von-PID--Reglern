from src.Matlab import MatlabInterface
from src.controlsys import Plant, PIDClosedLoop, itae
import matplotlib.pyplot as plt
import numpy as np

def main():
    num = [1]
    den = [1, 2, 1]
    plant: Plant = Plant(num, den)
    pid: PIDClosedLoop = PIDClosedLoop(plant=plant, Kp=10, Ti=9.6, Td=0.3)

    with MatlabInterface() as mat:
        G_num = f"{pid.plant: num}"
        G_den = f"{pid.plant: den}"
        F_num = f"{pid: tf_num}"
        F_den = f"{pid: tf_den}"
        Kp = pid.Kp
        Td = pid.Td
        Ti = pid.Ti
        mat.write_in_workspace(G_num=G_num, G_den=G_den, F_num=F_num, F_den=F_den, Kp=Kp, Td=Td, Ti=Ti)
        mat.run_simulation("closedloop_model_ClampingWindup", "yout")
        t_mat = mat.t
        y_mat = mat.values['value_y']['value']

    #np.savetxt("C:/Users/Flo/Desktop/mat_t.csv", t_mat, delimiter=",", header="t_mat", comments='')

    plt.figure("Unit step Matlab vs Python")
    plt.plot(t_mat, y_mat, label="y (Matlab)")

    itae_mat = itae(t_mat, y_mat, 1)
    print(f"ITAE Matlab: {itae_mat}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
