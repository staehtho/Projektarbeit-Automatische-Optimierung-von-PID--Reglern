from controlsys import Plant, PIDClosedLoop, bode_plot
from Matlab import MatlabInterface


def main():
    plant: Plant = Plant([4, 1], [1, 1, 1])

    pid: PIDClosedLoop = PIDClosedLoop(plant, kp=4.92, tn=19.2e-3, tv=5.7e-3, derivative_filter_ratio=0.001)

    with MatlabInterface() as mat:
        s = "tf('s');"
        R = f"{pid: mat};"
        G = f"{plant: mat};"
        mat.write_in_workspace(s=s, R=R, G=G)
        mag, phase, omega = mat.bode("G", -2, 3, 1000)
        mag_pid, phase_pid, omega_pid = mat.bode("R", -2, 3, 1000)

        bode_plot({
            "Plant": plant.system,
            "ClosedLoop": pid.closed_loop,
            "Matlab": (omega, mag, phase),
            "MatlabCL": (omega_pid, mag_pid, phase_pid)
        })


if __name__ == "__main__":
    main()
