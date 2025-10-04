from controlsys import Plant, PIDClosedLoop, bode_plot
from Matlab import MatlabInterface


def main():
    plant: Plant = Plant([4, 1], [1, 1, 2])

    pid: PIDClosedLoop = PIDClosedLoop(plant, kp=4.92, tn=19.2e-3, tv=5.7e-3, derivative_filter_ratio=0.001)

    with MatlabInterface() as mat:
        s = "tf('s');"
        G = f"{pid: mat};"
        mat.write_in_workspace(s=s, G=G)
        mag, phase, omega = mat.bode("G", -2, 3, 1000)

        bode_plot({
            "Plant": plant.system,
            "ClosedLoop": pid.closed_loop,
            "Matlab": (omega, mag, phase),
        })
        Kp = pid.kp
        Td = pid.tn
        Ti = pid.tn
        F = f"s / (0.01 * {plant.t1} * s + 1);"
        mat.write_in_workspace(Kp=Kp, Td=Td, Ti=Ti, F=F)
        mat.run_simulation("model", "yout", stop_time=5)
        mat.plot_simulation("1", "Test", show=True)


if __name__ == "__main__":
    main()
