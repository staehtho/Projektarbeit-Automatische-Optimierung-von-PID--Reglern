from controlsys import Plant, PIDClosedLoop, bode_plot
from Matlab import MatlabInterface


def main():
    plant: Plant = Plant([1], [1, 1, 1])

    pid: PIDClosedLoop = PIDClosedLoop(plant, kp=4.92, tn=19.2e-3, tv=5.7e-3, derivative_filter_ratio=0.001)

    with MatlabInterface() as mat:
        s = "tf('s');"
        G = "1 / (s^2 + s + 1);"
        mag, phase, omega = mat.bode("G", -2, 3, 1000, s=s, G=G)

        bode_plot({
            "Plant": plant.system,
            "ClosedLoop": pid.closed_loop,
            "Matlab": (omega, mag, phase)
        })


if __name__ == "__main__":
    main()
