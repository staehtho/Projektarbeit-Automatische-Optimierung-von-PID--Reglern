from src.controlsys import Plant, PIDClosedLoop, itae, bode_plot
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.figure()

    num = [16000]
    den = [1, 100.8, 96, 1600]
    plant: Plant = Plant(num, den)

    bode_plot({"1": plant.system})


if __name__ == "__main__":
    main()
