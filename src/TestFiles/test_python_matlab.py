from src.controlsys import Plant
from src.Matlab import MatlabInterface
import numpy as np


def main():

    with MatlabInterface() as mat:
        num = np.array([1, 2])
        dec = np.array([1, 2, 3])
        plant = Plant(num=num, dec=dec)
        s = "tf('s');"
        G = f"{plant: plant};"

        mat.run_simulation("stepresponse", "yout", stop_time=10, s=s, G=G)
        mat.plot_simulation("1", "Test", show=True)


if __name__ == "__main__":
    main()
