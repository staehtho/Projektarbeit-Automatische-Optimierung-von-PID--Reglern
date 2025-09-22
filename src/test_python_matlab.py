from Matlab import MatlabWrapper


def main():

    with MatlabWrapper() as mat:

        s = "tf('s');"
        C = "10 * (1 + 1/(5 * s) + (2 * s)/(10 * s + 1));"
        G = "1 / ((1 + s) * (2 + s));"

        mat.run_simulation("model", "yout", stop_time=20, s=s, C=C, G=G)
        mat.plot_simulation("1", "Test", show=True)


if __name__ == "__main__":
    main()
