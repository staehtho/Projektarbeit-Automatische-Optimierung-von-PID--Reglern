from src.Matlab import MatlabInterface


def main():

    with MatlabInterface() as mat:

        s = "tf('s');"
        G = "(1*s^4 + 2*s^3 + 3*s^2 + 4*s + 5) / (1*s^5 + 2*s^4 + 3*s^3 + 4*s^2 + 5*s + 6);"

        mat.run_simulation("stepresponse", "yout", stop_time=10, s=s, G=G)
        mat.plot_simulation("1", "Test", show=True)


if __name__ == "__main__":
    main()
