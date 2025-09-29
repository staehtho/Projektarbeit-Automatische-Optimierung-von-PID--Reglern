from Matlab import MatlabWrapper


def main():

    with MatlabWrapper() as mat:

        s = "tf('s');"
        G = "1 / ((s + 1)^2);"

        mat.run_simulation("stepresponse", "yout", stop_time=10, s=s, G=G)
        mat.plot_simulation("1", "Test", show=True)


if __name__ == "__main__":
    main()
