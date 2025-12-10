import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba.cuda import gridDim

# Linienstil für Stellgrößenbegrenzung
line_layout = {
    2: "-", 3: "--", 5: ":", 10: "-."
}

# Farben für PTn-Systeme
color_ptn = {
    "pt1": '#1f77bf',
    "pt2": '#ff7f0e',
    "pt3": '#2ca02c',
    "pt4": '#d62728',
    "pt5": '#9467bd',
    "pt6": '#8c564b'
}

# Farben für PT2-Systeme (zeta)
color_pt2 = {
    0: '#1f77bf',
    0.1: '#ff7f0e',
    0.2: '#2ca02c',
    0.3: '#d62728',
    0.4: '#9467bd',
    0.5: '#8c564b',
    0.6: '#e377c2',
    0.7: '#7f7f7f',
    1: '#bcbd22'
}


def load_data_ptn():
    return {
        2: {f"pt{i}": np.load(f"pso_iteration_pt{i}_2.npy") for i in range(1, 7)},
        3: {f"pt{i}": np.load(f"pso_iteration_pt{i}_3.npy") for i in range(1, 7)},
        5: {f"pt{i}": np.load(f"pso_iteration_pt{i}_5.npy") for i in range(1, 7)},
        10: {f"pt{i}": np.load(f"pso_iteration_pt{i}_10.npy") for i in range(1, 7)},
    }


def load_data_pt2():
    zetas = [0, 0.1]
    constrains = [2, 3, 5, 10]
    data = {}
    for z in zetas:
        data[z] = {c: np.load(f"pso_iteration_pt2_zeta{z}_{c}.npy") for c in constrains}
    return data


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def plot_ptn(data_ptn):
    plt.figure(figsize=(12, 8))
    for factor, pts in data_ptn.items():
        for pt_name, runs in pts.items():
            runs_sorted = np.sort(runs[:, 0])[::-1]
            data_norm = minmax(runs_sorted)
            plt.plot(data_norm, linestyle=line_layout[factor], color=color_ptn[pt_name], linewidth=1)

    # Legende PTn
    ptn_handler = [Line2D([0], [0], color=c, lw=2, label=name.upper()) for name, c in color_ptn.items()]
    constrain_handler = [Line2D([0], [0], color="black", linestyle=ls, lw=2, label=r"$\pm$" + str(name))
                         for name, ls in line_layout.items()]

    # Horizontale Linien
    plt.axhline(y=0.1, color='black', linestyle='--', linewidth=2)
    plt.axhline(y=0.05, color='black', linestyle='--', linewidth=2)
    plt.text(x=-20, y=0.05, s='5%', color='r', va='baseline')
    plt.text(x=-20, y=0.1, s='10%', color='r', va='baseline')

    plt.xlabel("Iterationen")
    plt.ylabel("Normiertes ITAE")

    legend1 = plt.legend(handles=ptn_handler, title=r"PT$n$", bbox_to_anchor=(1, 0.8), handlelength=3)
    plt.gca().add_artist(legend1)
    plt.legend(handles=constrain_handler, title="Stellgrössenbegrenzung", bbox_to_anchor=(1, 1), handlelength=3)

    plt.savefig("ptn_iteration.png", dpi=300, bbox_inches='tight')
    plt.grid()
    plt.show()


def plot_pt2(data_pt2):
    plt.figure(figsize=(12, 8))
    for zeta, constrains_dict in data_pt2.items():
        for constr, runs in constrains_dict.items():
            runs_sorted = np.sort(runs[:, 0])[::-1]
            data_norm = minmax(runs_sorted)
            plt.plot(data_norm, linestyle=line_layout[constr], color=color_pt2[zeta], linewidth=1)

    # Legende PT2 zeta
    pt2_handler = [Line2D([0], [0], color=c, lw=2, label=f"ζ={z}") for z, c in color_pt2.items()]
    constrain_handler = [Line2D([0], [0], color="black", linestyle=ls, lw=2, label=r"$\pm$" + str(name))
                         for name, ls in line_layout.items()]

    # Horizontale Linien
    plt.axhline(y=0.1, color='black', linestyle='--', linewidth=2)
    plt.axhline(y=0.05, color='black', linestyle='--', linewidth=2)
    plt.text(x=-20, y=0.05, s='5%', color='r', va='baseline')
    plt.text(x=-20, y=0.1, s='10%', color='r', va='baseline')

    plt.xlabel("Iterationen")
    plt.ylabel("Normiertes ITAE")

    legend1 = plt.legend(handles=pt2_handler, title="PT2", bbox_to_anchor=(1, 0.8), handlelength=3)
    plt.gca().add_artist(legend1)
    plt.legend(handles=constrain_handler, title="Stellgrössenbegrenzung", bbox_to_anchor=(1, 1), handlelength=3)

    plt.savefig("pt2_iteration.png", dpi=300, bbox_inches='tight')
    plt.grid()
    plt.show()


def main():
    data_ptn = load_data_ptn()
    data_pt2 = load_data_pt2()

    plot_ptn(data_ptn)
    plot_pt2(data_pt2)


if __name__ == "__main__":
    main()
