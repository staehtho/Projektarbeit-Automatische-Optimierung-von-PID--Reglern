import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba.cuda import gridDim

# Linienstil für Stellgrößenbegrenzung
line_layout = {
    2: "-", 3: "--", 5: ":", 10: "-."
}

# Harmonische Farben für PTn-Systeme
color_ptn = {
    "pt1": '#4e79a7',  # sanftes Blau
    "pt2": '#f28e2b',  # warmes Orange
    "pt3": '#59a14f',  # frisches Grün
    "pt4": '#e15759',  # klares Rot
    "pt5": '#b07aa1',  # weiches Violett
    "pt6": '#9c755f'   # erdiges Braun
}


# Harmonische Farbskala für PT2-Systeme (zeta)
color_pt2 = {
    0.0:  '#4e79a7',
    0.1:  '#76a5d2',
    0.2:  '#a2c1e6',
    0.3:  '#59a14f',
    0.4:  '#8cc07a',
    0.5:  '#b4d8a3',
    0.6:  '#f28e2b',
    0.7:  '#f5b66e',
    1.0:  '#e15759'
}


def load_data_ptn():
    return {
        2: {f"pt{i}": np.load(f"pso_iteration_pt{i}_2.npy") for i in range(1, 7)},
        3: {f"pt{i}": np.load(f"pso_iteration_pt{i}_3.npy") for i in range(1, 7)},
        5: {f"pt{i}": np.load(f"pso_iteration_pt{i}_5.npy") for i in range(1, 7)},
        10: {f"pt{i}": np.load(f"pso_iteration_pt{i}_10.npy") for i in range(1, 7)},
    }


def load_data_pt2():
    zetas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
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

    legend1 = plt.legend(handles=ptn_handler, title="PTn", bbox_to_anchor=(1, 0.8), handlelength=3)
    plt.gca().add_artist(legend1)
    plt.legend(handles=constrain_handler, title="Stellgrössenbegrenzung", bbox_to_anchor=(1, 1), handlelength=3)

    plt.grid()
    plt.savefig("ptn_iteration.png", dpi=300, bbox_inches='tight')
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

    plt.grid()
    plt.savefig("pt2_iteration.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    data_ptn = load_data_ptn()
    data_pt2 = load_data_pt2()

    # Alle Werte in eine Liste sammeln (nur erste Spalte)
    all_values = []

    # PTn-Daten
    for subdict in data_ptn.values():
        for arr in subdict.values():
            all_values.extend(arr[:, -1])  # nur die erste Spalte

    # PT2-Daten
    for subdict in data_pt2.values():
        for arr in subdict.values():
            all_values.extend(arr[:, -1])  # nur die erste Spalte

    all_values = np.array(all_values)

    # Globale Kennzahlen berechnen
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    global_median = np.median(all_values)
    global_mean = np.mean(all_values)

    print(f"Globale PSO-Kennzahlen über alle Durchläufe:")
    print(f"  Minimum: {global_min:.4f}")
    print(f"  Maximum: {global_max:.4f}")
    print(f"  Median : {global_median:.4f}")
    print(f"  Mean   : {global_mean:.4f}")
    print(f"  n      : {all_values.shape[0]}")

    plot_ptn(data_ptn)
    plot_pt2(data_pt2)


if __name__ == "__main__":
    main()
