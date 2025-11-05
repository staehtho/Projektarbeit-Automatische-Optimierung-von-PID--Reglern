import sys
from src.controlsys import System, PIDClosedLoop, PsoFunc, bode_plot, crossover_frequency
from src.PSO import SwarmNew
from tqdm import tqdm
from config_loader import load_config, ConfigError
import matplotlib.pyplot as plt
import numpy as np

config = load_config("../config/config.yaml")


def main():
    try:
        print("Configuration loaded successfully!")
    except ConfigError as e:
        print("error in configuration!:")
        print(e)
        return

    plant_num = config["system"]["plant"]["numerator"]
    plant_den = config["system"]["plant"]["denominator"]

    start_time = config["system"]["simulation_time"]["start_time"]
    end_time = config["system"]["simulation_time"]["end_time"]
    time_step = config["system"]["simulation_time"]["time_step"]

    anti_windup = config["system"]["anti_windup"]

    constraint_min = config["system"]["control_constraint"]["min_constraint"]
    constraint_max = config["system"]["control_constraint"]["max_constraint"]

    swarm_size = config["pso"]["swarm_size"]
    iterations = config["pso"]["iterations"]

    kp_min = config["pso"]["bounds"]["kp_min"]
    kp_max = config["pso"]["bounds"]["kp_max"]
    ti_min = config["pso"]["bounds"]["ti_min"]
    ti_max = config["pso"]["bounds"]["ti_max"]
    td_min = config["pso"]["bounds"]["td_min"]
    td_max = config["pso"]["bounds"]["td_max"]

    system: System = System(plant_num, plant_den)
    bounds = [[kp_min, ti_min, td_min], [kp_max, ti_max, td_max]]

    pid: PIDClosedLoop = PIDClosedLoop(system, Kp=10, Ti=5, Td=3, control_constraint=[constraint_min, constraint_max])
    pid.anti_windup_method = anti_windup
    obj_func = PsoFunc(pid, start_time, end_time, time_step, swarm_size=swarm_size)

    best_Kp = 0
    best_Ti = 0
    best_Td = 0
    best_itae = sys.float_info.max

    # einmaliges warm-up, damit JIT vor der tqdm-progressbar kompiliert
    _ = pid.step_response(start_time, start_time + time_step, time_step)

    # progressbar
    pbar = tqdm(range(iterations), desc="Processing", unit="step", colour="green")

    for i in pbar:
        swarm = SwarmNew(obj_func, swarm_size, 3, bounds)
        terminated_swarm = swarm.simulate_swarm()

        # Best parameters from the swarm
        Kp = terminated_swarm.gBest.p_best_position[0]
        Ti = terminated_swarm.gBest.p_best_position[1]
        Td = terminated_swarm.gBest.p_best_position[2]
        itae = terminated_swarm.gBest.p_best_cost

        if itae < best_itae:
            best_itae = itae
            best_Kp = Kp
            best_Ti = Ti
            best_Td = Td

    #print results
    print(f"\n✔ Optimization completed!\n\nswarm result:\n   {best_Kp=   :0.2f}\n   {best_Ti=   :0.2f}\n   {best_Td=   :0.2f}\n   {best_itae= :0.4f}\n")
    pid._kp = best_Kp
    pid._ti = best_Ti
    pid._td = best_Td

    # Durchtrittsfrequenz bestimmen
    L = lambda s: pid.controller(s) * system.system(s)
    wc = crossover_frequency(L)  # oder crossover_frequency falls schon definiert
    fs = 20000  # Hz, TODO: später aus System übernehmen

    # Zeitkonstanten-Grenzen berechnen:
    Tf_max = 1 / (10 * wc)  # darf nicht größer sein (Filter nicht "zu langsam")
    Tf_min = 10 / (np.pi * fs)  # darf nicht kleiner sein (nicht zu nahe an Nyquist)

    print("Recommended range for the filter time constant Tf:")
    print(f"  Tf_min = {Tf_min:.6e} s   (limit imposed by the sampling frequency: {fs}Hz)")
    print(f"  Tf_max = {Tf_max:.6e} s   (limit imposed by the crossover frequency)")
    print(f"→ Choose Tf such that  {Tf_min:.6e}  ≤  Tf  ≤  {Tf_max:.6e}\n")

    # Geregelte Schrittantwort
    t_cl, y_cl = pid.step_response(
        t0=start_time,
        t1=end_time,
        dt=time_step)

    # Ungeregelte Schrittantwort
    t_ol, y_ol = system.step_response(
        t0=start_time,
        t1=end_time,
        dt=time_step)

    # Bode
    systems_for_bode = {
        "Open Loop": system.system,
        "Closed Loop": pid.closed_loop}

    # Plot
    plt.figure()
    plt.plot(t_ol, y_ol, label="Open Loop")
    plt.plot(t_cl, y_cl, label="Closed Loop")
    plt.xlabel("time / s")
    plt.ylabel("output")
    plt.title("stepresponse")
    plt.grid(True)
    plt.legend()

    bode_plot(systems_for_bode)
    plt.show()

if __name__ == "__main__":
    main()
