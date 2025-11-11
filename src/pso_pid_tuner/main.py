import math
import sys
from src.pso_pid_tuner.controlsys import System, PIDClosedLoop, PsoFunc, bode_plot, crossover_frequency, \
    smallest_root_realpart
from src.pso_pid_tuner.PSO import Swarm
from tqdm import tqdm
from config_loader import load_config, ConfigError
import matplotlib.pyplot as plt
import numpy as np


def main():
    try:
        config = load_config("../pso_pid_tuner/config/config.yaml")
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

    # generate system
    system: System = System(plant_num, plant_den)
    bounds = [[kp_min, ti_min, td_min], [kp_max, ti_max, td_max]]

    # generate closed loop
    pid: PIDClosedLoop = PIDClosedLoop(system, Kp=10, Ti=5, Td=3, control_constraint=[constraint_min, constraint_max])
    pid.anti_windup_method = anti_windup

    # dominant pole (least negative real part)
    p_dom = smallest_root_realpart(system.den)

    # corresponding time constant
    tau = 1 / abs(p_dom)

    # set filter to be much faster than plant dynamics
    pid.set_filter(Tf=tau/100)

    # define simulation horizon so the plant settles
    #TODO: funktioniert so nicht. für mehrfache polstellen m erhöht sich die zeit um faktor m. (und kompl. konj. PS mischen auch mit.
    #end_time = math.ceil(5 * tau)

    # generate function to be optimized
    r = lambda t: np.ones_like(t)
    obj_func = PsoFunc(pid, start_time, end_time, time_step, r=r, swarm_size=swarm_size)

    best_Kp = 0
    best_Ti = 0
    best_Td = 0
    best_itae = sys.float_info.max

    # einmaliges warm-up, damit JIT vor der tqdm-progressbar kompiliert
    # TODO beobachten, ob problem noch auftritt auch ohne warm up
    #_ = pid.step_response(start_time, start_time + time_step, time_step)

    # progressbar
    pbar = tqdm(range(iterations), desc="Processing", unit="step", colour="green")

    for i in pbar:
        swarm = Swarm(obj_func, swarm_size, 3, bounds)
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

    # print results
    print(f"""
    ✔ Optimization completed!

    swarm result:
       {'best_Kp':<10}= {best_Kp:8.2f}
       {'best_Ti':<10}= {best_Ti:8.2f}
       {'best_Td':<10}= {best_Td:8.2f}
       {'best_itae':<10}= {best_itae:8.4f}
    """)

    # set new found parameters
    pid.set_pid_param(Kp=best_Kp, Ti=best_Ti, Td=best_Td)

    # determine crossoverfrequency
    L = lambda s: pid.controller(s) * system.system(s)
    wc = crossover_frequency(L)
    fs = 20000  # Hz, TODO: später aus System übernehmen

    # limitations of timeconstant of filter
    Tf_max = 1 / (100 * wc)  # can't be bigger, or filter would be too slow and impact the stepresponse
    Tf_min = 10 / (np.pi * fs)  # can't be smaller, or filter would be too close to Nyquistfrequency
    pid.set_filter(Tf=Tf_max)

    print("Recommended range for the filter time constant Tf:")
    print(f"  Tf_min = {Tf_min:.6e} s   (limit imposed by the sampling frequency: {fs}Hz)")
    print(f"  Tf_max = {Tf_max:.6e} s   (limit imposed by the crossover frequency)")
    print(f"→ Choose Tf such that  {Tf_min:.6e}  ≤  Tf  ≤  {Tf_max:.6e}\n")
    print(f"  For the generated plots, the filter time constant was set to Tf_max")

    # stepresponse feedbackloop
    t_cl, y_cl = pid.step_response(
        t0=start_time,
        t1=end_time,
        dt=time_step)

    # stepresponse plant without feedback
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
