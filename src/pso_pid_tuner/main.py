import sys
from controlsys import Plant, PIDClosedLoop, PsoFunc, smallest_root_realpart
from PSO import Swarm
from tqdm import tqdm
from config_loader import load_config, ConfigError
import numpy as np
from report_generator.report_generator import report_generator


def main():

    print("Loading Configuration..")

    try:
        config = load_config()
        print("Configuration loaded successfully!")
    except ConfigError as e:
        print("error in configuration!:")
        print(e)
        input("Press Enter to exit…")
        return

    plant_num = config["syste"]["plant"]["numerator"]
    plant_den = config["system"]["plant"]["denominator"]

    start_time = config["system"]["simulation_time"]["start_time"]
    end_time = config["system"]["simulation_time"]["end_time"]
    time_step = config["system"]["simulation_time"]["time_step"]

    anti_windup = config["system"]["anti_windup"]

    excitation_target = config["system"]["excitation_target"]

    constraint_min = config["system"]["control_constraint"]["min_constraint"]
    constraint_max = config["system"]["control_constraint"]["max_constraint"]

    performance_index = config["system"]["performance_index"]

    swarm_size = config["pso"]["swarm_size"]
    iterations = config["pso"]["iterations"]

    kp_min = config["pso"]["bounds"]["kp_min"]
    kp_max = config["pso"]["bounds"]["kp_max"]
    ti_min = config["pso"]["bounds"]["ti_min"]
    ti_max = config["pso"]["bounds"]["ti_max"]
    td_min = config["pso"]["bounds"]["td_min"]
    td_max = config["pso"]["bounds"]["td_max"]

    # generate plant
    plant: Plant = Plant(plant_num, plant_den)
    bounds = [[kp_min, ti_min, td_min], [kp_max, ti_max, td_max]]

    # generate closed loop
    pid: PIDClosedLoop = PIDClosedLoop(plant, Kp=10, Ti=5, Td=3,
                                       control_constraint=[constraint_min, constraint_max],
                                       anti_windup_method=anti_windup)

    # dominant pole (least negative real part)
    p_dom = smallest_root_realpart(plant.den)

    # corresponding time constant
    # TODO: 0 abfangen
    t_dom = 1 / abs(p_dom)

    # set filter to be much faster than plant dynamics
    pid.set_filter(Tf=t_dom/100)
    # TODO: 0 abfangen

    # define simulation horizon so the plant settles
    # TODO: funktioniert so nicht. für mehrfache polstellen m erhöht sich die zeit um faktor m. (und kompl. konj. PS mischen auch mit.
    # end_time = math.ceil(5 * t_dom)

    # generate function to be optimized
    match excitation_target:
        case "reference":
            r = lambda t: np.ones_like(t)
            l = lambda t: np.zeros_like(t)
            n = lambda t: np.zeros_like(t)
        case "input_disturbance":
            r = lambda t: np.zeros_like(t)
            l = lambda t: np.ones_like(t)
            n = lambda t: np.zeros_like(t)
        case "measurement_disturbance":
            r = lambda t: np.zeros_like(t)
            l = lambda t: np.zeros_like(t)
            n = lambda t: np.ones_like(t)
        case _:
            r = lambda t: np.zeros_like(t)
            l = lambda t: np.zeros_like(t)
            n = lambda t: np.zeros_like(t)

    obj_func = PsoFunc(pid, start_time, end_time, time_step, r=r, l=l, n=n,
                       performance_index=performance_index, swarm_size=swarm_size)

    best_Kp = 0
    best_Ti = 0
    best_Td = 0
    best_performance_index = sys.float_info.max

    # progressbar
    pbar = tqdm(range(iterations), desc="Processing", unit="step", colour="green")

    for _ in pbar:
        swarm = Swarm(obj_func, swarm_size, 3, bounds)
        terminated_swarm = swarm.simulate_swarm()

        # Best parameters from the swarm
        Kp = terminated_swarm.gBest.p_best_position[0]
        Ti = terminated_swarm.gBest.p_best_position[1]
        Td = terminated_swarm.gBest.p_best_position[2]
        performance_index_val = terminated_swarm.gBest.p_best_cost

        if performance_index_val < best_performance_index:
            best_performance_index = performance_index_val
            best_Kp = Kp
            best_Ti = Ti
            best_Td = Td

    data = {
        "best_Kp": best_Kp,
        "best_Ti": best_Ti,
        "best_Td": best_Td,
        "performance_index": performance_index,
        "best_performance_index": best_performance_index,

        "plant": plant,
        "pid": pid,

        "start_time": start_time,
        "end_time": end_time,
        "time_step": time_step,
        "excitation_target": excitation_target,

        "plant_num": plant_num,
        "plant_den": plant_den,
    }

    report_generator(data)


if __name__ == "__main__":
    main()
