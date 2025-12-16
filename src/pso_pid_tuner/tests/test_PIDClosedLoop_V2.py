from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, itae, AntiWindup

import numpy as np


def test_system_response():
    r = lambda t: np.ones_like(t)

    itea_tol = 1e-4

    test_param = {
        "pt1": {
            "num": [1], "den": [1, 1], "Kp": 9.3, "Ti": 2.9, "Td": 0, "itae_ref": 0.0721925136135404
        },
        "pt2": {
            "num": [1], "den": [1, 2, 1], "Kp": 10, "Ti": 9.6, "Td": 0.3, "itae_ref": 0.606225934463596
        },
        "pt3": {
            "num": [1], "den": [1, 3, 3, 1], "Kp": 5.4, "Ti": 9.4, "Td": 0.7, "itae_ref": 1.86907570608801
        },
        "pt4": {
            "num": [1], "den": [1, 4, 6, 4, 1], "Kp": 1.9, "Ti": 5, "Td": 1.1, "itae_ref": 4.71176623377312
        },
        "pt5": {
            "num": [1], "den": [1, 5, 10, 10, 5, 1], "Kp": 1.4, "Ti": 5.3, "Td": 1.4, "itae_ref": 9.96345993303873
        }
    }

    for key, value in test_param.items():

        plant = Plant(value["num"], value["den"])

        pid = PIDClosedLoop(plant, Kp=value["Kp"], Ti=value["Ti"], Td=value["Td"],
                            control_constraint=[-2, 2], anti_windup_method=AntiWindup.CLAMPING)

        t_step, y_step = pid.system_response(0, 20, 1e-4, r=r)

        assert (abs(itae(t_step, y_step, r(t_step)) - value["itae_ref"])) < itea_tol
