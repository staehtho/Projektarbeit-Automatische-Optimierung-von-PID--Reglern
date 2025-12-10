from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, itae, AntiWindup

import numpy as np


def test_system_response():
    r = lambda t: np.ones_like(t)

    itea_tol = 1e-4

    test_param = {
        "pt1": {
            "num": [1], "den": [1, 1], "Kp": 10, "Ti": 3, "Td": 0.3, "itae_ref": 0.31452269404161004
        },
        "pt2": {
            "num": [1], "den": [1, 2, 1], "Kp": 10, "Ti": 3, "Td": 0.3, "itae_ref": 1.0359858615494943
        },
        "pt3": {
            "num": [1], "den": [1, 3, 3, 1], "Kp": 10, "Ti": 3, "Td": 0.3, "itae_ref": 7.035901384691646
        },
        "pt4": {
            "num": [1], "den": [1, 4, 6, 4, 1], "Kp": 10, "Ti": 3, "Td": 0.3, "itae_ref": 42.65000765255898
        },
        "pt5": {
            "num": [1], "den": [1, 5, 10, 10, 5, 1], "Kp": 10, "Ti": 3, "Td": 0.3, "itae_ref": 68.26284513795815
        }
    }

    for key, value in test_param.items():

        plant = Plant(value["num"], value["den"])

        pid = PIDClosedLoop(plant, Kp=value["Kp"], Ti=value["Ti"], Td=value["Td"],
                            control_constraint=[-5, 5], anti_windup_method=AntiWindup.CLAMPING)

        t_step, y_step = pid.system_response(0, 10, 1e-4, r=r)

        assert (abs(itae(t_step, y_step, r(t_step)) - value["itae_ref"])) < itea_tol
