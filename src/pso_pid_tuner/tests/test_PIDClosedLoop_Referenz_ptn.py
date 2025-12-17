import numpy as np
import pytest

from pso_pid_tuner.controlsys import Plant, PIDClosedLoop, itae, AntiWindup

# ============================================================
# Allgemeine Parameter
# ============================================================

T_START = 0.0
T_END = 20.0
DT = 1e-4
ITAE_TOL = 1e-4


def step_reference(t):
    return np.ones_like(t)


# ============================================================
# Vollständig abgetippte Testtabelle (ITAE_ref_python)
# ============================================================

TEST_CASES = [
    # ===================== PT1 =====================
    dict(pt=1, num=[1], den=[1, 1], cc=(-2, 2),  Kp=9.3, Ti=2.9, Td=0.0, itae=0.072192514),
    dict(pt=1, num=[1], den=[1, 1], cc=(-3, 3),  Kp=9.5, Ti=1.9, Td=0.0, itae=0.030090101),
    dict(pt=1, num=[1], den=[1, 1], cc=(-5, 5),  Kp=9.1, Ti=1.2, Td=0.0, itae=0.015674508),
    dict(pt=1, num=[1], den=[1, 1], cc=(-10,10), Kp=10.0,Ti=1.0, Td=0.0, itae=0.009985),

    # ===================== PT2 =====================
    dict(pt=2, num=[1], den=[1, 2, 1], cc=(-2, 2),  Kp=10.0, Ti=9.6, Td=0.3, itae=0.606225934),
    dict(pt=2, num=[1], den=[1, 2, 1], cc=(-3, 3),  Kp=10.0, Ti=7.3, Td=0.3, itae=0.365212279),
    dict(pt=2, num=[1], den=[1, 2, 1], cc=(-5, 5),  Kp=9.6, Ti=5.4, Td=0.3, itae=0.246438833),
    dict(pt=2, num=[1], den=[1, 2, 1], cc=(-10,10), Kp=9.8, Ti=4.7, Td=0.3, itae=0.196893469),

    # ===================== PT3 =====================
    dict(pt=3, num=[1], den=[1, 3, 3, 1], cc=(-2, 2),  Kp=5.4, Ti=9.4, Td=0.7, itae=1.869075706),
    dict(pt=3, num=[1], den=[1, 3, 3, 1], cc=(-3, 3),  Kp=7.0, Ti=10.0,Td=0.7, itae=1.365173985),
    dict(pt=3, num=[1], den=[1, 3, 3, 1], cc=(-5, 5),  Kp=8.2, Ti=9.6, Td=0.7, itae=1.011914671),
    dict(pt=3, num=[1], den=[1, 3, 3, 1], cc=(-10,10), Kp=10.0,Ti=9.7, Td=0.7, itae=0.811847813),

    # ===================== PT4 =====================
    dict(pt=4, num=[1], den=[1, 4, 6, 4, 1], cc=(-2, 2),  Kp=1.9, Ti=5.0, Td=1.1, itae=4.711766234),
    dict(pt=4, num=[1], den=[1, 4, 6, 4, 1], cc=(-3, 3),  Kp=2.4, Ti=5.9, Td=1.2, itae=4.511774557),
    dict(pt=4, num=[1], den=[1, 4, 6, 4, 1], cc=(-5, 5),  Kp=2.3, Ti=5.7, Td=1.2, itae=4.433785069),
    dict(pt=4, num=[1], den=[1, 4, 6, 4, 1], cc=(-10,10), Kp=2.1, Ti=5.0, Td=1.1, itae=4.245377375),

    # ===================== PT5 =====================
    dict(pt=5, num=[1], den=[1, 5, 10, 10, 5, 1], cc=(-2, 2),  Kp=1.4, Ti=5.3, Td=1.4, itae=9.963459933),
    dict(pt=5, num=[1], den=[1, 5, 10, 10, 5, 1], cc=(-3, 3),  Kp=1.4, Ti=5.2, Td=1.4, itae=9.765665891),
    dict(pt=5, num=[1], den=[1, 5, 10, 10, 5, 1], cc=(-5, 5),  Kp=1.4, Ti=5.2, Td=1.4, itae=9.612845068),
    dict(pt=5, num=[1], den=[1, 5, 10, 10, 5, 1], cc=(-10,10), Kp=1.4, Ti=5.0, Td=1.4, itae=9.127438867),

    # ===================== PT6 =====================
    dict(pt=6, num=[1], den=[1, 6, 15, 20, 15, 6, 1], cc=(-2, 2),  Kp=1.1, Ti=5.5, Td=1.7, itae=16.78246818),
    dict(pt=6, num=[1], den=[1, 6, 15, 20, 15, 6, 1], cc=(-3, 3),  Kp=1.1, Ti=5.5, Td=1.7, itae=16.6418386),
    dict(pt=6, num=[1], den=[1, 6, 15, 20, 15, 6, 1], cc=(-5, 5),  Kp=1.1, Ti=5.4, Td=1.7, itae=16.25512263),
    dict(pt=6, num=[1], den=[1, 6, 15, 20, 15, 6, 1], cc=(-10,10), Kp=1.1, Ti=5.3, Td=1.7, itae=15.66437821),
]


# ============================================================
# Parametrisierter Test
# ============================================================

@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: f"pt{c['pt']}_{c['cc']}")
def test_system_response(case):

    plant = Plant(case["num"], case["den"])

    pid = PIDClosedLoop(
        plant,
        Kp=case["Kp"],
        Ti=case["Ti"],
        Td=case["Td"],
        control_constraint=list(case["cc"]),
        anti_windup_method=AntiWindup.CLAMPING,
    )

    t, y = pid.system_response(T_START, T_END, DT, r=step_reference)

    itae_val = itae(t, y, step_reference(t))

    assert abs(itae_val - case["itae"]) < ITAE_TOL
