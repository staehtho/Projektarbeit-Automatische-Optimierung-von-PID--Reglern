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
# Vollständig abgetippte Testtabelle (PT2 schwingungsfähig)
# ============================================================

TEST_CASES = [
    # D = 1.0
    dict(D=1.0, cc=(-2, 2),  Kp=10.0, Ti=9.6, Td=0.3,  itae=0.606225934),
    dict(D=1.0, cc=(-3, 3),  Kp=10.0, Ti=7.3, Td=0.3,  itae=0.36521227),
    dict(D=1.0, cc=(-5, 5),  Kp=9.6,  Ti=5.4, Td=0.3,  itae=0.246438833),
    dict(D=1.0, cc=(-10,10), Kp=9.8,  Ti=4.7, Td=0.3,  itae=0.196893469),

    # D = 0.7
    dict(D=0.7, cc=(-2, 2),  Kp=10.0, Ti=8.6, Td=0.35, itae=0.499781098),
    dict(D=0.7, cc=(-3, 3),  Kp=10.0, Ti=6.8, Td=0.35, itae=0.328496184),
    dict(D=0.7, cc=(-5, 5),  Kp=10.0, Ti=5.4, Td=0.35, itae=0.230316461),
    dict(D=0.7, cc=(-10,10), Kp=9.9,  Ti=4.6, Td=0.35, itae=0.189176625),

    # D = 0.6
    dict(D=0.6, cc=(-2, 2),  Kp=9.8,  Ti=8.3, Td=0.4,  itae=0.482159077),
    dict(D=0.6, cc=(-3, 3),  Kp=10.0, Ti=6.9, Td=0.4,  itae=0.323387778),
    dict(D=0.6, cc=(-5, 5),  Kp=10.0, Ti=5.2, Td=0.35, itae=0.230813788),
    dict(D=0.6, cc=(-10,10), Kp=9.9,  Ti=4.9, Td=0.4,  itae=0.192686296),

    # D = 0.5
    dict(D=0.5, cc=(-2, 2),  Kp=9.9,  Ti=8.1, Td=0.4,  itae=0.439313379),
    dict(D=0.5, cc=(-3, 3),  Kp=9.8,  Ti=6.5, Td=0.4,  itae=0.310257853),
    dict(D=0.5, cc=(-5, 5),  Kp=9.8,  Ti=5.3, Td=0.4,  itae=0.227781758),
    dict(D=0.5, cc=(-10,10), Kp=9.9,  Ti=4.7, Td=0.4,  itae=0.186949375),

    # D = 0.4
    dict(D=0.4, cc=(-2, 2),  Kp=9.7,  Ti=7.6, Td=0.4,  itae=0.425283026),
    dict(D=0.4, cc=(-3, 3),  Kp=10.0, Ti=6.4, Td=0.4,  itae=0.29966027),
    dict(D=0.4, cc=(-5, 5),  Kp=10.0, Ti=5.2, Td=0.4,  itae=0.225765466),
    dict(D=0.4, cc=(-10,10), Kp=9.9,  Ti=4.5, Td=0.4,  itae=0.190494121),

    # D = 0.3
    dict(D=0.3, cc=(-2, 2),  Kp=9.4,  Ti=7.3, Td=0.45, itae=0.418799505),
    dict(D=0.3, cc=(-3, 3),  Kp=9.7,  Ti=6.3, Td=0.45, itae=0.293531882),
    dict(D=0.3, cc=(-5, 5),  Kp=9.9,  Ti=5.4, Td=0.45, itae=0.221624995),
    dict(D=0.3, cc=(-10,10), Kp=9.9,  Ti=4.8, Td=0.45, itae=0.189844066),

    # D = 0.2
    dict(D=0.2, cc=(-2, 2),  Kp=9.7,  Ti=7.3, Td=0.45, itae=0.385999129),
    dict(D=0.2, cc=(-3, 3),  Kp=9.9,  Ti=6.2, Td=0.45, itae=0.291286953),
    dict(D=0.2, cc=(-5, 5),  Kp=9.9,  Ti=5.2, Td=0.45, itae=0.223639402),
    dict(D=0.2, cc=(-10,10), Kp=9.9,  Ti=4.6, Td=0.45, itae=0.191575729),

    # D = 0.1
    dict(D=0.1, cc=(-2, 2),  Kp=9.9,  Ti=7.5, Td=0.5,  itae=0.375774415),
    dict(D=0.1, cc=(-3, 3),  Kp=9.8,  Ti=6.3, Td=0.5,  itae=0.284026204),
    dict(D=0.1, cc=(-5, 5),  Kp=10.0, Ti=5.5, Td=0.5,  itae=0.22233142),
    dict(D=0.1, cc=(-10,10), Kp=9.9,  Ti=4.9, Td=0.5,  itae=0.193878237),

    # D = 0.0
    dict(D=0.0, cc=(-2, 2),  Kp=10.0, Ti=7.3, Td=0.5,  itae=0.368077704),
    dict(D=0.0, cc=(-3, 3),  Kp=10.0, Ti=6.2, Td=0.5,  itae=0.281296114),
    dict(D=0.0, cc=(-5, 5),  Kp=10.0, Ti=5.3, Td=0.5,  itae=0.213796203),
    dict(D=0.0, cc=(-10,10), Kp=9.9,  Ti=4.7, Td=0.5,  itae=0.193722741),
]


# ============================================================
# Parametrisierter Test
# ============================================================

@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=lambda c: f"PT2_D{c['D']}_cc{c['cc']}"
)
def test_pt2_oscillatory(case):

    # PT2 mit Dämpfung D
    plant = Plant(
        num=[1],
        den=[1, 2 * case["D"], 1],
    )

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
