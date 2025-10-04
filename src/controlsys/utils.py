import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt


def laplace_2_response(num: list[float] | np.ndarray,
                       dec: list[float] | np.ndarray,
                       *,
                       solver_methode: str = "RK23",
                       max_step: float = 0.1
                       ) -> np.ndarray:
    inverse_laplace(num=np.array(num), dec=np.array(dec))

    response: np.ndarray = calculate_response(solver_methode=solver_methode, max_step=max_step)

    return response


def inverse_laplace(num: np.ndarray, dec: np.ndarray):
    # https://arxiv.org/abs/2112.08306
    pass


def calculate_response(solver_methode: str, max_step: float) -> np.ndarray:
    pass


SystemData = Union[Callable[[np.ndarray], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


def bode_plot(
    systems: dict[str, SystemData],
    *,
    omega: np.ndarray | None = None,
    low_exp: float = -2,
    high_exp: float = 3,
    num_points: int = 400,
    grid: bool = True
):
    """
    Plot Bode diagrams (magnitude and phase) for multiple systems.

    Each system can be either:
        - a callable accepting `s = 1j*omega` and returning frequency response, or
        - a tuple `(omega, mag, phase)` of precomputed Bode data.

    Args:
        systems (dict[str, callable or tuple[np.ndarray, np.ndarray, np.ndarray]]):
            Mapping from system label to system data.
        omega (np.ndarray or None, optional):
            Frequency vector (rad/s) used to evaluate callables.
            If None and no precomputed omega is provided, a logarithmic sweep is generated.
        low_exp (float, optional):
            Lower exponent for logarithmic sweep (10**low_exp). Default is -2.
        high_exp (float, optional):
            Upper exponent for logarithmic sweep (10**high_exp). Default is 3.
        num_points (int, optional):
            Number of frequency points for logarithmic sweep. Default is 400.
        grid (bool, optional):
            If True, displays grid lines. Default is True.

    Raises:
        TypeError: If a system entry is neither callable nor a tuple of (omega, mag, phase).

    Returns:
        None
    """
    # Determine omega
    if omega is None:
        precomputed = [sys for sys in systems.values() if isinstance(sys, tuple)]
        if precomputed:
            omega = precomputed[0][0]
        else:
            omega = np.logspace(low_exp, high_exp, num_points)

    s = 1j * omega

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    all_mag = []
    all_phase = []

    for label, sys in systems.items():
        if callable(sys):
            y = sys(s)
            mag = 20 * np.log10(np.abs(y))
            phase = np.angle(y, deg=True)
        elif isinstance(sys, tuple) and len(sys) == 3:
            omega_sys, mag, phase = sys
            if not np.array_equal(omega_sys, omega):
                mag = np.interp(omega, omega_sys, mag)
                phase = np.interp(omega, omega_sys, phase)
        else:
            raise TypeError(f"Unsupported system type for '{label}'")

        phase = (phase + 180) % 360 - 180  # wrap phase to [-180, 180]

        ax_mag.semilogx(omega, mag, label=label)
        ax_phase.semilogx(omega, phase, label=label)

        all_mag.append(mag)
        all_phase.append(phase)

    all_mag = np.hstack(all_mag)
    all_phase = np.hstack(all_phase)

    mag_min = np.floor(all_mag.min() / 20) * 20
    mag_max = np.ceil(all_mag.max() / 20) * 20
    ax_mag.set_ylim(mag_min, mag_max)
    ax_mag.set_yticks(np.arange(mag_min, mag_max + 1, 20))
    ax_mag.set_ylabel("Magnitude [dB]")

    phase_min = max(np.floor(all_phase.min() / 45) * 45, -180)
    phase_max = min(np.ceil(all_phase.max() / 45) * 45, 180)
    ax_phase.set_ylim(phase_min, phase_max)
    ax_phase.set_yticks(np.arange(phase_min, phase_max + 1, 45))
    ax_phase.set_ylabel("Phase [°]")
    ax_phase.set_xlabel("Frequency [rad/s]")

    if grid:
        ax_mag.grid(True, which="both")
        ax_phase.grid(True, which="both")

    ax_mag.legend()
    plt.tight_layout()
    plt.show()



