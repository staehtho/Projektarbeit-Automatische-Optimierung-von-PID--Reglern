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
    # https: // arxiv.org / abs / 2112.08306?utm_source = chatgpt.com
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
        - a tuple (omega, mag, phase) of precomputed Bode data.

    If `omega` is given, it defines the frequency points for all callables.

    Parameters
    ----------
    systems : dict[str, callable or tuple[np.ndarray, np.ndarray, np.ndarray]]
        Mapping from label to system.
    omega : np.ndarray, optional
        Frequency vector (rad/s) used to evaluate callables.
        If None and no precomputed omega is given, a log sweep is generated.
    low_exp, high_exp, num_points : floats/ints
        Parameters for logarithmic sweep if omega is None.
    grid : bool, default=True
        Whether to display grid lines.
    """

    # Determine omega
    if omega is None:
        # Check if any precomputed system exists
        precomputed = [sys for sys in systems.values() if isinstance(sys, tuple)]
        if precomputed:
            # Use omega of the first precomputed system
            omega = precomputed[0][0]
        else:
            # Generate log sweep
            omega = np.logspace(low_exp, high_exp, num_points)

    s = 1j * omega

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    all_mag = []
    all_phase = []

    for label, sys in systems.items():
        if callable(sys):
            # Evaluate Python-callable on omega
            y = sys(s)
            mag = 20 * np.log10(np.abs(y))
            phase = np.angle(y, deg=True)
        elif isinstance(sys, tuple) and len(sys) == 3:
            omega_sys, mag, phase = sys
            # Ensure same omega points as global omega
            if not np.array_equal(omega_sys, omega):
                # Interpolation of mag/phase onto omega
                mag = np.interp(omega, omega_sys, mag)
                phase = np.interp(omega, omega_sys, phase)
        else:
            raise TypeError(f"Unsupported system type for '{label}'")

        phase = (phase + 180) % 360 - 180  # wrap phase to [-180, 180]

        ax_mag.semilogx(omega, mag, label=label)
        ax_phase.semilogx(omega, phase, label=label)

        all_mag.append(mag)
        all_phase.append(phase)

    # Flatten arrays for min/max calculation
    all_mag = np.hstack(all_mag)
    all_phase = np.hstack(all_phase)

    # Magnitude limits & ticks
    mag_min = np.floor(all_mag.min() / 20) * 20
    mag_max = np.ceil(all_mag.max() / 20) * 20
    ax_mag.set_ylim(mag_min, mag_max)
    ax_mag.set_yticks(np.arange(mag_min, mag_max + 1, 20))
    ax_mag.set_ylabel("Magnitude [dB]")

    # Phase limits & ticks
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


