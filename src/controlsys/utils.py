import numpy as np
from typing import Callable
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


SystemFunc = Callable[[complex | np.ndarray], complex | np.ndarray]


def bode_plot(systems: dict[str, SystemFunc], *, grid: bool = True):
    """
    Plots Bode diagrams (magnitude and phase) for multiple systems.

    Parameters
    ----------
    systems : dict[str, Callable]
        Dictionary mapping labels to transfer functions (callables).
        Each function must accept s=jω (complex or np.ndarray) as input.
    grid : bool, default=True
        Whether to display grid lines.

    Notes
    -----
    - Magnitude in dB: 20 * log10(|H(jω)|)
    - Phase in degrees, wrapped to [-180, 180]
    - Frequency axis is logarithmic (Hz)
    - Automatically adjusts y-limits to data
    """
    # Frequency axis
    f = np.logspace(-2, 3, 400)      # [Hz]
    w = 2 * np.pi * f                # rad/s
    s = 1j * w                        # Laplace variable

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    all_mag = []
    all_phase = []

    for label, sys in systems.items():
        y = sys(s)
        mag = 20 * np.log10(np.abs(y))
        phase = np.angle(y, deg=True)
        phase = (phase + 180) % 360 - 180  # wrap to [-180, 180]

        ax_mag.semilogx(f, mag, label=label)
        ax_phase.semilogx(f, phase, label=label)

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
    ax_phase.set_xlabel("Frequency [Hz]")

    # Grid
    if grid:
        ax_mag.grid(True, which="both")
        ax_phase.grid(True, which="both")

    ax_mag.legend()
    plt.tight_layout()
    plt.show()



