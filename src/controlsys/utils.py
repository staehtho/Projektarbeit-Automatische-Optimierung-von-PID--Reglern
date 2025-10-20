import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt

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

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex="all")
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

    # min und max dynamisch berechnen
    mag_min = np.floor(all_mag.min() / 20) * 20
    mag_max = np.ceil(all_mag.max() / 20) * 20
    ax_mag.set_ylim(mag_min, mag_max)
    ax_mag.set_yticks(np.arange(mag_min, mag_max + 1, 20))
    ax_mag.set_ylabel("Magnitude / dB")

    ax_phase.set_ylim(-180, 180)
    ax_phase.set_yticks(np.arange(-180, 180 + 1, 45))
    ax_phase.set_ylabel("Phase / °")
    ax_phase.set_xlabel("Frequency / rad/s")

    plt.xlim(omega.min(), omega.max())

    fig.suptitle("Bode Diagramm", fontsize=14, fontweight='bold')

    if grid:
        ax_mag.grid(True, which="both")
        ax_phase.grid(True, which="both")

    ax_mag.legend()
    plt.tight_layout()
    plt.show()


def itae(t: np.ndarray, y: np.ndarray, set_point: float) -> float:
    # TODO: ITAE mit dem von Büchi vergleichen
    """Compute the Integral of Time-weighted Absolute Error (ITAE).

    The ITAE criterion is defined as:

        ITAE = ∫ t * |set_point - y(t)| dt

    It penalizes errors that persist over time, emphasizing fast
    settling and minimal steady-state error.

    Args:
        t (np.ndarray): Time vector [s].
        y (np.ndarray): System output corresponding to `t`.
        set_point (float): Desired reference value.

    Returns:
        float: The computed ITAE value.
    """
    # berechnet delta t, beginnend mit t[0] - t[0]
    dt = np.diff(t, prepend=t[0])
#    # Compute absolute error
    error = np.abs(set_point - y)
#    # Integral approximation (time-weighted)
    return float(np.sum(t * error * dt))
